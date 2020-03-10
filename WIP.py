# MSubs 2020
import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt


__FRAME_DIM = (1920, 1080)
__FRAME_CENTER = ( __FRAME_DIM[0] // 2, __FRAME_DIM[1] // 2) # width x height = x, y

def video_to_photo():
	# Change to camera or file input if necessary
	cam = cv2.VideoCapture("/media/sf_VM_shared/MSub/test_clip.mp4") 
	  
	try: 
		  
		# creating a folder named data 
		if not os.path.exists('data'): 
			os.makedirs('data') 
	  
	# if not created then raise error 
	except OSError: 
		print ('Error: Creating directory of data') 
	  
	# frame 
	currentframe = 0
	  
	while(True): 
		  
		# reading from frame 
		ret,frame = cam.read() 
	  
		if ret: 
			# if video is still left continue creating images 
			name = './data/frame' + str(currentframe) + '.jpg'
			print ('Creating...' + name) 
	  
			# writing the extracted images 
			cv2.imwrite(name, frame) 
	  
			# increasing counter so that it will 
			# show how many frames are created 
			currentframe += 1
		else: 
			break
	  
	# Release all space and windows once done 
	cam.release() 
	cv2.destroyAllWindows() 

# Preprocessing, currently reads and saves to file. Copy line_segent_detect main into here and feed 
# into it edges instead of reading from file. Can combine this function with video-to-photo for frame-by-frame
# camera processing.
def contour_detect():
	cv2.namedWindow('modified', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('modified', 500, 500)

	for value in range(300, 600):
		print(value)
		img = cv2.imread("data/frame%d.jpg"%value)
		img = sharp_and_smooth(img)

		LAB_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

		white_balanced_LAB = white_balance(LAB_img)
		#enhanced_LAB = CLAHE_enhance(white_balanced_LAB)

		orange_mask = LAB_mask(white_balanced_LAB)
		converted_orange_mask = cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR)

		converted_BGR = cv2.cvtColor(white_balanced_LAB, cv2.COLOR_LAB2BGR)


		cv2.imshow("modified", np.hstack([converted_BGR, converted_orange_mask]))
		
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break



#------------------In-use functions Below ------------------------------------------------



# returns the smoothed image
def sharp_and_smooth(img):
	# Unsharp masking technique for shaperning edges
	# Helps make edges more 'connected'
	#gaussian_blurred = cv2.GaussianBlur(img, (11,11), 9);
	#sharpened = cv2.addWeighted(img, 1.5, gaussian_blurred, -0.5, 0);
	
	# Remove salt and pepper noise
	median_blurred = cv2.medianBlur(img, 11)
	combine_blurred = cv2.GaussianBlur(median_blurred, (5,5), 0 )

	return combine_blurred

# Returns the enhanced LAB image
def CLAHE_enhance(LAB_img):
	CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	LAB_img[:,:,0] = CLAHE.apply(LAB_img[:,:,0])

	return LAB_img

# Performs some morphological operations to improve mask
def mask_processing(mask):
	kernel = np.ones((5,5),np.uint8)
	closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	dilation = cv2.dilate(closing, kernel, iterations = 1)
	return dilation

# Returns the mask of the orange segments
# REQUIRES: tuning of bounds
def LAB_mask(LAB_img):
	lower_bound = np.array([0 , 128 , 126])
	upper_bound = np.array([255 , 255 , 255])
	orange_mask = cv2.inRange(LAB_img, lower_bound, upper_bound)
	return mask_processing(orange_mask)

# From https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption/46391574
# For color correction
def white_balance(LAB_img):
    avg_a = np.average(LAB_img[:, :, 1])
    avg_b = np.average(LAB_img[:, :, 2])
    LAB_img[:, :, 1] = LAB_img[:, :, 1] - ((avg_a - 128) * (LAB_img[:, :, 0] / 255.0) * 1.1)
    LAB_img[:, :, 2] = LAB_img[:, :, 2] - ((avg_b - 128) * (LAB_img[:, :, 0] / 255.0) * 1.1)

    return LAB_img

def contour_detection(mask_vertical_bar, frame):
	(contours,_) = cv2.findContours(mask_vertical_bar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	box_points = list()
	for contour in contours:
		# Filter out false positives
		rect = cv2.minAreaRect(contour)
		box = cv2.boxPoints(rect)
		if contour_filter(contour, rect, box):
			box = np.int0(box)
			box_points.append(box)
			cv2.drawContours(frame,[box],0,(0,0,255), 2)

	# DEBUGGING:
	if len(box_points) > 3:
			print("BREAKPOINT")
			print(box_points)
			for box in box_points:
				cv2.drawContours(frame,[box],0,(0,0,255), 10)
			cv2.imshow("frame", frame)
			cv2.waitKey(0)

# Transforms the [-90, 0] range of rect[2] to [-180, 0]
# designate 'height' as edge between box[0] and box[1]
# 'width' as edge between box[1] and box[2]
def calc_minAreaRect_orientation(rect, sides):
	if sides[1] < sides[0]:
		return rect[2] - 90
	else:
		return rect[2]

# Requires tuning
def contour_filter(contour, rect, box):
	# remove contours that are too small
	perimeter = cv2.arcLength(contour,True)
	if perimeter < 60:
		return False

	# remove contours that are too small again
	area = cv2.contourArea(contour)
	if area < 2000:
		return False

	# Remove 'boxy' contours - keep rod-like ones 
	# src says this is fast way to calc distance betw two points.
	# src: https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
	# Since contour's 0-index is the lowest (greatest y-val) point, and goes clockwise, 0-1 and 1-2 must refer
	# to height and width (in any order)
	sides = [np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])]
	height = np.amax(sides)
	width = np.amin(sides)
	aspect_ratio = height / width
	if aspect_ratio < 5:
		return False

	# We want our rod to be vertical i.e. long side should be vertical.
	# src says angle in rect[2] (from minAreaRect)is calculated from first edge counterclockwise from horizontal (edge between box[0] and box[3]).
	# src: https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
	transformed_angle = calc_minAreaRect_orientation(rect, sides)
	if transformed_angle < -135 or transformed_angle > -45:
		return False

	# Our rods should fit tightly inside its bounding box. If not, then its probably distorted surface reflection
	# Solidity can be thought of as an approximation of the fit
	hull = cv2.convexHull(contour)
	hull_area = cv2.contourArea(hull)
	solidity = float(area)/hull_area
	if solidity < 0.8:
		return False

	return True

def contour_processing(frame):
	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 1000, 1000)

	frame = sharp_and_smooth(frame)

	LAB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

	white_balanced_LAB = white_balance(LAB_img)
	enhanced_LAB = CLAHE_enhance(white_balanced_LAB)

	orange_mask = LAB_mask(enhanced_LAB)
	converted_orange_mask = cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR)
	converted_BGR = cv2.cvtColor(enhanced_LAB, cv2.COLOR_LAB2BGR)

	contour_detection(orange_mask, converted_BGR)

	cv2.imshow("frame", np.hstack([converted_BGR, converted_orange_mask]))
	
		
def main():
	cap = cv2.VideoCapture("/media/sf_VM_shared/MSub/test_clip.mp4")
	frame_count = 0
	while cap.isOpened():
		ret, frame = cap.read()
		print("frame count:", frame_count)
		frame_count += 1

		# if frame is read correctly ret is true
		if not ret:
				print("Can't receive frame (stream end?). Exiting...")
				break
		
		contour_processing(frame)

		if cv2.waitKey(1) == ord('q'):
			print("user exit!")
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	#video_to_photo()
	#contour_detect()
	main()
