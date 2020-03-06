# MSubs 2020
import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

__FRAME_CENTER = (960,540) # width x height = x, y

def video_to_photo():
	# Change to camera or file input if necessary
	cam = cv2.VideoCapture("test_clip.mp4") 
	  
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


# returns the center of contour box
def box_center(box):
	x_sum = 0
	y_sum = 0
	for corner in box:
		x_sum += corner[0]
		y_sum += corner[1]
	return (int(x_sum / 4), int(y_sum / 4))

def contour_gate_detection(processed_img):
# Find orange colors
	mask_vertical_bar = cv2.inRange(processed_img, (130,125, 90), (160, 155, 255))

	# Find rectangles through contours
	(contours,_) = cv2.findContours(mask_vertical_bar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	box_points = list()
	for contour in contours:
		perimeter = cv2.arcLength(contour,True)
		# remove false positives that are too small
		if perimeter > 60:
			rect = cv2.minAreaRect(contour)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			box_points.append(box)
			#cv2.drawContours(img,[box],0,(0,0,255),2)
	'''
	cv2.imshow("image", img)
	cv2.waitKey(0)
	d
	'''
	# sort by y value (ascending/ lower height)
	vertical_bar_boxes = sorted(box_points, key=lambda x: x[0][1], reverse=True)
	# Assume that the lowest three are the vertical bars (above are surface reflection/ false positives)
	vertical_bar_boxes = vertical_bar_boxes[:3]
	if(len(vertical_bar_boxes) >= 3):
		mid_bar_box = vertical_bar_boxes[2]
		# sort by x value ascending/ 'right'est
		vertical_bar_boxes = sorted(vertical_bar_boxes, key=lambda x: x[0][0], reverse=True)
		right_bar_box = vertical_bar_boxes[0]
		left_bar_box = vertical_bar_boxes[2]

		# calculate center of each bar
		mid_bar_center = box_center(mid_bar_box)
		left_bar_center = box_center(left_bar_box)
		right_bar_center = box_center(right_bar_box)
		cv2.circle(processed_img, mid_bar_center, 5, (0,255,0), thickness=5)
		cv2.circle(processed_img, left_bar_center, 5, (0,255,0), thickness=5)
		cv2.circle(processed_img, right_bar_center, 5, (0,255,0), thickness=5)

		# Calculate target point (center of right half)
		y_target = int((left_bar_center[1] + right_bar_center[1]) / 2)
		x_target = int((mid_bar_center[0] + right_bar_center[0]) / 2)
		target = (x_target, y_target)
		cv2.circle(processed_img, target, 5, (0,0,255), thickness=5)

		cv2.drawContours(processed_img,[mid_bar_box],0,(0,255,0),2)
		cv2.drawContours(processed_img,[right_bar_box],0,(0,255,0),2)
		cv2.drawContours(processed_img,[left_bar_box],0,(0,255,0),2)

		# Calculate heading to target
		heading = (target[0] - __FRAME_CENTER[0], target[1] - __FRAME_CENTER[1])
		print(heading)

		# If center of frame is close enough to center, move forward
		if(distance.euclidean(target, __FRAME_CENTER) < 30): 
			print("@@@@@@@@   CHARGE!!!!!!") # placeholder print
		cv2.arrowedLine(processed_img, __FRAME_CENTER, target, 255 , 2)
	else:
		for box in vertical_bar_boxes:
			center = box_center(box)
			cv2.circle(processed_img, center, 5, (0,255,255), thickness=5)
			cv2.drawContours(processed_img,[box],0,(0,255,255),2)

# Preprocessing, currently reads and saves to file. Copy line_segent_detect main into here and feed 
# into it edges instead of reading from file. Can combine this function with video-to-photo for frame-by-frame
# camera processing.
def gate_centering():
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 1600, 1200)
	#specific_img = 287
	for value in range(78,200):
		print(value, ":")
		img = cv2.imread("data/frame%d.jpg"%value)

		# sharpen image using unsharp masking technique
		out1 = cv2.GaussianBlur(img, (0,0), 3)
		out2 = cv2.addWeighted(img, 1.5, out1, -0.5, 0)

		# remove salt and pepper noise
		median_blurred = cv2.medianBlur(out2, 11)
		# smooth out details
		combine_blurred = cv2.GaussianBlur(median_blurred, (5,5), 0 )

		contour_gate_detection(combine_blurred)

		cv2.imshow("image", combine_blurred)
		cv2.waitKey(0)

#video_to_photo()
if __name__ == '__main__':
	gate_centering()

