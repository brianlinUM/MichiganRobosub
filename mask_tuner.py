# MSubs 2020
import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

def video_to_photo():
	# Change to camera or file input if necessary
	cam = cv2.VideoCapture("/media/sf_VM_shared/msub_train/tues_6_clip.mp4") 
	  
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

# returns the smoothed image
def sharp_and_smooth(img):
	# Unsharp masking technique for shaperning edges
	# Helps make edges more 'connected'
	gaussian_blurred = cv2.GaussianBlur(img, (11,11), 9);
	sharpened = cv2.addWeighted(img, 1.5, gaussian_blurred, -0.5, 0);
	
	# Remove salt and pepper noise
	median_blurred = cv2.medianBlur(sharpened, 11)
	combine_blurred = cv2.GaussianBlur(median_blurred, (5,5), 0 )

	return combine_blurred

# Returns the enhanced LAB image
def CLAHE_enhance(LAB_img):
	L_channel = LAB_img[:,:,0]

	CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	CLAHE_L_channel = CLAHE.apply(L_channel)

	LAB_img[:,:,0] = CLAHE_L_channel

	return LAB_img

# Returns the mask of the orange segments
# REQUIRES: tuning of bounds
def LAB_mask(LAB_img):
	lower_bound = np.array([0, 0, 0])
	upper_bound = np.array([255, 255, 255])
	orange_mask = cv2.inRange(LAB_img, lower_bound, upper_bound)
	return orange_mask

# From https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption/46391574
# For color correction
def white_balance(LAB_img):
    avg_a = np.average(LAB_img[:, :, 1])
    avg_b = np.average(LAB_img[:, :, 2])
    LAB_img[:, :, 1] = LAB_img[:, :, 1] - ((avg_a - 128) * (LAB_img[:, :, 0] / 255.0) * 1.1)
    LAB_img[:, :, 2] = LAB_img[:, :, 2] - ((avg_b - 128) * (LAB_img[:, :, 0] / 255.0) * 1.1)

    return LAB_img

# Preprocessing, currently reads and saves to file. Copy line_segent_detect main into here and feed 
# into it edges instead of reading from file. Can combine this function with video-to-photo for frame-by-frame
# camera processing.
def edge_detect():
	cv2.namedWindow('modified', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('modified', 1440, 1080)
	value = 90
	threshold1 = 0
	threshold2 = 0
	threshold3 = 0
	threshold1_u = 255
	threshold2_u = 255
	threshold3_u = 255
	step_size = 10
	lower_threshold_dict = {0: threshold1, 1: threshold2, 2: threshold3}
	upper_threshold_dict = {0: threshold1_u, 1: threshold2_u, 2: threshold3_u}
	selected_threshold = 1
	bound_dict = {0: lower_threshold_dict, 1: upper_threshold_dict}
	selected_bound = 1
	#files = [469, 609, 856]
	

	#for value in range(90, 300):
	while(True):
		masks = list()
		corrected = list()
		#print(value)
		#for value in files:
		img = cv2.imread("data/frame%d.jpg"%value)
		img = sharp_and_smooth(img)

		LAB_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

		white_balanced_LAB = white_balance(LAB_img)
		#enhanced_LAB = CLAHE_enhance(white_balanced_LAB)

		converted_BGR = cv2.cvtColor(white_balanced_LAB, cv2.COLOR_LAB2BGR)

		#orange_mask = HSV_mask(HSV_img)
		orange_mask = cv2.inRange(white_balanced_LAB, (bound_dict[0][0], bound_dict[0][1], bound_dict[0][2]), (bound_dict[1][0], bound_dict[1][1], bound_dict[1][2]))
		converted_orange_mask = cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR)
		
		#masks.append(orange_mask)
		#corrected.append(converted_BGR)

		#top = np.hstack([masks[0], masks[1], masks[2]])
		#bot = np.hstack([corrected[0], corrected[1], corrected[2]])


		cv2.imshow("modified", np.hstack([converted_BGR, converted_orange_mask]))
		'''
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
		'''
		# USER COMMAND
		key = cv2.waitKey(0) & 0xFF
		if(key == ord('q')):
			print("current lower: (", bound_dict[0][0], ",", bound_dict[0][1], ",", bound_dict[0][2], ") || upper: (", bound_dict[1][0], ",", bound_dict[1][1], ",", bound_dict[1][2], ")")
			break
		elif(key == ord('n')):
			bound_dict[selected_bound][selected_threshold] -= step_size
			print("Lower! Now: ", selected_bound, ":", selected_threshold, ":   ", bound_dict[selected_bound][selected_threshold])
		elif(key == ord('m')):
			bound_dict[selected_bound][selected_threshold] += step_size
			print("Higher! Now: ", selected_bound, ":", selected_threshold, ":   ", bound_dict[selected_bound][selected_threshold])
		elif(key == ord('v')):
			step_size -= 1
			print("Tighten Step! Step: ", step_size)
		elif(key == ord('b')):
			step_size += 1
			print("Widen Step! Step: ", step_size)
		elif(key == ord('x')):
			selected_threshold = (selected_threshold + 1) % 3
			print("Switch to threshold:", selected_threshold)
		elif(key == ord('z')):
			selected_bound = (selected_bound + 1) % 2
			print("Switch to bound:", selected_bound)
		elif(key == ord('d')):
			print("current lower: (", bound_dict[0][0], ",", bound_dict[0][1], ",", bound_dict[0][2], ") || upper: (", bound_dict[1][0], ",", bound_dict[1][1], ",", bound_dict[1][2], ")")
		elif(key == ord('o')):
			value -= 1
			print(value)
		elif(key == ord('p')):
			value += 1
			print(value)
		elif(key == ord('r')):
			print("RESET!")
			for j in range(0,2):
				for i in range(0,3):
					if j == 0:
						bound_dict[j][i] = 0
					if j == 1:
						bound_dict[j][i] = 255
			step_size = 10
			value = 90
		else:
			print("invalid key!")
	
#video_to_photo()
edge_detect()
