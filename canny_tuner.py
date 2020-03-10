# MSubs 2020
import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

# Preprocessing, currently reads and saves to file. Copy line_segent_detect main into here and feed 
# into it edges instead of reading from file. Can combine this function with video-to-photo for frame-by-frame
# camera processing.
def canny_tuner():
	cv2.namedWindow('modified', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('modified', 1440, 1080)
	print("Look at source code for controls")
	value = 90
	threshold1 = 0
	threshold2 = 255
	step_size = 10
	threshold_dict = {0: threshold1, 1: threshold2}
	selected_threshold = 1

	while(True):
		img = cv2.imread("data/frame%d.jpg"%value)
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		v_channel = hsv_img[:, :, 2] # Extract 'intensity'

		# Unsharp masking technique for shaperning edges
		# Helps make edges more 'connected'
		gaussian_blurred = cv2.GaussianBlur(v_channel, (11,11), 9);
		sharpened = cv2.addWeighted(v_channel, 1.5, gaussian_blurred, -0.5, 0);
		
		# Remove salt and pepper noise
		median_blurred = cv2.medianBlur(sharpened, 11)
		# Remove details
		#combine_blurred = cv2.GaussianBlur(median_blurred, (5,5), 0 )

		edges = cv2.Canny(median_blurred, threshold_dict[0], threshold_dict[1])
		kernel = np.ones((5,5),np.uint8)
		opening = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

		cv2.imshow("modified", opening)
		
		# USER COMMAND
		key = cv2.waitKey(0) & 0xFF
		if(key == ord('q')):
			print("current: (", threshold_dict[0], ",", threshold_dict[1], ")")
			break
		elif(key == ord('n')):
			threshold_dict[selected_threshold] -= step_size
			print("Lower! Now: ", selected_threshold, ":", threshold_dict[selected_threshold])
		elif(key == ord('m')):
			threshold_dict[selected_threshold] += step_size
			print("Higher! Now: ", selected_threshold, ":", threshold_dict[selected_threshold])
		elif(key == ord('v')):
			step_size -= 1
			print("Tighten Step! Step: ", step_size)
		elif(key == ord('b')):
			step_size += 1
			print("Widen Step! Step: ", step_size)
		elif(key == ord('x')):
			selected_threshold = (selected_threshold + 1) % 2
			print("Switch to threshold:", selected_threshold)
		elif(key == ord('d')):
			print("current: (", threshold_dict[0], ",", threshold_dict[1], ")")
		elif(key == ord('o')):
			value -= 1
			print(value)
		elif(key == ord('p')):
			value += 1
			print(value)
		elif(key == ord('r')):
			print("RESET!")
			threshold_dict[0] = 0
			threshold_dict[1] = 255
			step_size = 10
		else:
			print("invalid key!")


if __name__ == '__main__':
	canny_tuner()

