# MSubs 2020
import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

def line_segment_detect(edges_image, threshold1, threshold2, threshold3):
    # Need to tune parameters through trial-and-error
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = threshold1  # minimal of votes
    line_segments = cv2.HoughLinesP(edges_image, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=threshold2, maxLineGap=threshold3)
    return line_segments

def draw_lines(image, lines, color=255, line_width=2):
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(image, (x1,y1), (x2,y2), color, line_width)

# Preprocessing, currently reads and saves to file. Copy line_segent_detect main into here and feed 
# into it edges instead of reading from file. Can combine this function with video-to-photo for frame-by-frame
# camera processing.
def hough_tuner():
	cv2.namedWindow('modified', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('modified', 1440, 1080)
	print("Look at source code for controls")
	value = 90
	threshold1 = 0
	threshold2 = 0
	threshold3 = 10
	step_size = 10
	threshold_dict = {0: threshold1, 1: threshold2, 2: threshold3}
	selected_threshold = 0

	while(True):
		img = cv2.imread("data/frame%d.jpg"%value)
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		v_channel = hsv_img[:, :, 2] # Extract 'intensity'
		empty = np.zeros_like(v_channel)

		# Unsharp masking technique for shaperning edges
		# Helps make edges more 'connected'
		gaussian_blurred = cv2.GaussianBlur(v_channel, (11,11), 9);
		sharpened = cv2.addWeighted(v_channel, 1.5, gaussian_blurred, -0.5, 0);
		
		# Remove salt and pepper noise
		median_blurred = cv2.medianBlur(sharpened, 11)
		# Remove details
		#combine_blurred = cv2.GaussianBlur(median_blurred, (5,5), 0 )

		edges = cv2.Canny(median_blurred, 27,14)
		kernel = np.ones((5,5),np.uint8)
		dilation = cv2.dilate(edges,kernel,iterations = 1)
		
		#lines = line_segment_detect(dilation, threshold_dict[0], threshold_dict[1], threshold_dict[2])
		
		#draw_lines(empty, lines)

		cv2.imshow("modified", np.hstack([edges, dilation]))
		
		# USER COMMAND
		key = cv2.waitKey(0) & 0xFF
		if(key == ord('q')):
			print("current: (", threshold_dict[0], ",", threshold_dict[1], ",", threshold_dict[2], ")")
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
			selected_threshold = (selected_threshold + 1) % 3
			print("Switch to threshold:", selected_threshold)
		elif(key == ord('d')):
			print("current: (", threshold_dict[0], ",", threshold_dict[1], ",", threshold_dict[2], ")")
		elif(key == ord('o')):
			value -= 1
			print(value)
		elif(key == ord('p')):
			value += 1
			print(value)
		elif(key == ord('r')):
			print("RESET!")
			threshold_dict[0] = 0
			threshold_dict[1] = 0
			threshold_dict[2] = 10
			step_size = 10
		else:
			print("invalid key!")


if __name__ == '__main__':
	hough_tuner()

