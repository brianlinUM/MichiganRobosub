# MSubs 2020
import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
from random import randint

__FRAME_CENTER = (960,540) # width x height = x, y

# Preprocessing, currently reads and saves to file. Copy line_segent_detect main into here and feed 
# into it edges instead of reading from file. Can combine this function with video-to-photo for frame-by-frame
# camera processing.
def gate_centering():
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 1600, 1200)
	first = 1
	bboxes = []
	colors = []
	multiTracker = cv2.MultiTracker_create()
	#specific_img = 287
	for value in range(90,200):
		print(value)
		img = cv2.imread("data/frame%d.jpg"%value)

		# sharpen image using unsharp masking technique
		#out1 = cv2.GaussianBlur(img, (0,0), 3)
		#out2 = cv2.addWeighted(img, 1.5, out1, -0.5, 0)

		# remove salt and pepper noise
		#median_blurred = cv2.medianBlur(out2, 11)
		# smooth out details
		#combine_blurred = cv2.GaussianBlur(median_blurred, (5,5), 0 )

		if first == 1:
			## Select boxes

			# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
			# So we will call this function in a loop till we are done selecting all objects
			while True:
				# draw bounding boxes over objects
				# selectROI's default behaviour is to draw box starting from the center
				# when fromCenter is set to false, you can draw box starting from top left corner
				bbox = cv2.selectROI('image', img)
				print(bbox)
				d
				bboxes.append(bbox)
				colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
				print("Press q to quit selecting boxes and start tracking")
				print("Press any other key to select next object")
				k = cv2.waitKey(0) & 0xFF
				if (k == 113):  # q is pressed
					break

			print('Selected bounding boxes {}'.format(bboxes))

			# Initialize MultiTracker 
			for bbox in bboxes:
				multiTracker.add(cv2.TrackerKCF_create(), img, bbox)

			first = 0
		
		success, boxes = multiTracker.update(img)
		print("success status:", success)

		for index, newbox in enumerate(boxes):
			p1 = (int(newbox[0]), int(newbox[1]))
			p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
			cv2.rectangle(img, p1, p2, colors[index], 2, 1)

		cv2.imshow("image", img)
		cv2.waitKey(0)


#video_to_photo()
gate_centering()
