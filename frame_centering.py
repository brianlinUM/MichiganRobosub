# MSubs 2020
import numpy as np
import cv2
from math import sqrt

__FRAME_CENTER = (960,540) # width x height = x, y

# Takes in lines and outputs their mid-points
def calc_mid_points(lines):
	mid_points = list()
	lengths = list()
	for line in lines:
		for x1, y1, x2, y2 in line:
			mid_points.append(((x1 + x2) / 2, (y1 + y2) / 2))
			lengths.append(sqrt((x1 - x2)**2 + (y1-y2)**2))
	return mid_points, lengths

# Given the coords of several points, find their center
def calc_points_center(points, lengths):
	x_sum = 0
	y_sum = 0
	total_length = 0
	for index, point in enumerate(points):
		length = lengths[index]
		x_sum += point[0] * length
		y_sum += point[1] * length
		total_length += length
	return (int(x_sum / total_length), int(y_sum / total_length))

# Finds the heading (angle and pixel distance) of the gate center from the frame center
def calc_heading(lines):
	mid_points, lengths = calc_mid_points(lines)
	gate_center = calc_points_center(mid_points, lengths)
	# Basically AB = OB - OA
	return (gate_center[0] - __FRAME_CENTER[0], gate_center[1] - __FRAME_CENTER[1])

# Testing function to display the heading arrow on the frame.
def draw_heading(frame, lines):
	mid_points, lengths = calc_mid_points(lines)
	gate_center = calc_points_center(mid_points, lengths)
	
	cv2.arrowedLine(frame, __FRAME_CENTER, gate_center, 255 , 2)
	return frame
