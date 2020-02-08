# MSubs 2020
import numpy as np
import cv2

class Centering:
	# Uses detected lines to detect how to center the robot.
	# Currently assuming we the robot is heading head on into the gate.
	# Need to upgrade to handle possible angled-approach.

	# TO ADD IN
	__FRAME_CENTER = (XX, YY)

	# Takes in lines and outputs their mid-points
	def calc_mid_points(lines):
		mid_points = np.zeros_like(lines)
		for index, line in enum(lines):
			for x1, y1, x2, y2 in line:
				mid_points[index] = ((x1 + x2) / 2, (y1 + y2) / 2)
		return mid_points

	# Given the coords of several points, find their center
	def calc_points_center(points):
		x_sum = 0
		y_sum = 0
		for point in points:
			x_sum += point[0]
			y_sum += point[1]
		return (x_sum / points.shape[0], y_sum / points.shape[0])

	# Finds the heading (angle and pixel distance) of the gate center from the frame center
	def calc_heading(lines):
		mid_points = calc_mid_points(lines)
		gate_center = calc_points_center(mid_points)
		# Basically AB = OB - OA
		return (gate_center[0] - __FRAME_CENTER[0], gate_center[1] - __FRAME_CENTER[1])

	# Testing function to display the heading arrow on the frame.
	def display_heading(frame, gate_center):
		line_image = np.zeros_like(frame)
		cv2.line(line_image, __FRAME_CENTER, gate_center, line_color, line_width)
        line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        gate_lines_image = display_lines(frame, line_image)
        cv2.imshow("heading line", gate_lines_image)
