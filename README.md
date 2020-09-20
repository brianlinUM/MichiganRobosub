# Michigan Robosub 2020

## About student organization:
We are a student design team at the University of Michigan. Our goal is to design an
autonomous underwater robot capable of navigating through an obstacle course and
complete various tasks while submerged and with no human input. We aim to compete
in the RoboSub competition: https://robonation.org/programs/robosub/

## Gate Navigation Algorithm
Given camera input, guides the robot to the center of the gate by calculating the target
heading using computer vision techniques.

To run:
	python robosub.py

To use real-time camera input instead of file input, you would have to change how robosub.py 
gate_centering() function gets its data i.e. move it into the while loop in video_to_photo.

Make sure to suppress debugging image displays (cv2.imshow() ) and prints as necessary in robosub.py
Currently, the navigation system works best when entirety of gate is seen.

Future direction: object tracking with and detection using deep learning.
Probably more accurate and faster once gate bars are initially detected

### Current algorithm outline:

pre-processing:
	sharpen image, remove noise, threshold filter orange colors

contour boxing:
	calculate the contours of the orange bars, then create a box representation of each bar

false postive removal:
	assume lowest 3 boxes are valid bars (above ones are false positives/ surface reflections)

target calculation:
	based on box boundaries (corners), calculate the target point (center of right half)
	
### Utility scripts:
cam_test.py: tests if camera device is working in opencv.
mask_tuner.py: interactive tool to tune color mask thresholds.
