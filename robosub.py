# MSubs 2020
import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

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

# Preprocessing, currently reads and saves to file. Copy line_segent_detect main into here and feed 
# into it edges instead of reading from file. Can combine this function with video-to-photo for frame-by-frame
# camera processing.
def edge_detect():

	for value in range(0, 105):
		img = cv2.imread("data/frame%d.jpg"%value)
		print(type(img))
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		v_channel = hsv_img[:, :, 2] # Extract 'intensity'

		# Unsharp masking technique for shaperning edges
		# Helps make edges more 'connected'
		out1 = cv2.GaussianBlur(v_channel, (0,0), 3);
		out2 = cv2.addWeighted(v_channel, 1.5, out1, -0.5, 0);

		# Remove salt and pepper noise
		median_blurred = cv2.medianBlur(out2, 15)
		# Remove details
		combine_blurred = cv2.GaussianBlur(median_blurred, (5,5), 0 )

		print((img).shape)

		edges = cv2.Canny(combine_blurred, 20, 20)
		#cv2.imshow("modified", edges)

		#cv2.waitKey(0)
		
		img = Image.fromarray(edges)
		img.save("data/edge_frame%d.jpg"%value)

#video_to_photo()
edge_detect()
