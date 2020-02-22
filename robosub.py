import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

def video_to_photo():
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


def edge_detect():

	for value in range(90,105):
		img = cv2.imread("data/frame%d.jpg"%value)
		print(type(img))
		img = cv2.GaussianBlur(img,(15,15), 0)
		print((img).shape)

		edges = cv2.Canny(img,20,20)
		#scipy.misc.imsave("data/edge_frame%d.jpg"%value, edges)
		img = Image.fromarray(edges)
		img.save("data/edge_frame%d.jpg"%value)

		#plt.subplot(121),plt.imshow(img,cmap = 'hsv')
		#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		#plt.show()
		#plt.savefig('data/frame%d.png'%value)
#video_to_photo()
edge_detect()
