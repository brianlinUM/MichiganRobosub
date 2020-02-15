
# Essentially clusters the line segments
import numpy as np
import cv2
import math
import frame_centering as centering
from PIL import Image

class HoughBundler:
    # Cluster
    '''
    def combine_lines(line_segments, labels){
        cv2.kmeans()
    }'''

    # From the DeepPiCar tutorial:
    def line_segment_detect(self, edges):
        # Need to tune parameters through trial-and-error
        rho = 1  # distance precision in pixel, i.e. 1 pixel
        angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
        min_threshold = 275  # minimal of votes
        line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold, 
                                        np.array([]), minLineLength=75, maxLineGap=20)
        print(edges.shape)
        print(line_segments)
        return line_segments

    # A testing function to draw the lines output by process_lines onto frame
    def display_lines(self, frame, lines, line_width=2):
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, line_width)
        cv2.imshow("line_segs", line_image)
        #cv2.waitKey(0)
        line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        after = centering.draw_heading(line_image, lines)
        cv2.imshow("gate lines", after)
        #after_img = Image.fromarray(after)
        #after_img.save("edge_frame%d.jpg")

        cv2.waitKey(0)


def main():
    HB = HoughBundler()
    for value in range(90,105):
        img = cv2.imread("data/edge_frame%d.jpg"%value, cv2.CV_8UC1)
        line_segs = HB.line_segment_detect(img)
        

        #merged_lines = HB.process_lines(line_segs)
        HB.display_lines(img, line_segs)

        

    #plt.subplot(121),plt.imshow(img,cmap = 'hsv')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
    #plt.savefig('data/frame%d.png'%value)

if __name__ == '__main__':
    main()
