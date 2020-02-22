
# Essentially clusters the line segments
import numpy as np
import cv2
import math
import frame_centering as centering
from PIL import Image

mid = 0

class HoughBundler:
    # Cluster
    '''
    def combine_lines(line_segments, labels){
        cv2.kmeans()
    }'''

    def filter_line_segments(self, line_segments):
        horizontal_list = list()
        angle_threshold = 175
        lines_list = list(line_segments)
        for index, line_seg in enumerate(lines_list):
            for x1, y1, x2, y2 in line_seg:
                if(x1 - x2 != 0):
                    angle = math.atan2(y1 - y2, x1 - x2)
                    #tmp_img = np.zeros((1080, 1920))
                    
                    #cv2.line(tmp_img, (x1, y1), (x2, y2), 255, 10)
                    #print(math.degrees(angle))
                    #cv2.imshow("dog", tmp_img)
                    #cv2.waitKey(0)
                    #print(math.degrees(angle))
                    if( (math.radians(angle_threshold) <= angle and angle <= math.radians(180) ) or ( math.radians(-180) <= angle and angle <= math.radians(-angle_threshold) )  ) :
                        horizontal_list.append(line_seg)

        #print(bar_height)

        return np.asarray(horizontal_list)

    def average_height(self, lines):
        sum = 0;
        for line in lines:
            sum += line[0][1] # y1
            sum += line[0][3] # y2
        return int(sum / (len(lines)*2))

    def cut_top(self, line_segments):
        global mid
        slack = 20
        horizontal_list = self.filter_line_segments(line_segments)
        new_list = []
        bar_height = self.average_height(horizontal_list) - slack
        mid = bar_height
        for line_seg in line_segments:
            for x1, y1, x2, y2 in line_seg:
                if y1 < bar_height and y2 < bar_height:
                    print("too high! delete")
                elif y1 < bar_height:
                    #print("trim y1!")
                    new_list.append(np.array([[x1, bar_height, x2, y2]]))
                elif y2 < bar_height:
                    #print("trim y2!")
                    new_list.append(np.array([[x1, y1, x2, bar_height]]))
                else:
                    #print("Keep!")
                    new_list.append(np.array([[x1, y1, x2, y2]]))
        return new_list

    # From the DeepPiCar tutorial:
    def line_segment_detect(self, edges):
        # Need to tune parameters through trial-and-error
        rho = 1  # distance precision in pixel, i.e. 1 pixel
        angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
        min_threshold = 200  # minimal of votes
        line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold, 
                                        np.array([]), minLineLength=75, maxLineGap=20)
        print(edges.shape)
        print(line_segments)

        return self.cut_top(line_segments)
        #return line_segments

    # A testing function to draw the lines output by process_lines onto frame
    def display_lines(self, frame, lines, line_width=2):
        global mid
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, line_width)
        cv2.imshow("line_segs", line_image)
        #cv2.waitKey(0)
        line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        line_image = cv2.circle(line_image, (960, mid), 5, 255, thickness=5)
        line_image = cv2.circle(line_image, (960, mid - 20), 5, 125, thickness=5)
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
