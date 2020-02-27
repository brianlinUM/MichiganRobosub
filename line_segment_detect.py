# MSubs 2020, maintained by Brian the dog
# Essentially clusters the line segments. ASSUMES camera can see the entirety of gate clearly. Does not 
# work well with tilted, or incomplete view of the gate.

import numpy as np
import cv2
import math
import frame_centering as centering
from PIL import Image
from sklearn.cluster import KMeans

mid = 0

# Seperate horizontal and vertical lines by angle. Also calculate average horizontal line height
def filter_line_segments(line_segments):
    horizontal_list = list()
    vertical_list = list()
    angle_threshold = 175
    lines_list = list(line_segments)
    horizontal_line_height_sum = 0;
    for index, line_seg in enumerate(lines_list):
        for x1, y1, x2, y2 in line_seg:
            if(x1 - x2 != 0):
                angle = math.atan2(y1 - y2, x1 - x2)
                if( (math.radians(angle_threshold) <= angle and angle <= math.radians(180) ) or ( math.radians(-180) <= angle and angle <= math.radians(-angle_threshold) )  ) :
                    horizontal_list.append(line_seg)
                    horizontal_line_height_sum += line_seg[0][1]; # y1
                    horizontal_line_height_sum += line_seg[0][3]; # y2
                else:
                    vertical_list.append(line_seg)
    if(len(horizontal_list) == 0):
        average_bar_height = 0
    else:
        average_bar_height = int(horizontal_line_height_sum / (len(horizontal_list)*2))
    return np.asarray(horizontal_list), np.asarray(vertical_list), average_bar_height

# For reflection handling. Remove or trim lines going above the horizontal bar
def cut_top(horizontal_list, vertical_list, average_bar_height):
    global mid
    slack = 10
    new_vertical_list = []
    bar_height = average_bar_height - slack # make higher
    mid = bar_height
    for line_seg in vertical_list:
        for x1, y1, x2, y2 in line_seg:
            if y1 < bar_height and y2 < bar_height:
                pass
            elif y1 < bar_height:
                #print("trim y1!")
                new_vertical_list.append(np.array([[x1, bar_height, x2, y2]]))
            elif y2 < bar_height:
                #print("trim y2!")
                new_vertical_list.append(np.array([[x1, y1, x2, bar_height]]))
            else:
                #print("Keep!")
                new_vertical_list.append(np.array([[x1, y1, x2, y2]]))
    return new_vertical_list

# From the DeepPiCar tutorial (part 4). Outputs line segments as (x1,y1,x2,y2)
def line_segment_detect(edges):
    # Need to tune parameters through trial-and-error
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 60  # minimal of votes
    line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=15, maxLineGap=20)


    return line_segments

def cluster_vertical_lines(vertical_centers):

    clusterer = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
    labels = clusterer.fit_predict(vertical_centers)
    return labels

# A testing function to draw the lines output by process_lines onto frame
def display_lines(frame, horizontal_list, vertical_lines, line_width=2):
    global mid
    line_image = np.zeros_like(frame)
    '''
    mid_points = centering.calc_mid_points(vertical_lines, x_only=1)
    labels = cluster_vertical_lines(mid_points)#

    if vertical_lines is not None:
        for index, label in enumerate(labels):
            if label == 0:
                for x1, y1, x2, y2 in vertical_lines[index]:
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, line_width)
            if label == 1:
                for x1, y1, x2, y2 in vertical_lines[index]:
                    cv2.line(line_image, (x1, y1), (x2, y2), 160, line_width)
            if label == 2:
                for x1, y1, x2, y2 in vertical_lines[index]:
                    cv2.line(line_image, (x1, y1), (x2, y2), 40, line_width)
        cv2.imshow("cluster 1", line_image)
        cv2.waitKey(0)
    '''
    # Draw vertical lines first only
    if vertical_lines is not None:
        for line in vertical_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), 255, line_width)

    if horizontal_list is not None:
        for line in horizontal_list:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), 125, line_width)
    
    
    #cv2.imshow("blur line", line_image)
    line_image = cv2.circle(line_image, (960, mid), 5, 255, thickness=5)
    line_image = cv2.circle(line_image, (960, mid - 20), 5, 125, thickness=5)
    
    # Change to calc_heading and print result if do not want image (also remove cv2.line drawing)
    after = centering.draw_heading(line_image, vertical_lines)
    cv2.imshow("result", after)
    
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
