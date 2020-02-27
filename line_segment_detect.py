# MSubs 2020
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
    angle_threshold = 160
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
    mid = average_bar_height
    bar_height = average_bar_height - slack # make higher
    
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
    
    mid_points = centering.calc_mid_points(vertical_lines, x_only=1)
    labels = cluster_vertical_lines(mid_points)
    average_angle1 = 0
    num_1 = 0
    line_1 = [0,0,0,1080] # x_1, y_min, x_2, y_max
    x_sum_1 = 0
    cluster_1 = list()

    average_angle2 = 0
    num_2 = 0
    line_2 = [0,0,0,1080]
    x_sum_2 = 0
    cluster_2 = list()

    average_angle3 = 0
    line_3 = [0,0,0,1080]
    num_3 = 0
    x_sum_3 = 0
    cluster_3 = list()

    if vertical_lines is not None:
        for index, label in enumerate(labels):
            if label == 0:
                for x1, y1, x2, y2 in vertical_lines[index]:
                    num_1 += 1
                    average_angle1 += math.atan2(y1 - y2, x1 - x2)
                    cluster_1.append(index)
                    x_sum_1 += x1 + x2
                    if max((y1,y2)) > line_1[1]: 
                        line_1[1] = max((y1,y2))
                        line_1[0] = vertical_lines[index][0][np.argmax((y1,y2)) * 2]
                    if min((y1,y2)) < line_1[3]: 
                        line_1[3] = min((y1,y2))
                        line_1[2] = vertical_lines[index][0][np.argmin((y1,y2)) * 2]
                    cv2.line(line_image, (x1, y1), (x2, y2), 200, line_width)
            if label == 1:
                for x1, y1, x2, y2 in vertical_lines[index]:
                    num_2 += 1
                    average_angle2 += math.atan2(y1 - y2, x1 - x2)
                    cluster_2.append(index)
                    x_sum_2 += x1 + x2
                    if max((y1,y2)) > line_2[1]: 
                        line_2[1] = max((y1,y2))
                        line_2[0] = vertical_lines[index][0][np.argmax((y1,y2)) * 2]
                    if min((y1,y2)) < line_2[3]: 
                        line_2[3] = min((y1,y2))
                        line_2[2] = vertical_lines[index][0][np.argmin((y1,y2)) * 2]
                    cv2.line(line_image, (x1, y1), (x2, y2), 160, line_width)
            if label == 2:
                for x1, y1, x2, y2 in vertical_lines[index]:
                    num_3 += 1
                    average_angle3 += math.atan2(y1 - y2, x1 - x2)
                    cluster_3.append(index)
                    x_sum_3 += x1 + x2
                    if max((y1,y2)) > line_3[1]: 
                        line_3[1] = max((y1,y2))
                        line_3[0] = vertical_lines[index][0][np.argmax((y1,y2)) * 2]
                    if min((y1,y2)) < line_3[3]: 
                        line_3[3] = min((y1,y2))
                        line_3[2] = vertical_lines[index][0][np.argmin((y1,y2)) * 2]
                    cv2.line(line_image, (x1, y1), (x2, y2), 40, line_width)

        #horizontal = [1920,mid, 0, mid]
        #cut_horizontal = cut_top(horizontal_list, horizontal_list, mid)
        
        if horizontal_list is not None:
            for horiz_line in horizontal_list:
                for x1, y1, x2, y2 in horiz_line:
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, line_width)
                    #if min((x1, x2)) < horizontal[0]:
                    #    horizontal[0] = min((x1, x2))
                    #if max((x1, x2)) > horizontal[2]:
                    #    horizontal[2] = max((x1, x2))


        cv2.line(line_image, (line_1[0], line_1[1]),(line_1[2], line_1[3]), 255, 5)
        cv2.line(line_image, (line_2[0], line_2[1]),(line_2[2], line_2[3]), 255, 5)
        cv2.line(line_image, (line_3[0], line_3[1]),(line_3[2], line_3[3]), 255, 5)
        #cv2.line(line_image, (horizontal[0], horizontal[1]),(horizontal[2], horizontal[3]), 255, 5)

        #print(math.degrees(average_angle1 / num_1), math.degrees(average_angle2 / num_2), math.degrees(average_angle3 / num_3))
        cv2.imshow("clustered", line_image)
        cv2.waitKey(0)
    """
    # Draw vertical lines first only
    if vertical_lines is not None:
        for line in vertical_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), 255, line_width)

    if horizontal_list is not None:
        for line in horizontal_list:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), 125, line_width)
    
    """

    # since bars are pretty much vertical, can approximate length by y-coords
    y_lengths = [abs(line_1[1] - line_1[3]), abs(line_2[1] - line_2[3]), abs(line_3[1] - line_3[3])]
    short_index = np.argmin(y_lengths)
    #cv2.imshow("blur line", line_image)
    line_image = cv2.circle(line_image, (960, mid), 5, 255, thickness=5)
    line_image = cv2.circle(line_image, (960, mid - 20), 5, 125, thickness=5)
    lines = [line_1, line_2, line_3]
    lines = [[line] for index, line in enumerate(lines) if index != short_index]
    # Change to calc_heading and print result if do not want image (also remove cv2.line drawing)
    after = centering.draw_heading(line_image, lines)
    cv2.imshow("result", after)
    
    cv2.waitKey(0)
