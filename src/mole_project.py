# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 12:01:39 2017

@author: ukalwa
"""
import cv2
import numpy as np
import os
import posixpath
import json
# import Tkinter as tk
import tkFileDialog as filedialog

import color_contour
import features
import active_contour

# import utilities

# PATH = r'G:\Upender\Melanoma Project\PH2Dataset\MoleDetection'
failed_files = []
PROCESS_DIR = False
results = []
value_threshold = 150
hsv_colors = {
    'Blue Gray': [np.array([15, 0, 0]), np.array([179, 255, value_threshold]),
                  (0, 153, 0)],  # Green
    'White': [np.array([0, 0, 145]), np.array([15, 80, value_threshold]),
              (255, 255, 0)],  # Cyan
    'Light Brown': [np.array([0, 80, value_threshold + 3]),
                    np.array([15, 255, 255]), (0, 255, 255)],  # Yellow
    'Dark Brown': [np.array([0, 80, 0]),
                   np.array([15, 255, value_threshold - 3]),
                   (0, 0, 204)],  # Red
    'Black': [np.array([0, 0, 0]), np.array([15, 140, 90]),
              (0, 0, 0)],  # Black
}


def write_to_file(file_name, save_list):
    target = open(file_name, 'w')
    target.write(json.dumps(save_list, sort_keys=True, indent=2) + '\n')
    target.close()


def draw_mask(img, iterations=9999):
    borders = 2
    # rows, cols, dim = img.shape
    img2 = cv2.copyMakeBorder(img, borders, borders, borders, borders,
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])
    mask = np.zeros(img2.shape[:2], dtype=np.uint8)
    temp_mask = active_contour.run(img2, iterations)
    im_mask, mask_contours, hierarchy = cv2.findContours(temp_mask,
                                                         cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_NONE)
    cnt = len(mask_contours)
    if cnt > 0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            area[i] = cv2.contourArea(mask_contours[i])
        max_area_pos = np.argpartition(area, -1)[-1:][0]
        cv2.drawContours(mask, mask_contours, max_area_pos, (255, 255, 255),
                         -1)
    if cnt <= 0:
        return
    mask = mask[2:-2, 2:-2]
    im_mask, mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_NONE)
    cnt = len(mask_contours)
    if cnt > 0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            area[i] = cv2.contourArea(mask_contours[i])
        max_area_pos = np.argpartition(area, -1)[-1:][0]
        return [mask, max_area_pos, mask_contours]
    if cnt <= 0:
        return


def process(full_file_path, failed, result):
    global value_threshold, hsv_colors
    base_file, _ = os.path.splitext(full_file_path)
    #        filename = r'IMD020'
    img = cv2.imread(full_file_path)

    # if os.path.exists(base_file+'_mask.PNG'):
    #     mask2 = cv2.imread(base_file+'_mask.PNG',0)
    #     warp_mask = cv2.imread(os.path.join(PATH,base_file+'_mask_warp.PNG'),
    #                            0)
    #     if warp_mask.shape[0]%2!=0:
    #         continue
    # else:
    #     failed_files.append(filename)
    #     return
    returnVars = draw_mask(img, 100)
    if returnVars is None:
        return
    mask2, max_area_pos, mask_contours = returnVars
    contour_area = cv2.contourArea(mask_contours[max_area_pos])
    feature_set = features.extract(mask2, mask_contours[max_area_pos],
                                   base_file)

    im_mask, mask_contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_NONE)
    cnt = len(mask_contours)
    if cnt > 0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            area[i] = cv2.contourArea(mask_contours[i])
        max_area_pos = np.argpartition(area, -1)[-1:][0]
        contour_area = cv2.contourArea(mask_contours[max_area_pos])
        feature_set = features.extract(mask2, mask_contours[max_area_pos],
                                       base_file)
        # a = np.reshape(mask_contours[max_area_pos],
        #                [len(mask_contours[max_area_pos]), 2])
        # smoothened_boundary, _ = utilities.smooth_boundary(
        #     mask_contours[max_area_pos])
        # a = smoothened_boundary
        # b = np.roll(a, 1, axis=0)
        # c = np.roll(a, -1, axis=0)
        # k = utilities.menger_curve_array(a, b, c)
    if cnt <= 0:
        failed_files.append(filename)
        return

    if len(feature_set) == 0:
        failed.append(full_file_path)
        return
    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tolerance = 30
    value_threshold = np.uint8(cv2.mean(hsv2)[2]) - tolerance
    #        print filename, cv2.mean(hsv2)
    #        print feature_set

    frame = cv2.bitwise_and(img, img, mask=mask2)
    #    frame=cv2.medianBlurq(frame,3)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    no_of_colors = []
    frame_for_viewing = np.copy(img)
    for color in hsv_colors:
        #            print color
        cnt = color_contour.extract(frame, hsv, hsv_colors[color],
                                    contour_area)
        #            print cnt1
        centroid = []
        dist = []
        color_attr = {}
        if len(cnt) > 0:
            for contour in cnt:
                moments = cv2.moments(contour)
                if moments['m00'] == 0:
                    print color
                    continue
                color_ctrd = [int(moments['m10'] / moments['m00']),
                              int(moments['m01'] / moments['m00'])]
                color_dist = ((color_ctrd[0] - feature_set['centroid'][
                    0]) ** 2 + (color_ctrd[1] - feature_set['centroid'][
                        1]) ** 2) ** 0.5
                if color_dist > 0.6 * np.min(np.array(feature_set['diam'][
                                                      -2:]) / 2) \
                        and color == 'White':  # White cannot be at the center
                    continue
                dist.append(color_dist)
                centroid.append(color_ctrd)
        if len(dist) != 0 and len(centroid) != 0:
            cv2.drawContours(frame_for_viewing, cnt, -1, hsv_colors[color][2],
                             2)
            color_attr['color'] = color
            color_attr['centroids'] = centroid
            color_attr['dist'] = np.int0(dist).tolist()
            no_of_colors.append(color_attr)
    # else:
    #        print hsv_colors[color]
    feature_set['image'] = full_file_path
    feature_set['colors_attr'] = no_of_colors
    feature_set['no_of_colors'] = len(no_of_colors)
    print "Features extracted", feature_set
    result.append(feature_set)
    cv2.imwrite(base_file + '_colors.PNG', frame_for_viewing)
    write_to_file(base_file + ".json", feature_set)

    return result


# cv2.imshow("Windows",frame_for_viewing)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

if __name__ == "__main__":
    if PROCESS_DIR:
        dir_path = filedialog.askdirectory(
            initialdir=r'G:\Upender\Melanoma Project\PH2Dataset')
        if len(dir_path) != 0:
            for name in os.listdir(dir_path):
                if not os.path.isdir(
                        name) and "lesion" not in name \
                        and "Label" not in name \
                        and "_cropped" in name \
                        and name.endswith('.jpg'):
                    filename = posixpath.join(dir_path, name)
                    print "FILE: %s" % filename
                    return_vars = process(filename, failed_files, results)
                    if return_vars is not None:
                        results.append(return_vars)
        else:
            print "Invalid directory"
    else:
        file_path = filedialog.askopenfilename(
            initialdir=r'G:\Upender\Melanoma Project\PH2Dataset')
        if len(file_path) != 0:
            return_vars = process(file_path, failed_files, results)
        else:
            print "Invalid file"
