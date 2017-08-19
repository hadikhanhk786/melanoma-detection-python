# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:28:59 2017

@author: ukalwa
"""
import os
from os.path import join

import cv2
import numpy as np

# Initialize directory locations of different results
DIR = r"G:\Upender\Melanoma Project\PH2Dataset\Python_script"
lesion_dir = r"G:\Upender\Melanoma Project\PH2Dataset\Images"
results_dir = r"G:\Upender\Melanoma Project\PH2Dataset\MoleDetection"

# Array to store results
comparison_results = []
failed_results = []

# Method to find mask area
def find_area(mask):
    im_mask, mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_NONE)
    cnt = len(mask_contours)
    if cnt > 0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            area[i] = cv2.contourArea(mask_contours[i])
            max_area_pos = np.argpartition(area, -1)[-1:][0]
        return area[max_area_pos]


# main

segmentation_results_dir = join(DIR,"Segmentation_results")
if not os.path.isdir(segmentation_results_dir):
    os.mkdir(segmentation_results_dir)
else:
    print "Directory exists"

# loop through all the lesion masks
for manual_seg_mask_name in os.listdir(lesion_dir):
    if "_lesion" in manual_seg_mask_name \
            and manual_seg_mask_name.endswith(".jpg"):
        base_name = manual_seg_mask_name[:-10]
        print base_name

        # get the name of ftc segmentation result
        ftc_seg_mask_name = base_name + "cropped_mask.PNG"

        # Image with automated segmentation contours overlaid
        automatic_seg_overlaid_image = cv2.imread(base_name +"cropped.PNG")

        # Read grayscale manual and slice it to match ftc mask
        manual_mask = cv2.imread(join(lesion_dir,manual_seg_mask_name)
                                ,0)[:,57:707]
        # Read ftc segmented mask
        ftc_mask = cv2.imread(join(results_dir,ftc_seg_mask_name),0)

        # post process image using dilation with 11x11 ellipse kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        dilated_mask = cv2.dilate(ftc_mask,kernel)

        # Calculate lesion mask areas of manual and ftc segmented masks
        ftc_mask_area = find_area(dilated_mask)
        lesion_mask_area = find_area(manual_mask)

        max_area = max(ftc_mask_area,lesion_mask_area)
        min_area = min(ftc_mask_area,lesion_mask_area)

        # Calculate error_percentage
        error_percentage = ((max_area-min_area)/lesion_mask_area)*100

        if error_percentage > 40:
            failed_results.append([base_name, lesion_mask_area, ftc_mask_area,
                                   error_percentage])
        else:
            comparison_results.append([lesion_mask_area, ftc_mask_area,
                                       error_percentage])


















