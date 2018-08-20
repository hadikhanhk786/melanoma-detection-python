# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:28:59 2017

@author: ukalwa
"""
import os
from os.path import join

import cv2

from utilities import extract_largest_contour

# Initialize directory locations of different results
DIR = r"G:\Upender\result_set"

# Array to store results
comparison_results = []
failed_results = []

# loop through all the lesion masks
for manual_seg_mask_name in os.listdir(DIR):
    if "_lesion" in manual_seg_mask_name \
            and manual_seg_mask_name.endswith(".jpg"):
        base_name = manual_seg_mask_name[:-10]
        print base_name

        # get the name of ftc segmentation result
        ftc_seg_mask_name = base_name + "cropped_mask.PNG"

        # Image with automated segmentation contours overlaid
        automatic_seg_overlaid_image = cv2.imread(base_name + "cropped.PNG")

        # Read grayscale manual and slice it to match ftc mask
        manual_mask = cv2.imread(join(DIR, manual_seg_mask_name)
                                 , 0)[:, 57:707]
        # Read ftc segmented mask
        ftc_mask = cv2.imread(join(DIR, ftc_seg_mask_name), 0)

        # Calculate lesion mask areas of manual and ftc segmented masks
        contours, pos = extract_largest_contour(ftc_mask)
        ftc_mask_area = cv2.contourArea(contours[pos])
        contours, pos = extract_largest_contour(manual_mask)
        lesion_mask_area = cv2.contourArea(contours[pos])

        max_area = max(ftc_mask_area, lesion_mask_area)
        min_area = min(ftc_mask_area, lesion_mask_area)

        # Calculate error_percentage
        error_percentage = ((max_area - min_area) / max_area) * 100

        if error_percentage > 100:
            failed_results.append([base_name, lesion_mask_area, ftc_mask_area,
                                   error_percentage])
        else:
            comparison_results.append([lesion_mask_area, ftc_mask_area,
                                       error_percentage])
