# -*- coding: utf-8 -*-
"""
Methods related to extract color information from images.

Created on Tue Jun 06 11:31:01 2017

@author: ukalwa
"""
# third-party imports
import cv2
import numpy as np


def extract(frame, hsv, colors, max_area):
    """
    Extract the color contour of a region determined by upper and lower
    threshold values of colors.

    :param frame:  3-d numpy array of an RGB image
    :param hsv: 3-d numpy array of an HSV image
    :param colors: A list of tuples containing upper and lower color thresholds
    :param max_area: Total object area
    :return: numpy array of contour points if color is detected else None
    """
    hsv_low = colors[0]
    hsv_high = colors[1]
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_NONE)
    cnt = len(contours)
    if cnt > 0:
        new_contours = []
        for i in np.arange(cnt):
            a = cv2.contourArea(contours[i])
            # To determine the color is not spurious noise
            if a > max_area * 0.02:
                new_contours.append(contours[i])

        return np.array(new_contours)
    else:
        return None
