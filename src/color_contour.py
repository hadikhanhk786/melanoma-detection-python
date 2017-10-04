# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 11:31:01 2017

@author: ukalwa
"""
import cv2
import numpy as np


def extract(frame, hsv, colors, max_area):
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
            if a > max_area * 0.02:
                new_contours.append(contours[i])

        return np.array(new_contours)
    else:
        return []
