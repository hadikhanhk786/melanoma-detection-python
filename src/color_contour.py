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
    #    print cnt, COLOR
    if cnt > 0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            a = cv2.contourArea(contours[i])
            if a > max_area * 0.02:
                area[i] = a

        if np.max(area) > 0:
            if len(area) > 1:
                max_area_pos = np.argpartition(area, -2)[-2:]
            else:
                max_area_pos = 0
            #    cv2.drawContours(frame,contours,max_area_pos,(0,255,0),2)
            return np.array(contours)[max_area_pos]
        else:
            return []
    else:
        return []
