# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 11:29:16 2017

@author: ukalwa
"""
import cv2
import numpy as np


def extract(mask, contour, path):
    moments = cv2.moments(contour)
    #            contour_area = moments['m00']
    # (mH, mW) = mask.shape[:2]

    contour_area = cv2.countNonZero(mask)
    try:
        contour_centroid = [int(moments['m10'] / moments['m00']),
                            int(moments['m01'] / moments['m00'])]
        contour_perimeter = cv2.arcLength(contour, True)

        # Get max and min diameter
        # xx, yy, w, h = np.int0(cv2.boundingRect(contour))
        #        cv2.rectangle(mask,(xx,yy),(xx+w,yy+h),(255,255,255),2)
        rect = cv2.fitEllipse(contour)
        (x, y) = rect[0]
        (w, h) = rect[1]
        angle = rect[2]
        #        box = np.int0(cv2.boxPoints(rect))
        #        cv2.drawContours(mask,[box],0,(255,255,255,2))

        if w < h:
            if angle < 90:
                angle -= 90
            else:
                angle += 90
        rows, cols = mask.shape
        rot = cv2.getRotationMatrix2D((x, y), angle, 1)
        cos = np.abs(rot[0, 0])
        sin = np.abs(rot[0, 1])
        #        print(rot)
        nW = int((rows * sin) + (cols * cos))
        nH = int((rows * cos) + (cols * sin))

        rot[0, 2] += (nW / 2) - cols / 2
        rot[1, 2] += (nH / 2) - rows / 2
        warp_img = cv2.warpAffine(mask, rot, (nH, nW))

        im_mask, cnts, hierarchy = cv2.findContours(warp_img, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in cnts]
        contour = cnts[np.argmax(areas)]
        xx, yy, nW, nH = cv2.boundingRect(contour)
        #        cv2.rectangle(warp_img,(xx,yy),(xx+w,yy+h),(255,255,255),2)
        warp_img = warp_img[yy:yy + nH, xx:xx + nW]

        # get horizontal asymmetry
        flipContourHorizontal = cv2.flip(warp_img, 1)
        flipContourVertical = cv2.flip(warp_img, 0)

        diff_horizontal = cv2.compare(warp_img, flipContourHorizontal,
                                      cv2.CV_8UC1)
        diff_vertical = cv2.compare(warp_img, flipContourVertical, cv2.CV_8UC1)

#        cv2.imwrite(path + '_roi_vertical.PNG', diff_vertical)
#        cv2.imwrite(path + '_roi_horizontal.PNG', diff_horizontal)

        diff_horizontal = cv2.bitwise_not(diff_horizontal)
        diff_vertical = cv2.bitwise_not(diff_vertical)

        h_asym = cv2.countNonZero(diff_horizontal)
        v_asym = cv2.countNonZero(diff_vertical)

        return [{'area': int(contour_area), 'centroid': contour_centroid,
                'perimeter': int(contour_perimeter),
                'diam': np.int0([nW, nH]).tolist(), 'asym': [h_asym, v_asym]},
                cv2.bitwise_not(diff_horizontal),
                cv2.bitwise_not(diff_vertical)]
    except:
        return {}
