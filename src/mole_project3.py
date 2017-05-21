# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 15:49:53 2017

@author: ukalwa
"""
import cv2
import numpy as np
import os
#optional argument
def nothing(x):
    pass
PATH = r'G:\Upender\Melanoma Project\PH2Dataset\Images'

filename = r'IMD002'
img = cv2.imread(os.path.join(PATH,filename+'_cropped.jpg'))
mask2 = cv2.imread(os.path.join(PATH,filename+'_lesion.jpg'),0)
rows,cols = mask2.shape
img_center_x = rows/2
img_center_y = cols/2
crop_width = 650/2
mask2 = mask2[:,img_center_y - crop_width: img_center_y + crop_width]
hsv2=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
tolerance = 20
value_threshold = np.uint8(cv2.mean(hsv2)[2]) - tolerance
print filename, cv2.mean(hsv2)

value_threshold = np.uint8(cv2.mean(hsv2)[2]) - tolerance
im_mask, mask_contours, hierarchy = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = len(mask_contours)
if cnt>0:
    area = np.zeros(cnt)
    for i in np.arange(cnt):
        area[i] = cv2.contourArea(mask_contours[i])
    max_area_pos = np.argpartition(area, -1)[-1:][0]
    contour_area = cv2.contourArea(mask_contours[max_area_pos])
        
def get_color_contour(frame, hsv, colors, COLOR):
    HSVLOW=colors[0]
    HSVHIGH=colors[1]
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
    res = cv2.bitwise_and(frame,frame, mask =mask)
    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv2.findContours(res,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = len(contours)
#    print cnt, COLOR
    if cnt>0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            a = cv2.contourArea(contours[i])
            if a > contour_area*0.01:
                area[i] = a
                    
        if np.max(area)>0:
            max_area_pos = np.argpartition(area, -2)[-2:]
    #    cv2.drawContours(frame,contours,max_area_pos,(0,255,0),2)
            return np.array(contours)[max_area_pos]
        else:
            return []
    else:
        return []
    



frame = cv2.bitwise_and(img,img,mask = mask2)
#    frame=cv2.medianBlurq(frame,3)
hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

hsv_colors = {
                'Blue Gray': [np.array([15,0,0]),np.array([179,255,value_threshold]), (0,153,0)], # Green
                'White': [np.array([0,0,value_threshold]),np.array([15,80,255]),(255,255,0)], # Cyan
                'Light Brown': [np.array([0,80,value_threshold+3]),np.array([15,255,255]),(0,255,255)], # Yellow
                'Dark Brown': [np.array([0,80,0]),np.array([15,255,value_threshold-3]),(0,0,204)], # Red
                'Black': [np.array([0,0,0]),np.array([15,140,90]),(0,0,0)], # Black
                                      
            }

no_of_colors = []
frame_for_viewing = np.copy(img)
for color in hsv_colors:
    cnt = get_color_contour(frame, hsv, hsv_colors[color], color)
    if len(cnt)>0:
        cv2.drawContours(frame_for_viewing,cnt,-1,hsv_colors[color][2],2)
        no_of_colors.append(color)
#    else:
#        print hsv_colors[color]
print "Colors found", no_of_colors
#cv2.imwrite(os.path.join(filename+'_colors.PNG'),frame_for_viewing)

cv2.imshow("Windows",frame_for_viewing)
cv2.waitKey(10000)
cv2.destroyAllWindows()