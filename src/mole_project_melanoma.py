# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:14:47 2017

@author: ukalwa
"""
import cv2
import numpy as np
import os
import json

import color_contour
import features

PATH = r'G:\Upender\Melanoma Project\PH2Dataset\Images for figures\Melanoma'

failed_files = []
target = open(os.path.join(PATH,'results_melanoma.txt'), 'w')
for name in os.listdir(PATH):
    if '_cropped' not in name and os.path.isfile(os.path.join(PATH,name)) and name.endswith('.jpg') \
    and '_lesion' not in name:
        filename, file_extension = os.path.splitext(name)
        img = cv2.imread(os.path.join(PATH,filename+'.jpg'))
        if os.path.exists(os.path.join(PATH,filename+'_lesion.bmp')):# and os.path.join(PATH,base_file+'_mask_warp.PNG'):
            mask2 = cv2.imread(os.path.join(PATH,filename+'_lesion.bmp'),0)
        else:
            failed_files.append(filename)
            continue

        im_mask, mask_contours, hierarchy = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = len(mask_contours)
        if cnt>0:
            area = np.zeros(cnt)
            for i in np.arange(cnt):
                area[i] = cv2.contourArea(mask_contours[i])
            max_area_pos = np.argpartition(area, -1)[-1:][0]
            contour_area = cv2.contourArea(mask_contours[max_area_pos])
            feature_set = features.extract(mask2,mask_contours[max_area_pos], os.path.join(PATH,filename))
        if cnt<=0:
            failed_files.append(filename)
            continue

        if len(feature_set)==0:
            failed_files.append(filename)
            continue
        hsv2=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        tolerance = 30
        value_threshold = np.uint8(cv2.mean(hsv2)[2]) - tolerance
#        print filename, cv2.mean(hsv2)
#        print feature_set


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
#            print color
            cnt = color_contour.extract(frame, hsv, hsv_colors[color],
                                        contour_area)
#            print cnt1
            centroid =[]
            dist = []
            color_attr = {}
            if len(cnt)>0:
                for contour in cnt:
                    moments = cv2.moments(contour)
                    if moments['m00'] == 0:
                        print color
                        continue
                    color_ctrd = [int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])]
                    color_dist = ((color_ctrd[0]-feature_set['centroid'][0])**2 + (color_ctrd[1]-feature_set['centroid'][1])**2)**0.5
                    if color_dist > 0.6*np.min(np.array(feature_set['diam'][-2:])/2) and color == 'White': # White cannot be at the center
                        continue
                    dist.append(color_dist)
                    centroid.append(color_ctrd)
            if len(dist) != 0 and len(centroid) != 0:
                cv2.drawContours(frame_for_viewing,cnt,-1,hsv_colors[color][2],2)
                color_attr['color'] = color
                color_attr['centroids'] = centroid
                color_attr['dist'] = np.int0(dist).tolist()
                no_of_colors.append(color_attr)
        #    else:
        #        print hsv_colors[color]
        feature_set['image'] = filename
        feature_set['colors_attr'] = no_of_colors
        feature_set['no_of_colors'] = len(no_of_colors)
        print "Features extracted", feature_set
        cv2.imwrite(os.path.join(PATH,filename+'_colors.PNG'),frame_for_viewing)
#        target.write(filename +' , '+ str(len(no_of_colors)) + ' , ' + str(feature_set)+'\n')
        target.write(json.dumps(feature_set)+'\n')

target.close()
#cv2.imshow("Windows",frame_for_viewing)
#cv2.waitKey(10000)
#cv2.destroyAllWindows()