# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 12:01:39 2017

@author: ukalwa
"""
import cv2
import numpy as np
import os
import json
from scipy.interpolate import splprep, splev
#optional argument
def nothing(x):
    pass

def get_features(mask, contour,PATH):
    moments = cv2.moments(contour)
#            contour_area = moments['m00']
    (mH, mW) = mask.shape[:2]

    contour_area = cv2.countNonZero(mask)
    try:
        contour_centroid = [int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])]
        contour_perimeter = cv2.arcLength(contour, True)

        # Get max and min diameter
        xx,yy,w,h = np.int0(cv2.boundingRect(contour))
#        cv2.rectangle(mask,(xx,yy),(xx+w,yy+h),(255,255,255),2)
        rect = cv2.fitEllipse(contour)
        (x,y) = rect[0]
        (w,h) = rect[1]
        angle = rect[2]
#        box = np.int0(cv2.boxPoints(rect))
#        cv2.drawContours(mask,[box],0,(255,255,255,2))

        if w < h:
            if angle < 90:
                angle -= 90
            else:
                angle +=90
        rows, cols = mask.shape
        rot = cv2.getRotationMatrix2D((x,y), angle, 1)
        cos = np.abs(rot[0, 0])
        sin = np.abs(rot[0, 1])
#        print(rot)
        nW = int((rows * sin) + (cols * cos))
        nH = int((rows * cos) + (cols * sin))

        rot[0, 2] += (nW / 2) - cols/2
        rot[1, 2] += (nH / 2) - rows/2
        warp_img = cv2.warpAffine(mask, rot, (nH,nW))

        im_mask, cnts, hierarchy = cv2.findContours(warp_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in cnts]
        contour = cnts[np.argmax(areas)]
        xx,yy,nW,nH = cv2.boundingRect(contour)
#        cv2.rectangle(warp_img,(xx,yy),(xx+w,yy+h),(255,255,255),2)
        warp_img = warp_img[yy:yy+nH, xx:xx+nW]

        # get horizontal asymmetry
        flipContourHorizontal = cv2.flip (warp_img, 1)
        flipContourVertical = cv2.flip(warp_img,  0)

        diff_horizontal = cv2.compare(warp_img,flipContourHorizontal,cv2.CV_8UC1)
        diff_vertical = cv2.compare(warp_img,flipContourVertical,cv2.CV_8UC1)

        cv2.imwrite(PATH+'_roi_vertical.PNG', diff_vertical)
        cv2.imwrite(PATH+'_roi_horizontal.PNG', diff_horizontal)

        diff_horizontal = cv2.bitwise_not(diff_horizontal)
        diff_vertical = cv2.bitwise_not(diff_vertical)

        h_asym = cv2.countNonZero(diff_horizontal)
        v_asym = cv2.countNonZero(diff_vertical)

        return {'area':int(contour_area), 'centroid':contour_centroid, 'perimeter':int(contour_perimeter), \
                'diam':np.int0([nW,nH]).tolist(), 'asym':[h_asym, v_asym]}
    except:
        return {}


def get_color_contour(frame, hsv, colors, COLOR):
    global contour_area
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
            if a > contour_area*0.02:
                area[i] = a

        if np.max(area)>0:
            max_area_pos = np.argpartition(area, -2)[-2:]
    #    cv2.drawContours(frame,contours,max_area_pos,(0,255,0),2)
            return np.array(contours)[max_area_pos]
        else:
            return []
    else:
        return []

def sq_dist_two_vectors(p, q):
    return np.square(p[:, 0] - q[:, 0]) + np.square(p[:, 1] - q[:, 1])


def smooth_borders(a, smoothing_coefficient):
    for i in range(smoothing_coefficient):
        b = np.roll(a, 1, axis=0)
        c = np.roll(a, -1, axis=0)
        a = (a + b + c) / 3
    return a, b, c


def menger_curve_array(a, b, c):
    #    a = a[:-2,:]
    #    b = b[:-2,:]
    #    c = c[:-2,:]
    curvature_top = 2 * (
    np.multiply(a[:, 0] - b[:, 0], c[:, 1] - b[:, 1]) - np.multiply(
        c[:, 0] - b[:, 0], a[:, 1] - b[:, 1]))
    curvature_bottom = np.sqrt(
        np.multiply(sq_dist_two_vectors(a, b), sq_dist_two_vectors(b, c),
                    sq_dist_two_vectors(c, a)))
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.true_divide(curvature_top, curvature_bottom)
        curvature[curvature == np.inf] = 0
        #        curvature = np.concatenate(([0],np.nan_to_num(curvature),[0]))
        return curvature

def smooth_boundary(boundary):
    pts = boundary[:, 0, :]
    tck, u = splprep(pts.T, u=None, s=np.float(len(pts) + 1), per=1, quiet=3)
    u_new = np.linspace(u.min(), u.max(), len(pts) + 1)
    new_points = splev(u_new, tck, der=0)
    #    print np.array([new_points[0][:-1],new_points[1][:-1]]).T.shape
    smoothened_boundary = np.array([new_points[0][:-1], new_points[1][:-1]]).T
    # smoothened_boundary,_,_ = smooth_borders(smoothened_boundary,
    #                                          smoothing_coefficient)
    # print smoothened_boundary.shape, len(pts),1,2
    smoothened_boundary_opencv_format = [
        np.reshape(np.int32(np.around(smoothened_boundary)), [len(pts), 1, 2])]
    return smoothened_boundary, smoothened_boundary_opencv_format

PATH = r'G:\Upender\Melanoma Project\PH2Dataset\MoleDetection'
failed_files = []
target = open(os.path.join(PATH,'results5.txt'), 'w')
for filename in os.listdir(PATH):
    if "lesion" not in filename and "Label" not in filename \
    and "IMD002" in filename and "_cropped" in filename \
    and filename.endswith('.jpg'):
        print filename
        base_file, _ = os.path.splitext(filename)
#        filename = r'IMD020'
        img = cv2.imread(os.path.join(PATH,base_file+'.jpg'))
        if os.path.exists(os.path.join(PATH,base_file+'_mask.PNG')):# and os.path.join(PATH,base_file+'_mask_warp.PNG'):
            mask2 = cv2.imread(os.path.join(PATH,base_file+'_mask.PNG'),0)
#            warp_mask = cv2.imread(os.path.join(PATH,base_file+'_mask_warp.PNG'),0)
#            if warp_mask.shape[0]%2!=0:
#                continue
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
            feature_set = get_features(mask2,mask_contours[max_area_pos], os.path.join(PATH,base_file))
#            a = np.reshape(mask_contours[max_area_pos],
#                           [len(mask_contours[max_area_pos]),2])
            smoothened_boundary, _ = smooth_boundary(mask_contours[max_area_pos])
            a = smoothened_boundary
            b = np.roll(a, 1, axis=0)
            c = np.roll(a, -1, axis=0)
            k = menger_curve_array(a, b, c)
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
                        'White': [np.array([0,0,145]),np.array([15,80,value_threshold]),(255,255,0)], # Cyan
                        'Light Brown': [np.array([0,80,value_threshold+3]),np.array([15,255,255]),(0,255,255)], # Yellow
                        'Dark Brown': [np.array([0,80,0]),np.array([15,255,value_threshold-3]),(0,0,204)], # Red
                        'Black': [np.array([0,0,0]),np.array([15,140,90]),(0,0,0)], # Black

                    }

        no_of_colors = []
        frame_for_viewing = np.copy(img)
        for color in hsv_colors:
#            print color
            cnt = get_color_contour(frame, hsv, hsv_colors[color], color)
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
        cv2.imwrite(os.path.join(PATH,base_file+'_colors.PNG'),frame_for_viewing)
#        target.write(filename +' , '+ str(len(no_of_colors)) + ' , ' + str(feature_set)+'\n')
        target.write(json.dumps(feature_set)+'\n')

target.close()
#cv2.imshow("Windows",frame_for_viewing)
#cv2.waitKey(10000)
#cv2.destroyAllWindows()