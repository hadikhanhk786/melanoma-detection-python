# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 11:37:30 2017

@author: ukalwa
"""
# third-party imports
import numpy as np
from scipy.interpolate import splprep, splev
import cv2


def sq_dist_two_vectors(p, q):
    if len(p.shape) == 1 and len(q.shape) == 1:
        return np.square(p[0] - q[0]) + np.square(p[1] - q[1])
    elif len(p.shape) == 1:
        return np.square(p[0] - q[:, 0]) + np.square(p[1] - q[:, 1])
    elif len(q.shape) == 1:
        return np.square(p[:, 0] - q[0]) + np.square(p[:, 1] - q[1])
    else:
        return np.square(p[:, 0] - q[:, 0]) + np.square(p[:, 1] - q[:, 1])


def smooth_borders(a, smoothing_coefficient):
    b = np.roll(a, 1, axis=0)
    c = np.roll(a, -1, axis=0)
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
            np.multiply(a[:, 0] - b[:, 0], c[:, 1] - b[:, 1])
            - np.multiply(c[:, 0] - b[:, 0], a[:, 1] - b[:, 1]))
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


def draw_closing_lines(img, contours):
    pts = []
    rows, cols = None, None
    for cont in contours:
        v1 = (np.roll(cont, -2, axis=0) - cont)
        v2 = (np.roll(cont, 2, axis=0) - cont)
        dotprod = np.sum(v1 * v2, axis=2)
        norm1 = np.sqrt(np.sum(v1 ** 2, axis=2))
        norm2 = np.sqrt(np.sum(v2 ** 2, axis=2))
        np.seterr(divide='ignore', invalid='ignore')
        cosinus = (dotprod / norm1) / norm2
        cosinus[np.isnan(cosinus)] = 0
        indexes = np.where(0.90 < cosinus)[0]
        rows, cols = img.shape[:2]
        if len(indexes) == 0:
            continue
        print "Found possible breaks at : %s locations" % len(indexes)
        print "Points are : ", cont[indexes, 0]
        for i in xrange(len(indexes)):
            pt = cont[indexes[i], 0]
            if 0 <= pt[0] < 10 or 0 <= cols - pt[0] <= 10:
                pts.append(tuple(pt))
            elif 0 <= pt[1] < 10 or 0 <= rows - pt[1] <= 10:
                pts.append(tuple(pt))

    print "Found breaks at : %s locations" % len(pts)
    print "Points are : ", pts
    pts = sorted(pts)
    if len(pts) == 0:
        return
    elif len(pts) == 2:
        # two u-turns found, draw the closing line
        diff = np.diff(pts, axis=0)
        if 0 <= np.min(diff) <= 10:
            cv2.line(img, pts[0], pts[1], (255, 255, 255))
        else:
            # condition where end points are on different axes
            # Add additional point to close the contour smoothly
            pt3 = []
            if 0 <= pts[0][0] < 10 or 0 <= cols - pts[0][0] < 10:
                pt3.append(pts[0][0])
            else:
                pt3.append(pts[1][0])
            if 0 <= pts[0][1] < 10 or 0 <= cols - pts[0][1] < 10:
                pt3.append(pts[0][1])
            else:
                pt3.append(pts[1][1])
            cv2.line(img, pts[0], tuple(pt3), (255, 255, 255))
            cv2.line(img, tuple(pt3), pts[1], (255, 255, 255))
    elif len(pts) > 2:
        print "Manual check required"
        paired_pts = []
        #        pts = sorted(pts)
        for pt in pts:
            if pt not in paired_pts:
                temp_pts = list(set(pts) - set(paired_pts))
                if len(temp_pts) == 1:
                    continue
                dist = sq_dist_two_vectors(np.array(pt), np.array(temp_pts))
                pos = np.argpartition(dist, 1)[1]
                pt2 = temp_pts[pos]
                paired_pts.append(pt)
                paired_pts.append(pt2)
                print "pairs formed:", pt, pt2
                cv2.line(img, pt, pt2, (255, 255, 255))


def extract_largest_contour(gray_image):
    im_mask, mask_contours, hierarchy = \
        cv2.findContours(gray_image, cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_NONE)
    cnt = len(mask_contours)
    if cnt > 0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            area[i] = cv2.contourArea(mask_contours[i])
        max_area_pos = np.argpartition(area, -1)[-1:][0]
        return [mask_contours, max_area_pos]
    else:
        return []
