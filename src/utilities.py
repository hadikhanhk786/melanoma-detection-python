# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 11:37:30 2017

@author: ukalwa
"""
import numpy as np
from scipy.interpolate import splprep, splev


def sq_dist_two_vectors(p, q):
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
