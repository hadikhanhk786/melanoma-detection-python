#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:03:33 2017

@author: upenderkalwa
"""
import cv2
from active_contour_ftc import ActiveContourFTC
#import os
#import matplotlib.pyplot as plt
import numpy as np

filepath = '../Images/IMD002_cropped.PNG'
img = cv2.imread(filepath)
ftc = ActiveContourFTC(img,True, 0.95,0.95, 0.0, 0.0, True, 5, 1.0, True, 50,
                       1, 1, 1, 1, 1, 1)
dummy_image = np.zeros(img.shape[:2],dtype=np.uint8)
ftc.initialize_params()
L_in = ftc.L_in
L_out = ftc.L_out
interior_points = ftc.interior_points
exterior_points = ftc.exterior_points
phi = ftc.phi

#ftc.evolve_one_iteration()
#dummy_image[ftc.L_out[:,0],ftc.L_out[:,1]] = 255

ftc.evolve_n_iterations(10)
dummy_image[ftc.L_out[:,0],ftc.L_out[:,1]] = 255

L_in = ftc.L_in
L_out = ftc.L_out
interior_points = ftc.interior_points
exterior_points = ftc.exterior_points
