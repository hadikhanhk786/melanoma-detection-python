import cv2
import numpy as np
import matplotlib.pyplot as plt
import active_contour

# img = cv2.imread("Images/IMD036_cropped.jpg")
img = cv2.imread(r"G:\Upender\Melanoma Project\PH2Dataset\Images\IMD035_cropped.jpg")
img = cv2.GaussianBlur(img, (11,11),0)
iter_list = [50, 25]
gaussian_list = [7, 1.0]
energy_list = [2, 1, 1, 1, 1]
res = active_contour.run(img, 500, 0, 0.50, 0.50, iter_list, energy_list, gaussian_list)

print res.shape
cv2.imshow("Result",res)
cv2.waitKey(5000)
