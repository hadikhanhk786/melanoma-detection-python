
import os
import sys
os.system("scons")

import cv2

import active_contour


img = cv2.imread(r"Z:\PH2Dataset\Images\IMD035_cropped.jpg")
if img is None:
    print("Image file not found or unable to read")
    sys.exit()
img = cv2.GaussianBlur(img, (11,11),0)
iter_list = [50, 25]
gaussian_list = [7, 1.0]
energy_list = [2, 1, 1, 1, 1]
res = active_contour.run(img, 500, 0, 0.50, 0.50, iter_list, energy_list, gaussian_list)

print(res.shape)
cv2.imshow("Result",res)
cv2.waitKey(5000)
