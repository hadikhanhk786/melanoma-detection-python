import cv2
import numpy as np
import matplotlib.pyplot as plt
import active_contour

img = cv2.imread("Images/IMD002_cropped.jpg")
iter_list = [25, 1]
gaussian_list = [5, 1.0]
energy_list = [1, 1, 1, 1, 1]
res = active_contour.run(img, 100, 0, 0.95, 0.95, iter_list, energy_list, gaussian_list)

print res.shape
plt.imshow(res)
# plt.show()
# plt.waitforbuttonpress(timeout=2)
# plt.close()
