import cv2
import numpy as np
import matplotlib.pyplot as plt
import active_contour

img = cv2.imread("Images/IMD002_cropped.jpg")

res = active_contour.run(img, 100)

print res.shape
plt.imshow(res)
plt.show()
plt.waitforbuttonpress()
