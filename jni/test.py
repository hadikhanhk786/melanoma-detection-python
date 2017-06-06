import cv2
import numpy as np
import matplotlib.pyplot as plt
import active_contour
plt.ion()
img = cv2.imread("../Images/IMD002_cropped.jpg")
zero_image = np.zeros(img.shape[:2], dtype=np.uint8)

res = active_contour.run(img,zero_image)

print res.shape
plt.imshow(res)
plt.show()
plt.waitforbuttonpress