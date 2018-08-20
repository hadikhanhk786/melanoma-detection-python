# -*- coding: utf-8 -*-
"""
GUI program to identify the color using HSV track bars.

Created on Tue Mar 07 11:25:21 2017

@author: ukalwa
"""
# Built-in imports
import os

# third-party imports
import cv2
import numpy as np


# optional argument
def nothing(x):
    pass


def onmouse(event, x, y, flags, params):
    """
    Handler method to capture mouse events and store (x,y) coordinates.

    :param event: cv2 events such as EVENT_LBUTTONDOWN etc.,
    :param x: x coordinate of the mouse
    :param y: y coordinate of the mouse
    :param flags: Any
    :param params: Any
    """
    global point, hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        point = [x, y]
        print x, y, hsv[x, y]


PATH = '../Images'
point = []
filename = r'IMD002'
img = cv2.imread(os.path.join(PATH, filename + '.jpg'))
mask2 = cv2.imread(os.path.join(PATH, filename + '_mask.PNG'), 0)
hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
tolerance = 30
value_threshold = np.uint8(cv2.mean(hsv2)[2]) - tolerance
print "BGR mean: ", cv2.mean(img)
print "hsv mean: ", cv2.mean(hsv2)
# hsv2=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# tolerance = 20
# value_threshold = np.uint8(cv2.mean(hsv2)[2]) - tolerance
# print filename, cv2.mean(hsv2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', onmouse)

# easy assigments
hh = 'Hue High'
hl = 'Hue Low'
sh = 'Saturation High'
sl = 'Saturation Low'
vh = 'Value High'
vl = 'Value Low'

cv2.createTrackbar(hl, 'image', 140, 255, nothing)
cv2.createTrackbar(hh, 'image', 165, 255, nothing)
cv2.createTrackbar(sl, 'image', 135, 255, nothing)
cv2.createTrackbar(sh, 'image', 255, 255, nothing)
cv2.createTrackbar(vl, 'image', 125, 255, nothing)
cv2.createTrackbar(vh, 'image', 215, 255, nothing)

while True:
    frame = cv2.bitwise_and(img, img, mask=mask2)
    # convert to HSV from BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # read trackbar positions for all
    hul = cv2.getTrackbarPos(hl, 'image')
    huh = cv2.getTrackbarPos(hh, 'image')
    sal = cv2.getTrackbarPos(sl, 'image')
    sah = cv2.getTrackbarPos(sh, 'image')
    val = cv2.getTrackbarPos(vl, 'image')
    vah = cv2.getTrackbarPos(vh, 'image')
    # make array for final values
    hsv_low = np.array([hul, sal, val])
    hsv_high = np.array([huh, sah, vah])

    # apply the range on a mask
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('image', res)
    #    cv2.imshow('yay', frame)
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
