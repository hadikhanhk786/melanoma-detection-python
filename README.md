**Melanoma detection**

This application reads dermoscopic lesion images and classfies them as melanoma or beingn.
It is developed using OpenCV 3.1.0 and Python 2.7

Requirments
===========

*Environment Setup*

-   Download & Install [OpenCV 3.1.0]
-   Download & Install [Python 2.7]
-   Using pip install [numpy] , [matplotlib] and [skimage]
-   Copy cv2.pyd file from \[OPENCV\_LOCATION\]/build/python/2.7/\[x64
    or x86\]/ to \[PYTHON\_LOCATION\]/Lib/site\_packages/

It was tested on Windows and Mac OS X.

Usage
=====

Run `python src/mole_project2.py`

Steps involved
==============

The code performs following steps

-   Reads in dermoscopic lesion image and mask image
-   Segment the lesion from the dermoscopic image using mask
-   calculate ABCD features of the lesion
-   classify them using classifier (In progress)
-   Working on porting active contour c++ code to Python which was used to generate the mask images (In progress)



License
=======

This code is GNU GENERAL PUBLIC LICENSED.

Contributing
============

If you have any suggestions or identified bugs please feel free to post
them!

  [OpenCV 3.1.0]: http://opencv.org/downloads.html
  [Python 2.7]: https://www.python.org/downloads/
  [numpy]: https://www.scipy.org/scipylib/download.html
  [matplotlib]: https://matplotlib.org/
  [skimage]: http://scikit-image.org/download.html