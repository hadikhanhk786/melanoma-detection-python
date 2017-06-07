Melanoma detection
==================

This application reads dermoscopic lesion images and classfies them as melanoma or benign.
It is developed using OpenCV 3.2.0 and Python 2.7

Requirments
===========

*Environment Setup*

-   Download & Install [OpenCV 3.2.0]
-   Download & Install [Python 2.7]
-   Using pip install [numpy]
-   Copy cv2.pyd file from \[OPENCV\_LOCATION\]/build/python/2.7/\[x64
    or x86\]/ to \[PYTHON\_LOCATION\]/Lib/site\_packages/

It was tested on Windows and Mac OS X.

Usage
=====

Run `python src/run_script.py [--file <filename> | --dir <dirname>]`

Steps involved
==============

The code performs following steps

-   Reads in dermoscopic lesion image specified by --file or a directory name specified by --dir
-   If directory name is mentioned, the code checks the validity of path and loops through the ".jpg" files present in the directory
-   Checks the validity of the image
-   Get the mask by calling active_contour library (in C++) with number of iterations
-   Segment the lesion from the dermoscopic image using mask
-   Calculate ABCD (<i>Asymmetry, Border irregularity, number of Colors, Diameter</i>) features of the lesion
-   Classify them using classifier (In progress)
-   Save the processed images and results



License
=======

This code is GNU GENERAL PUBLIC LICENSED.

Contributing
============

If you have any suggestions or identified bugs please feel free to post
them!

  [OpenCV 3.2.0]: http://opencv.org/downloads.html
  [Python 2.7]: https://www.python.org/downloads/
  [numpy]: https://www.scipy.org/scipylib/download.html