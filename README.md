**Lipid Tracking**

This application reads microscope image and tracks fluorescent lipids
using OpenCV 3.1 and Python

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

Run `python src/lipid_tracking.py`

Steps involved
==============

The code tracks fluorescent lipids by following steps

-   Detect all the lipids in every frame
-   Eliminate undocked lipids detected and track only docked frames
-   Handle the lipids burst
-   Handle the background changes caused due to lipids burst

Here are some of the snapshots
==============================

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