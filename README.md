# Melanoma detection

This application reads dermoscopic lesion images and classfies them as melanoma or benign.
It is developed using OpenCV 3.2.0 and Python 2.7

## Environment Setup

- Download & Install [Python] using [Anaconda] or [Miniconda] 
(**Recommended**)

- Then you can run the following commands install required packages

```bash
  sudo install python-pip # if pip is not installed (Linux only)
  pip install opencv-python==3 numpy scikit-learn scipy scons
  # (or)
  conda install -y opencv=3.2 numpy scikit-learn scipy scons -c conda-forge
```

- Clone this repository and change directory

```bash
  git clone https://github.com/ukalwa/melanoma_project_python
  cd melanoma_project_python
```

- Install prerequisites for active contour (C++) module (Windows)
  
  - Download and extract boost-python 1.62.0 either from official website or unofficial prebuilt [binaries] and set an environment variable (BOOST_DIR) pointing to the root of the extracted folder.

  - Download and extract [OpenCV 2.4.10] libraries and set an environment variable (OPENCV_DIR) pointing to `{folder path}\build\x64\vc12`.

- Create the `active_contour` module by running `scons` in the terminal in the cloned folder.

*It is compatible with both Python 2.7 and Python 3.5 tested on Windows.*

## Usage

The usage of the program is as mentioned below

```bash
python run_script.py [--file <filename> | --dir <dirname>]
```

## Steps involved

The code performs following steps:

1. Reads in dermoscopic lesion image specified by --file or a directory name specified by --dir
2. Preprocess the image by applying color transformation and filtering
3. Segment the lesion from the image using active contour model
4. Extract features (*Asymmetry, Border irregularity, Colors, Diameter*) from the lesion
5. Classify the lesion based on the features extracted using an SVM classifier and output the result.
6. Save the processed images and results

## Documentation

For generating documentation, please follow these steps:

- Make sure you have sphinx installed, you can install it like this

```bash
pip install sphinx sphinx_rtd_theme
```

- Move to the docs directory and run make. It takes couple of minutes to generate the build files.

```bash
cd docs
make html
explorer build\html\index.html
```

## Troubleshooting

- If you face any vcvarsall.bat errors, try installing visual studio 2017 community edition.

- If you are unable to import `active_contour` module, please follow the steps in the environment setup to generate the module.

## License

This code is GNU GENERAL PUBLIC LICENSED.

## Contributing

If you have any suggestions or identified bugs please feel free to post
them!

  [OpenCV 3.1.0]: http://opencv.org/downloads.html
  [Python]: https://www.python.org/downloads/
  [numpy]: https://www.scipy.org/scipylib/download.html
  [matplotlib]: https://matplotlib.org/
  [Anaconda]: https://www.anaconda.com/download/
  [Miniconda]: https://conda.io/miniconda.html
  [binaries]: http://boost.teeks99.com
  [OpenCV 2.4.10]: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.10/opencv-2.4.10.exe/download