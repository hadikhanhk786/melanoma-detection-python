#include <boost/python.hpp>
#include <vector>
#include <cassert>
#include <iostream>
// #include <fstream>

#include "linked_list.hpp"
#include "ac_withoutedges_yuv.hpp"
#include "conversion.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace py = boost::python;

typedef unsigned char uchar_t;



static void init()
{
    Py_Initialize();
    import_array();
}



void get_active_contour(Mat img, Mat mGr, int iterations){

	// Mat img = imread("/Images/IMD002_cropped.jpg",1);
	// Mat mGr = Mat::zeros(img.size(), CV_8UC1);
	// ofstream myfile;
	// myfile.open("/Images/res.txt");
	// cout << "Rows: " << img.rows << " Cols: " << img.cols;
//	vector<KeyPoint> v;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
//	int largest_area =0,largest_contour_index=0;
	Rect bounding_rect;
//	Scalar color = Scalar(255,255,255);

	const ofeli::List* Lout1;
	const ofeli::List* Lin1;

	int img1_width = img.cols;
	int img1_height = img.rows;
	unsigned char* img1_rgb_data = (unsigned char*)img.data;
	unsigned char* img1_gray_data = (unsigned char*)mGr.data;
	ofeli::ACwithoutEdgesYUV ac(img1_rgb_data, img1_width, img1_height,
			true, 0.95, 0.95, 0.0, 0.0, true, 5, 1.0, true, 50, 1, 1, 1, 1, 1, 1);
	int i,j;
	// myfile << "CoutRGB \n";
	// myfile << ac.get_CoutR() << ac.get_CoutG() << ac.get_CoutB();

	// myfile << "CoutYUV \n";
	// myfile << ac.get_CoutY() << ac.get_CoutU() << ac.get_CoutV();

	// myfile << "CinRGB \n";
	// myfile << ac.get_CinR() << ac.get_CinG() << ac.get_CinB();

	// myfile << "CoutYUV \n";
	// myfile << ac.get_CinY() << ac.get_CinU() << ac.get_CinV();

	// myfile << "n_out \n";
	// myfile << ac.get_CinY() << ac.get_CinU() << ac.get_CinV();
	if(iterations!=9999){
		ac.evolve_n_iterations(iterations);
	}
	else{
		ac.evolve_to_final_state();
	}
	
	// myfile << "After iterations \n";
	// myfile << "CoutRGB \n";
	// myfile << ac.get_CoutR() << ac.get_CoutG() << ac.get_CoutB();

	// myfile << "CoutYUV \n";
	// myfile << ac.get_CoutY() << ac.get_CoutU() << ac.get_CoutV();

	// myfile << "CinRGB \n";
	// myfile << ac.get_CinR() << ac.get_CinG() << ac.get_CinB();

	// myfile << "CoutYUV \n";
	// myfile << ac.get_CinY() << ac.get_CinU() << ac.get_CinV();
	Lout1 = &ac.get_Lout();
	Lin1 = &ac.get_Lin();

//	ac.evolve_to_final_state();
	// myfile << "Lout \n";
	// display
	for( ofeli::List::const_Link iterator = Lout1->get_begin(); !iterator->end(); iterator = iterator->get_next() )
	{
		j=iterator->get_elem();
		i = 3*j;
		//img1_rgb_data[i] = 0; // H
		//img1_rgb_data[i+1] = 255; // S
		//img1_rgb_data[i+2] = 0; // V

		img1_gray_data[j]=255;
		// myfile << "[" << i << "," << j << "],";
	}
	// myfile << "Lout \n";
	for( ofeli::List::const_Link iterator = Lin1->get_begin(); !iterator->end(); iterator = iterator->get_next() )
	{
		i = 3*iterator->get_elem();

		//img1_rgb_data[i] = 0; // H
		//img1_rgb_data[i+1] = 0; // S
		//img1_rgb_data[i+2] = 255; // V

		img1_gray_data[j]=255;
		// myfile << "[" << i << "," << j << "],";
	}

	// imwrite("/Images/result.png",mGr);

}


/**
 * Converts a grayscale image to a bilevel image.
 */
PyObject*
run(PyObject *hsvImg, int iterations)
{
    NDArrayConverter cvt;
    cv::Mat hsv_img { cvt.toMat(hsvImg) };
	cv::Mat zero_img = Mat::zeros(hsv_img.size(), CV_8UC1);
    get_active_contour(hsv_img, zero_img, iterations);
    return cvt.toNDArray(zero_img);
}


BOOST_PYTHON_MODULE(active_contour)
{
    init();
    py::def("run", run);
}

