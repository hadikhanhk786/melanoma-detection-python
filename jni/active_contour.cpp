/****************************************************************************
** active_contour.cpp defines the entry point of the application
**
** Author : Upender Kalwa
** Version : 2.1
** License : BSD
** Last modified : 08/28/2017
** Contact : ukalwa@iastate.edu
**
****************************************************************************/

// in-built headers
#include <vector>
#include <iostream>

// third party headers
#include <boost/python.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Custom headers
#include "linked_list.hpp"
#include "ac_withoutedges_yuv.hpp"
#include "conversion.h"

using namespace cv;

namespace py = boost::python;

typedef unsigned char uchar_t;
py::list list;
int iter[] = {50, 1};
int energy_params[] = { 1, 1, 1, 1, 1};
double gaussian_params[] = {5,1.0};
double init_width = 0.65;
double init_height = 0.65;

int max_area_pos;
int rows,cols;
size_t cnt;

std::vector<std::vector<Point> > contours;
std::vector<Vec4i> hierarchy;
std::vector<Point> break_points;

// Initialize boost python modules
static void init()
{
    Py_Initialize();
    import_array();
}


void get_active_contour(Mat img, Mat mGr, int iterations, int init_contour) {
    /********************************************************************************************
    ** Run the active contour function with the given parameters and evolve the contur until the
    ** number of iterations passed
    *********************************************************************************************/
    const ofeli::List* Lout;

    // store cols and rows of the image
    cols = img.cols;
    rows = img.rows;

    // Convert cols x rows x dim image array to a linear array
    unsigned char* img_rgb_data = img.data;
    unsigned char* img_gray_data = mGr.data;

    bool shape = init_contour != 1; // 1 - Rectangle , 0 - ellipse (default)

    // call the active contour function with the necessary parameters
    ofeli::ACwithoutEdgesYUV ac(img_rgb_data, cols, rows,
                                shape, init_width, init_height, 0.0, 0.0, true,
                                (int)gaussian_params[0], gaussian_params[1], true,
                                iter[0], iter[1], energy_params[0], energy_params[1],
                                energy_params[2], energy_params[3], energy_params[4]);

    int j; //loop variable
    if (iterations != 9999) {
        // Evolve only until given number of iterations
        ac.evolve_n_iterations(iterations);
    }
    else {
        // Evolve until energy is minimized
        ac.evolve_to_final_state();
    }
    Lout = &ac.get_Lout();
    std::cout << "Lout size:" << Lout->size() << std::endl;

    // Draw contour formed by points in Lout
    for (ofeli::List::const_Link iterator = Lout->get_begin(); !iterator->end();
         iterator = iterator->get_next())
    {
        j = iterator->get_elem();
        img_gray_data[j] = 255; // Mark the pixel as white
    }

}

void create_mask(Mat contour_image, Mat zero_img) {
    /********************************************************************************************
    ** Find the contours in the image and determine the largest contour in them. And draw a filled
    ** contour of the maximum contour on a newly created zero image to form a mask
    *********************************************************************************************/
    // Create a zero image with same dimensions as original image to draw mask on it
    double max_area = 0; // max_area to determine the largest contour in the image
    double area; // Loop variable
    // Create a temporary clone of the contour image as the function findContours is known
    // to modify the pixels in the image passed as parameter
    Mat temp_img = contour_image.clone();
    findContours(temp_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Loop through the contours to find the largest contour in the image by measuring
    // area of each contour and store the array position in max_area_pos
    cnt = contours.size();
    if (cnt > 0) {
        for (int j = 0; j < cnt; j++) {
            area = contourArea(contours[j]);
            if (area > max_area) {
                max_area = area;
                max_area_pos = j;
            }
        }
    }

    // draw a filled contour of the maximum contour to form mask on the zero image
    drawContours(zero_img, contours, max_area_pos, 255, -1);
}

void find_contour_breaks() {
    /********************************************************************************************
    ** When there are breaks in contours, contours returned by findContours function make a u-turn
    ** at the break point. To identify the break point, we use cosine rule to determine the break
    ** points
    *********************************************************************************************/
    int spacing = 1; // Spacing between the three points to find curvature (U-turn)
    Point pt1, pt2, pt3, v1, v2;
    double dotprod, norm1, norm2, cosine; // Parameters required to measure cosine of three points
    if (cnt > 0) {
        std::cout << "break points : ";
        // Loop through all the break points in the image (only occurs at the edges of the image)
        // We assume breaks happen in pairs (connecting them forms a complete contour)
        for (int j = 0; j < cnt; j++) {
            if (contours[j].size() <= 1) // Might be a mistake
                continue;
            for (int k = 0; k < contours[j].size(); k++) {
                // Load three consecutive points separated by spacing
                pt2 = contours[j][k];
                pt1 = k - spacing < 0 ? contours[j][contours[j].size() + k - spacing] :
                      contours[j][k - spacing];
                pt3 = k + spacing >= contours[j].size() ? contours[j][k + spacing -
                        contours[j].size()] : contours[j][k + spacing];
                // We determine u-turn based on cosine rule
                // cos(theta) = ((pt1-pt2).(pt3-pt2)) / (|pt1-pt2||pt3-pt2|)
                // Calculate differences from center point pt2
                v1 = pt1 - pt2;
                v2 = pt3 - pt2;
                dotprod = v1.ddot(v2); // Calculate dot product of the two differences
                // Calculate magnitudes of the points
                norm1 = sqrt(pow(v1.x, 2) + pow(v1.y, 2));
                norm2 = sqrt(pow(v2.x, 2) + pow(v2.y, 2));
                if (norm1 == 0 || norm2 == 0) { // To avoid divide-by-zero condition
                    cosine = 0;
                }
                else {
                    cosine = (dotprod) / (norm1*norm2); // cos(theta) formula
                }
                if (cosine > 0.91) { // If greater than 0.90, there is a u-turn
                    // The cosine outputs high values when there is a sharp curve (in actual contour
                    // and at break points). We are interested in the break points that occur at
                    // the edges of the image (i.e., 0-10 pixels away from the edge)
                    if ((pt2.x >= 0 && pt2.x <= 10) || (cols - pt2.x >= 0 && cols - pt2.x <= 10) ||
                        (pt2.y >= 0 && pt2.y <= 10) || (rows - pt2.y >= 0 && rows - pt2.y <= 10)) {
                        break_points.push_back(pt2); // add the point to the vector
                        std::cout << pt2 << ",";
                    }
                }
            }
        }
    }
    std::cout << break_points.size() << std::endl;
}

bool detect_break_diff_axes(Point &pt1, Point &pt2, Point &pt3, Point &pt4) {
    /********************************************************************************************
    ** This functions handles breaks occuring in two different axes. i.e., each point is on either
    ** on x or y axis. This situtation happens when the object of interest is either too big to fit
    ** in the image or imaged in such a way that it touches the edges of the image
    *********************************************************************************************/
    bool axes_break;
    int diff_x = pt1.x - pt2.x;
    int diff_y = pt1.y - pt2.y;
    int diff = diff_x < diff_y ? diff_x : diff_y;
	// std::cout << "diffs x" << diff_x << "diffs y" << diff_y << std::endl;
	// If the break points are on the same axes but on two ends
	if (abs(diff_y) >= rows - 10 ){

		Point pt = Point(0,0);
		if (pt1.x > pt2.x){
			pt = pt1;
			pt1 = pt2;
			pt2 = pt;
		}
		if (pt1.x > abs(cols - pt2.x)){
			pt3.x = 1;
			pt4.x = 1;
		}
		else{
			pt3.x = cols - 1;
			pt4.x = cols - 1;
		}
		pt3.y = pt2.y;
		pt4.y = pt1.y;
		axes_break = true;
		std::cout << "Entered y" << pt1 << pt2 << pt3 << pt4 << std::endl;
		return axes_break;
	}

	if (abs(diff_x) >= cols - 10){
		Point pt = Point(0,0);
		if (pt1.y > pt2.y){
			pt = pt1;
			pt1 = pt2;
			pt2 = pt;
		}
		if (pt1.y > abs(rows - pt2.y)){
			pt3.y = 1;
			pt4.y = cols - 1;
		}
		else{
			pt3.y = cols - 1;
			pt4.y = 1;
		}
		pt3.x = pt2.x;
		pt4.x = pt1.x;
		axes_break = true;
		std::cout << "Entered x" << pt1 << pt2 << pt3 << pt4 << std::endl;
		return axes_break;
	}
    // If the break points are on the same axes,  return false
    if (diff >= 0 && diff <= 10) {
        axes_break = false;
    }
    else {
        //  Add another point to close the contour with the nearest edge of the image
        if ((pt1.x >= 0 && pt1.x <= 10) || (cols - pt1.x >= 0 && cols - pt1.x <= 10)) {
            pt3.x = pt1.x;
        }
        else {
            pt3.x = pt2.x;
        }
        if ((pt1.y >= 0 && pt1.y <= 10) || (rows - pt1.y >= 0 && rows - pt1.y <= 10)) {
            pt3.y = pt1.y;
        }
        else {
            pt3.y = pt2.y;
        }
        axes_break = true;
    }
    return axes_break;
}

void stitch_contour_breaks(Mat contour_image) {
    /********************************************************************************************
    ** This functions fixes the breaks in the contour by determining the pairs and joining them
    ** together by drawing a line between the pairs. If the pairs are on different axes, it draws
    ** two lines connecting the third point to paired points
    *********************************************************************************************/
    Point pt, pt1, pt2; // Loop variables
    Point pt3 = Point(0,0); // Initialize pt3 incase breaks happen on different axes
	Point pt4 = Point(0,0); // Initialize pt3 incase breaks happen on different axes
    int temp_idx;
    // Paired point indexes are stored to avoid each point from forming additional pairs
    std::vector<int> idx_array;
    bool flag_x;
    int min_dist_pos=0;
    double min_dist, dist;
    // handling if there is only break in the image i.e., two break points
    if (break_points.size() == 2) {
        pt1 = break_points[0];
        pt2 = break_points[1];
        // Detect pairs on different axes
        if (detect_break_diff_axes(pt1, pt2, pt3, pt4)) {
            line(contour_image, pt1, pt3, Scalar(255), 1);
            line(contour_image, pt3, pt2, Scalar(255), 1);
        }
        else {
            // if both the points are on same axes, draw a line between them
            line(contour_image, pt1, pt2, Scalar(255), 1);
        }
    }
    else { // handle multiple break points in the image
        for (int i = 0; i < break_points.size(); i++) {
            flag_x = false;
            pt = break_points[i];
            min_dist = 9999;
            // Skip the point if it is already paired with another point
            for (int k = 0; k < idx_array.size(); k++) {
                //std::cout << idx_array[k] << ",";
                if (i == idx_array[k]) {
                    flag_x = true; // set flag to indicate it is already paired
                    break;
                }
            }
            //std::cout << std::endl;
            if (flag_x) // skip the point if it is paired
                continue;
            for (int j = i + 1; j < break_points.size(); j++) {
                bool flag_y = false;
                // Skip the point if it is already paired with another point
                for (int k = 0; k < idx_array.size(); k++) {
                    if (j == idx_array[k]) {
                        flag_y = true; // set flag to indicate it is already paired
                        break;
                    }
                }
                if (flag_y) // skip the point if it is paired
                    continue;
                // calculate distances from the current point to all other points
                dist = sqrt(pow(pt.x - break_points[j].x, 2) + pow(pt.y - break_points[j].y, 2));
                // calculate the minimum distance from all other points
                // and store its index in min_dist_pos
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_pos = j;
                }
            }
            temp_idx = min_dist_pos;
            idx_array.push_back(i);
            idx_array.push_back(temp_idx);
            std::cout << "pairs formed : " << pt << break_points[temp_idx] << std::endl;
            // Detect pairs on different axes
            if (detect_break_diff_axes(pt, break_points[temp_idx], pt3, pt4)) {
				if (pt4.x != 0 && pt4.y != 0){
					line(contour_image, pt, pt4, Scalar(255), 1);
					line(contour_image, pt3, break_points[temp_idx], Scalar(255), 1);
					line(contour_image, pt3, pt4, Scalar(255), 1);
					pt4 = Point(0,0);
				}else{
					line(contour_image, pt, pt3, Scalar(255), 1);
					line(contour_image, pt3, break_points[temp_idx], Scalar(255), 1);
				}
            }
            else { // if both the points are on same axes, draw a line between them
                line(contour_image, pt, break_points[temp_idx], Scalar(255), 1);
            }

        }
    }

}


/**
 * Converts a grayscale image to a bilevel image.
 */
PyObject*
run(PyObject *img, int iterations, int init_contour, double init_width_python,
	double init_height_python,  py::list& iter_list, py::list& energy_list,
	py::list& gaussian_list)
{
	// Emtpy contents in the arrays
	break_points = {};
	// Numpy ndarray converter to convert numpy arrays to cv::Mat arrays
    NDArrayConverter cvt;

	//convert numpy array image to cv::Mat image
    Mat image { cvt.toMat(img) };

	// Create zero images to draw contour and mask
	Mat mGr = Mat::zeros(image.size(), CV_8UC1);
	Mat mask_image = Mat(image.size(), CV_8UC1);
	int non_zero_before = 0, non_zero_after = 0; // Used to find breaks in contour
	// store cols and rows of the image
	cols = image.cols;
	rows = image.rows;

	// retrieve init params sent from python
	init_height = init_height_python;
	init_width = init_width_python;

	// retrieve lists sent from python and load it into c++ arrays
	for (int i=0;i<len(gaussian_list);i++){
		gaussian_params[i] = py::extract<double>(gaussian_list[i]);
	}
	std::cout << "Gaussian passed : " << gaussian_params[0] << "," << gaussian_params[1] << std::endl;

	for (int i=0;i<len(iter_list);i++){
		iter[i] = py::extract<int>(iter_list[i]);
	}
	std::cout << "Iterations passed : " << iter[0] << "," << iter[1] << std::endl;

	std::cout << "Energy params passed : ";
	for (int i=0;i<len(energy_list);i++){
		energy_params[i] = py::extract<int>(energy_list[i]);
		std::cout << energy_params[i] << ",";
	}
	std::cout << std::endl;

	// call active contour function with the params
    get_active_contour(image, mGr, iterations, init_contour);
    non_zero_before = countNonZero(mGr); // non zero pixel count in contour image
    std::cout << "Non Zero Pixels count before : " << non_zero_before << std::endl;

    // find largest contour and draw mask
    mask_image = Scalar(0);
    create_mask(mGr, mask_image);
    non_zero_after = countNonZero(mask_image); // non zero pixel count in mask
    // Print non zero pixels to debug breaks in the contour
    std::cout << "Non Zero Pixels count after : " << non_zero_after << std::endl;

    // We assume a filled mask has atleast 10 times more number of non zero pixels
    if (non_zero_after > 10 * non_zero_before){
        std::cout << "No breaks in image" << std::endl;
    }
    else {
        // If there are breaks in contour, drawContours doesn't fill the contour
        // hence the non zero pixel count would be around the same
        std::cout << "breaks found in image" << std::endl;
        find_contour_breaks(); // find end points of open contours
        stitch_contour_breaks(mGr); // join the break points to form closed contour

        // find largest contour and draw mask
        mask_image = Scalar(0);
        create_mask(mGr, mask_image);
        non_zero_after = countNonZero(mask_image);
        std::cout << "Non Zero Pixels count after : " << non_zero_after << std::endl;
    }
    return cvt.toNDArray(mask_image);
}


BOOST_PYTHON_MODULE(active_contour)
{
    init();
    py::def("run", run);
}

