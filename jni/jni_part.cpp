#include <jni.h>

#include <vector>
#include "stdio.h"
#include <cassert>

#include "linked_list.hpp"
#include "ac_withoutedges_yuv.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

extern "C" {
JNIEXPORT void JNICALL Java_com_moledetection_android_ProcessImage_ActiveContour(JNIEnv*, jobject,jlong addrHsv, jlong addrGray, jlong addrRgba);

JNIEXPORT void JNICALL Java_com_moledetection_android_ProcessImage_ActiveContour(JNIEnv*, jobject, jlong addrHsv,jlong addrGray, jlong addrRgba)
{
    Mat& mHsv  = *(Mat*)addrHsv;
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;

    vector<KeyPoint> v;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int largest_area =0,largest_contour_index=0;
    Rect bounding_rect;
    Scalar color = Scalar(255,255,255);

    const ofeli::List* Lout1;
        const ofeli::List* Lin1;

    	int img1_width = mHsv.cols;
    	int img1_height = mHsv.rows;
    	unsigned char* img1_rgb_data = (unsigned char*)mHsv.data;
    	unsigned char* img1_gray_data = (unsigned char*)mGr.data;
    	ofeli::ACwithoutEdgesYUV ac(img1_rgb_data, img1_width, img1_height, true, 0.95, 0.95, 0.0, 0.0, true, 5, 1.0, true, 50, 1, 1, 1, 1, 1, 1);
    		//ac.evolve_to_final_state();
    /*
    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mHsv, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }*/
    		int i,j; // offset of the current pixel
    		// initialize the active contour for the new frame
    		//ac.initialize_for_each_frame();

    		// evolve 100 times the active contour
    		ac.evolve_n_iterations(100);

    		Lout1 = &ac.get_Lout();
    		Lin1 = &ac.get_Lin();

    		ac.evolve_to_final_state();

    		// display
    		for( ofeli::List::const_Link iterator = Lout1->get_begin(); !iterator->end(); iterator = iterator->get_next() )
    		{
    		    j=iterator->get_elem();
    		    i = 3*j;
 				//img1_rgb_data[i] = 0; // H
    			//img1_rgb_data[i+1] = 255; // S
    			//img1_rgb_data[i+2] = 0; // V

    			img1_gray_data[j]=255;
    		}

    		for( ofeli::List::const_Link iterator = Lin1->get_begin(); !iterator->end(); iterator = iterator->get_next() )
    		{
    		    i = 3*iterator->get_elem();

    			//img1_rgb_data[i] = 0; // H
    			//img1_rgb_data[i+1] = 0; // S
    			//img1_rgb_data[i+2] = 255; // V

 				img1_gray_data[j]=255;
  			}
}
}
