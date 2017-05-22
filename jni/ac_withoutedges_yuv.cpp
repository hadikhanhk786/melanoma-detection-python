/****************************************************************************
**
** Copyright (C) 2010-2011 Fabien Bessy.
** All rights reserved.
** Contact: fabien.bessy@gmail.com
**
** This file is part of Ofeli.
**
** You may use this file under the terms of the CeCILL license version 2 as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Fabien Bessy nor the names of its contributors
**     may be used to endorse or promote products derived from this
**     software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
**                    For all legal intents and purposes.
**
****************************************************************************/

#include "ac_withoutedges_yuv.hpp"
#include <iostream> // for the object "std::cerr"

namespace ofeli
{

ACwithoutEdgesYUV::ACwithoutEdgesYUV(const unsigned char* img_rgb_data1, int img_width1, int img_height1) : ActiveContour(img_width1, img_height1, false, 0.9, 0.9, 0.0, 0.0, true, 5, 2.0, true, 30, 3), img_rgb_data(img_rgb_data1), lambda_out(1), lambda_in(1), alpha(1), beta(10), gamma(10)
{
    if( img_rgb_data1 == NULL )
    {
        std::cerr << std::endl << "The pointer img_rgb_data1 must be a non-null pointer, it must be allocated." << std::endl;
    }
    else
    {
        initialize_sums();
        calculate_means();
    }
}

ACwithoutEdgesYUV::ACwithoutEdgesYUV(const unsigned char* img_rgb_data1, int img_width1, int img_height1, bool hasEllipse1, double init_width1, double init_height1, double center_x1, double center_y1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1, int lambda_out1, int lambda_in1, int alpha1, int beta1, int gamma1) : ActiveContour(img_width1, img_height1, hasEllipse1, init_width1, init_height1, center_x1, center_y1, hasSmoothingCycle1, kernel_length1, sigma1, hasCleaning1, Na1, Ns1), img_rgb_data(img_rgb_data1), lambda_out(lambda_out1), lambda_in(lambda_in1), alpha(alpha1), beta(beta1), gamma(gamma1)
{
    if( img_rgb_data1 == NULL )
    {
        std::cerr << std::endl << "The pointer img_rgb_data1 must be a non-null pointer, it must be allocated." << std::endl;
    }
    else
    {
        initialize_sums();
        calculate_means();
    }
}

ACwithoutEdgesYUV::ACwithoutEdgesYUV(const unsigned char* img_rgb_data1, int img_width1, int img_height1, const char* phi_init1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1, int lambda_out1, int lambda_in1, int alpha1, int beta1, int gamma1) : ActiveContour(img_width1, img_height1, phi_init1, hasSmoothingCycle1, kernel_length1, sigma1, hasCleaning1, Na1, Ns1), img_rgb_data(img_rgb_data1), lambda_out(lambda_out1), lambda_in(lambda_in1), alpha(alpha1), beta(beta1), gamma(gamma1)
{
    if( img_rgb_data1 == NULL )
    {
        std::cerr << std::endl << "The pointer img_rgb_data1 must be a non-null pointer, it must be allocated." << std::endl;
    }
    else
    {
        initialize_sums();
        calculate_means();
    }
}

void ACwithoutEdgesYUV::initialize_sums()
{
    sum_in_R = 0;
    sum_out_R = 0;

    sum_in_G = 0;
    sum_out_G = 0;

    sum_in_B = 0;
    sum_out_B = 0;

    n_in = 0;
    n_out = 0;

    for( int i = 0; i < img_size; i++ )
    {
        // RGB planar image data buffer, with R1 R2 R3 ... G1 G2 G3 ... B1 B2 B3 ... (for Matlab)
        /*
        R = int(img_rgb_data[ i ]);
        G = int(img_rgb_data[ i+img_size ]);
        B = int(img_rgb_data[ i+2*img_size ]);
        */

        // BGR interleaved image data buffer, with B1 G1 R1 B2 G2 R2 ...
        B = int(img_rgb_data[ 3*i ]);
        G = int(img_rgb_data[ 3*i+1 ]);
        R = int(img_rgb_data[ 3*i+2 ]);

        if( phi[i] > 0 )
        {
            sum_out_R += R;
            sum_out_G += G;
            sum_out_B += B;

            n_out++;
        }
        else
        {
            sum_in_R += R;
            sum_in_G += G;
            sum_in_B += B;

            n_in++;
        }
    }

    return;
}

inline void ACwithoutEdgesYUV::calculate_means()
{
    // protection against /0
    if( n_out == 0 )
    {
        CoutR = 0;
        CoutG = 0;
        CoutB = 0;
    }
    else
    {
        CoutR = sum_out_R/n_out;
        CoutG = sum_out_G/n_out;
        CoutB = sum_out_B/n_out;
    }

    // protection against /0
    if( n_in == 0 )
    {
        CinR = 255;
        CinG = 255;
        CinB = 255;
    }
    else
    {
        CinR = sum_in_R/n_in;
        CinG = sum_in_G/n_in;
        CinB = sum_in_B/n_in;
    }

    CoutY = calculate_Y(CoutR, CoutG, CoutB);
    CoutU = calculate_U(CoutR, CoutG, CoutB);
    CoutV = calculate_V(CoutR, CoutG, CoutB);

    CinY = calculate_Y(CinR, CinG, CinB);
    CinU = calculate_U(CinR, CinG, CinB);
    CinV = calculate_V(CinR, CinG, CinB);

    return;
}

inline int ACwithoutEdgesYUV::compute_external_speed_Fd(int i, int x, int y)
{
    // RGB planar image data buffer, with R1 R2 R3 ... G1 G2 G3 ... B1 B2 B3 ... (for Matlab)
    /*
    R = int(img_rgb_data[ i ]);
    G = int(img_rgb_data[ i+img_size ]);
    B = int(img_rgb_data[ i+2*img_size ]);
    */

    // BGR interleaved image data buffer, with B1 G1 R1 B2 G2 R2 ...
	B = int(img_rgb_data[ 3*i ]);
	G = int(img_rgb_data[ 3*i+1 ]);
	R = int(img_rgb_data[ 3*i+2 ]);

    int Y = calculate_Y(R, G, B);
    int U = calculate_U(R, G, B);
    int V = calculate_V(R, G, B);

    return lambda_out*( alpha*square(Y-CoutY)+beta*square(U-CoutU)+gamma*square(V-CoutV) ) - lambda_in*( alpha*square(Y-CinY)+beta*square(U-CinU)+gamma*square(V-CinV) );
}

inline void ACwithoutEdgesYUV::do_specific_in_step(int i, int x, int y)
{
    sum_out_R -= R;
    sum_out_G -= G;
    sum_out_B -= B;
    n_out--;

    sum_in_R += R;
    sum_in_G += G;
    sum_in_B += B;
    n_in++;

    return;
}

inline void ACwithoutEdgesYUV::do_specific_out_step(int i, int x, int y)
{
    sum_in_R -= R;
    sum_in_G -= G;
    sum_in_B -= B;
    n_in--;

    sum_out_R += R;
    sum_out_G += G;
    sum_out_B += B;
    n_out++;

    return;
}

inline void ACwithoutEdgesYUV::do_specific_in_step2(int i, int x, int y)
{
    // RGB planar image data buffer, with R1 R2 R3 ... G1 G2 G3 ... B1 B2 B3 ... (for Matlab)
    /*
    R = int(img_rgb_data[ i ]);
    G = int(img_rgb_data[ i+img_size ]);
    B = int(img_rgb_data[ i+2*img_size ]);
    */

    // BGR interleaved image data buffer, with B1 G1 R1 B2 G2 R2 ...
	B = int(img_rgb_data[ 3*i ]);
	G = int(img_rgb_data[ 3*i+1 ]);
	R = int(img_rgb_data[ 3*i+2 ]);

    sum_out_R -= R;
    sum_out_G -= G;
    sum_out_B -= B;
    n_out--;

    sum_in_R += R;
    sum_in_G += G;
    sum_in_B += B;
    n_in++;

    return;
}

inline void ACwithoutEdgesYUV::do_specific_out_step2(int i, int x, int y)
{
    // RGB planar image data buffer, with R1 R2 R3 ... G1 G2 G3 ... B1 B2 B3 ... (for Matlab)
    /*
    R = int(img_rgb_data[ i ]);
    G = int(img_rgb_data[ i+img_size ]);
    B = int(img_rgb_data[ i+2*img_size ]);
    */

    // BGR interleaved image data buffer, with B1 G1 R1 B2 G2 R2 ...
	B = int(img_rgb_data[ 3*i ]);
	G = int(img_rgb_data[ 3*i+1 ]);
	R = int(img_rgb_data[ 3*i+2 ]);

    sum_in_R -= R;
    sum_in_G -= G;
    sum_in_B -= B;
    n_in--;

    sum_out_R += R;
    sum_out_G += G;
    sum_out_B += B;
    n_out++;

    return;
}

void ACwithoutEdgesYUV::do_specific_beginning_step()
{
    calculate_means();

    return;
}

inline int ACwithoutEdgesYUV::calculate_Y(int R, int G, int B)
{
    return ( (  66 * R + 129 * G +  25 * B + 128) >> 8) +  16;
}

inline int ACwithoutEdgesYUV::calculate_U(int R, int G, int B)
{
    return ( ( -38 * R -  74 * G + 112 * B + 128) >> 8) + 128;
}

inline int ACwithoutEdgesYUV::calculate_V(int R, int G, int B)
{
    return ( ( 112 * R -  94 * G -  18 * B + 128) >> 8) + 128;
}



int ACwithoutEdgesYUV::get_CoutR() const
{
    return CoutR;
}
int ACwithoutEdgesYUV::get_CoutG() const
{
    return CoutG;
}
int ACwithoutEdgesYUV::get_CoutB() const
{
    return CoutB;
}

int ACwithoutEdgesYUV::get_CinR() const
{
    return CinR;
}
int ACwithoutEdgesYUV::get_CinG() const
{
    return CinG;
}
int ACwithoutEdgesYUV::get_CinB() const
{
    return CinB;
}


int ACwithoutEdgesYUV::get_CoutY() const
{
    return CoutY;
}
int ACwithoutEdgesYUV::get_CoutU() const
{
    return CoutU;
}
int ACwithoutEdgesYUV::get_CoutV() const
{
    return CoutV;
}

int ACwithoutEdgesYUV::get_CinY() const
{
    return CinY;
}
int ACwithoutEdgesYUV::get_CinU() const
{
    return CinU;
}
int ACwithoutEdgesYUV::get_CinV() const
{
    return CinV;
}

// initialization for video tracking
void ACwithoutEdgesYUV::initialize_for_each_frame()
{
    do_general_initialization_for_each_frame();
    initialize_sums();
    calculate_means();

    return;
}

// initialization for video tracking
void ACwithoutEdgesYUV::initialize_for_each_frame(const unsigned char* img_rgb_data1)
{
    img_rgb_data = img_rgb_data1;
    this->initialize_for_each_frame();

    return;
}

}
