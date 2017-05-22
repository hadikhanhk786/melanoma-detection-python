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

#ifndef AC_WITHOUTEDGES_YUV_HPP
#define AC_WITHOUTEDGES_YUV_HPP

#include "activecontour.hpp"

namespace ofeli
{

class ACwithoutEdgesYUV: public ActiveContour
{

public :

    //! Constructor to initialize the active contour with a centered rectangle and the default values of the algorithm parameters.
    ACwithoutEdgesYUV(const unsigned char* img_data1, int img_width1, int img_height1);

    //! Constructor to initialize the active contour from geometrical parameters of an unique shape, an ellipse or a rectangle.
    ACwithoutEdgesYUV(const unsigned char* img_rgb_data1, int img_width1, int img_height1, bool hasEllipse1, double init_width1, double init_height1, double center_x1, double center_y1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1, int lambda_out1, int lambda_in1, int alpha1, int beta1, int gamma1);

    //! Constructor to initialize the active contour from an initial phi level-set function.
    ACwithoutEdgesYUV(const unsigned char* img_rgb_data1, int img_width1, int img_height1, const char* phi_init1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1, int lambda_out1, int lambda_in1, int alpha1, int beta1, int gamma1);

    //! Getter function for #CoutR
    int get_CoutR() const;
    //! Getter function for #CoutG
    int get_CoutG() const;
    //! Getter function for #CoutB
    int get_CoutB() const;

    //! Getter function for #CinR
    int get_CinR() const;
    //! Getter function for #CinG
    int get_CinG() const;
    //! Getter function for #CinB
    int get_CinB() const;

    //! Getter function for #CoutY
    int get_CoutY() const;
    //! Getter function for #CoutU
    int get_CoutU() const;
    //! Getter function for #CoutV
    int get_CoutV() const;

    //! Getter function for #CinY
    int get_CinY() const;
    //! Getter function for #CinU
    int get_CinU() const;
    //! Getter function for #CinV
    int get_CinV() const;

    //! Initialization for each new frame buffer, used for video tracking.
    void initialize_for_each_frame();

    //! Initialization for each new frame buffer, used for video tracking.
    void initialize_for_each_frame(const unsigned char* img_rgb_data1);

private :

    //! Initializes the six sums and #n_in and #n_out with scanning through the image.
    void initialize_sums();

    //! Calculates means (#CoutY, #CoutU, #CoutV) and (#CinY, #CinU, #CinV) in \a O(1) or accounting for the previous updates of (#sum_out_R, #sum_out_G, #sum_out_B) and (#sum_in_R, #sum_in_G, #sum_in_B), in \a O(#lists_length) and not in \a O(#img_size).
    inline void calculate_means();

    //! Computes external speed \a Fd with the Chan-Vese model for a current point \a (x,y) of #Lout or #Lin.
    inline int compute_external_speed_Fd(int i, int x, int y);


    //! Calculates means (#CoutY, #CoutU, #CoutV) and (#CinY, #CinU, #CinV) in \a O(1) or accounting for the previous updates of (#sum_out_R, #sum_out_G, #sum_out_B) and (#sum_in_R, #sum_in_G, #sum_in_B), in \a O(#lists_length) and not in \a O(#img_size).
    inline void do_specific_beginning_step();

    //! Updates the six sums, #n_in and #n_out, before each #switch_in, in the cycle 1, in order to calculate means (#CoutY, #CoutU, #CoutV) and (#CinY, #CinU, #CinV).
    inline void do_specific_in_step(int i, int x, int y);

    //! Updates the six sums, #n_in and #n_out, before each #switch_out, in the cycle 1, in order to calculate means (#CoutY, #CoutU, #CoutV) and (#CinY, #CinU, #CinV).
    inline void do_specific_out_step(int i, int x, int y);

    //! Updates the six sums, #n_in and #n_out, before each #switch_in, in the cycle 2, in order to calculate means (#CoutY, #CoutU, #CoutV) and (#CinY, #CinU, #CinV).
    inline void do_specific_in_step2(int i, int x, int y);

    //! Updates the six sums, #n_in and #n_out, before each #switch_out, in the cycle 2, in order to calculate means (#CoutY, #CoutU, #CoutV) and (#CinY, #CinU, #CinV).
    inline void do_specific_out_step2(int i, int x, int y);

    //! Calculates component Y with a (#R,#G,#B) value.
    static inline int calculate_Y(int R, int G, int B);

    //! Calculates component U with a (#R,#G,#B) value.
    static inline int calculate_U(int R, int G, int B);

    //! Calculates component V with a (#R,#G,#B) value.
    static inline int calculate_V(int R, int G, int B);





    //! Input pointer on the RGB image data buffer. This buffer must be row-wise and interleaved (R1 G1 B1 R2 G2 B2 ...).
    const unsigned char* img_rgb_data;

    //! Red component of the current pixel.
    int R;
    //! Green component of the current pixel.
    int G;
    //! Blue component of the current pixel.
    int B;

    //! Mean of component \a Y of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int CoutY;
    //! Mean of component \a U of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int CoutU;
    //! Mean of component \a V of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int CoutV;

    //! Mean of component \a Y of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int CinY;
    //! Mean of component \a U of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int CinU;
    //! Mean of component \a V of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int CinV;

    //! Mean of component \a R of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int CoutR;
    //! Mean of component \a G of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int CoutG;
    //! Mean of component \a B of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int CoutB;

    //! Mean of component \a R of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int CinR;
    //! Mean of component \a G of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int CinG;
    //! Mean of component \a B of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int CinB;

    //! Weight of component \a Y to calculate external speed \a Fd.
    const int alpha;
    //! Weight of component \a U to calculate external speed \a Fd.
    const int beta;
    //! Weight of component \a V to calculate external speed \a Fd.
    const int gamma;

    //! Weight of the outside homogeneity criterion in the Chan-Vese model.
    const int lambda_out;
    //! Weight of the inside homogeneity criterion in the Chan-Vese model.
    const int lambda_in;

    //! Sum of component #R of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int sum_out_R;
    //! Sum of component #R of the pixels intside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int sum_in_R;

    //! Sum of component #G of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int sum_out_G;
    //! Sum of component #G of the pixels intside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int sum_in_G;

    //! Sum of component #B of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int sum_out_B;
    //! Sum of component #B of the pixels intside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int sum_in_B;

    //! Number of pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$ .
    int n_out;
    //! Number of pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$ .
    int n_in;
};

}

#endif // AC_WITHOUTEDGES_YUV_HPP


//! \class ofeli::ACwithoutEdgesYUV
//! The child class ACwithoutEdgesYUV implements a function to calculate specifically speed \a Fd based on the Chan-Vese model, a region-based energy functional.
//! The regularization of our active contour is performed by a gaussian smoothing of #phi so we are interested uniquely by the external or data dependant term of this energy functional.\n
//! \f$F_{d}=\lambda _{out}\left[ \alpha \left( Y_{out}-C_{outC}\right) ^{2}+ \beta \left( U_{out}-C_{outU}\right) ^{2}+ \gamma \left( V_{out}-C_{outV}\right) ^{2}\right] + \lambda _{in}\left[ \alpha \left( Y_{in}-C_{inY}\right) ^{2}+ \beta \left( U_{in}-C_{inU}\right) ^{2}+ \gamma \left( V_{in}-C_{inV}\right) ^{2}\right]\f$
//!  - \f$F_{d}\f$ : data dependant evolution speed calculated for each point of the active contour, only it sign is used by the algorithm. \n
//!  - \f$Y\f$ : luminance component Y of the (Y,U,V) color space of the current pixel of the active contour. \n
//!  - \f$U\f$ : chrominance component U of the (Y,U,V) color space of the current pixel of the active contour. \n
//!  - \f$V\f$ : chrominance component V of the (Y,U,V) color space of the current pixel of the active contour. \n
//!  - \f$C_{out}\f$ : mean of the intensities or grey-levels of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$. \n
//!  - \f$C_{in}\f$ : mean of the intensities or grey-levels of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$. \n
//!  - \f$C_{out}\f$ : mean of the intensities or grey-levels of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$. \n
//!  - \f$C_{in}\f$ : mean of the intensities or grey-levels of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$. \n
//!  - \f$C_{out}\f$ : mean of the intensities or grey-levels of the pixels outside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) >0\f$. \n
//!  - \f$C_{in}\f$ : mean of the intensities or grey-levels of the pixels inside the curve, i.e. pixels \f$i\f$ with \f$\phi \left( i\right) <0\f$. \n
//!  - \f$\lambda _{out}\f$ : weight of the outside homogeneity criterion in the Chan-Vese model. \n
//!  - \f$\lambda _{in}\f$ : weight of the inside homogeneity criterion in the Chan-Vese model. \n
//!  - \f$\alpha\f$ : weight of the luminance component Y. \n
//!  - \f$\beta\f$ : weight of the chrominance component U. \n
//!  - \f$\gamma\f$ : weight of the chrominance component V.

/**
 * \fn ACwithoutEdgesYUV::ACwithoutEdgesYUV(const unsigned char* img_rgb_data1, int img_width1, int img_height1, bool hasEllipse1, double init_width1, double init_height1, double center_x1, double center_y1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1, int lambda_out1, int lambda_in1, int alpha1, int beta1, int gamma1)
 * \param img_rgb_data1 Input pointer on the RGB image data buffer. This buffer must be row-wise and interleaved (R1 G1 B1 R2 G2 B2 ...). Passed to #img_rgb_data.
 * \param img_width1 Image width, i.e. number of columns. Passed to #img_width.
 * \param img_height1 Image height, i.e. number of rows. Passed to #img_height.
 * \param hasEllipse1 Boolean to choose the shape of the active contour initialization, \c true for an ellipse or \c false for a rectangle.
 * \param init_width1 Width of the shape divided by the image #img_width.
 * \param init_height1 Height of the shape divided by the image #img_height.
 * \param center_x1 X-axis position (or column index) of the center of the shape divided by the image #img_width subtracted by 0.5
 * \param center_y1 Y-axis position (or row index) of the center of the shape divided by the image #img_height subtracted by 0.5\n
          To have the center of the shape in the image : -0.5 < center_x1 < 0.5 and -0.5 < center_y1 < 0.5
 * \param hasSmoothingCycle1 Boolean to have or not the curve smoothing, evolutions in the cycle 2 with an internal speed \a Fint. Passed to #hasSmoothingCycle.
 * \param kernel_length1 Kernel length of the gaussian filter for the curve smoothing. Passed to #kernel_length.
 * \param sigma1 Standard deviation of the gaussian kernel for the curve smoothing. Passed to #sigma.
 * \param hasCleaning1 Boolean to eliminate or not redundant points in #Lin and #Lout lists. Passed to #hasCleaning.
 * \param Na1 Number of times the active contour evolves in the cycle 1, external or data dependant evolutions with \a Fd speed. Passed to #Na_max.
 * \param Ns1 Number of times the active contour evolves in the cycle 2, curve smoothing or internal evolutions with \a Fint speed. Passed to #Ns_max.
 * \param lambda_out1 Weight of the outside homogeneity criterion. Passed to #lambda_out.
 * \param lambda_in1 Weight of the inside homogeneity criterion. Passed to #lambda_in.
 * \param alpha1 Weight of luminance Y. Passed to #alpha.
 * \param beta1 Weight of chrominance U. Passed to #beta.
 * \param gamma1 Weight of chrominance V. Passed to #gamma.
 */

/**
 * \fn ACwithoutEdgesYUV::ACwithoutEdgesYUV(const unsigned char* img_rgb_data1, int img_width1, int img_height1, const char* phi_init1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1, int lambda_out1, int lambda_in1, int alpha1, int beta1, int gamma1)
 * \param img_rgb_data1 Input pointer on the RGB data image buffer. This buffer must be row-wise and interleaved (R1 G1 B1 R2 G2 B2 ...). Passed to #img_rgb_data.
 * \param img_width1 Image width, i.e. number of columns. Passed to #img_width.
 * \param img_height1 Image height, i.e. number of rows. Passed to #img_height.
 * \param phi_init1 Pointer on the initialized level-set function buffer. Copied to #phi.
 * \param hasSmoothingCycle1 Boolean to have or not the curve smoothing, evolutions in the cycle 2 with an internal speed \a Fint. Passed to #hasSmoothingCycle.
 * \param kernel_length1 Kernel length of the gaussian filter for the curve smoothing. Passed to #kernel_length.
 * \param sigma1 Standard deviation of the gaussian kernel for the curve smoothing. Passed to #sigma.
 * \param hasCleaning1 Boolean to eliminate or not redundant points in #Lin and #Lout lists. Passed to #hasCleaning.
 * \param Na1 Number of times the active contour evolves in the cycle 1, external or data dependant evolutions with \a Fd speed. Passed to #Na_max.
 * \param Ns1 Number of times the active contour evolves in the cycle 2, curve smoothing or internal evolutions with \a Fint speed. Passed to #Ns_max.
 * \param lambda_out1 Weight of the outside homogeneity criterion. Passed to #lambda_out.
 * \param lambda_in1 Weight of the inside homogeneity criterion. Passed to #lambda_in.
 * \param alpha1 Weight of luminance Y. Passed to #alpha.
 * \param beta1 Weight of chrominance U. Passed to #beta.
 * \param gamma1 Weight of chrominance V. Passed to #gamma.
 */

/**
 * \fn virtual inline void ACwithoutEdgesYUV::do_specific_in_step(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel in the image X-axis or column index
 * \param y position of the current pixel in the image Y-axis or row index
 */

/**
 * \fn virtual inline void ACwithoutEdgesYUV::do_specific_out_step(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel in the image X-axis or column index
 * \param y position of the current pixel in the image Y-axis or row index
 */

/**
 * \fn virtual inline void ACwithoutEdgesYUV::do_specific_out_step2(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel in the image X-axis or column index
 * \param y position of the current pixel in the image Y-axis or row index
 */

/**
 * \fn virtual inline void ACwithoutEdgesYUV::do_specific_out_step2(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel in the image X-axis or column index
 * \param y position of the current pixel in the image Y-axis or row index
 */

/**
 * \fn inline void ACwithoutEdgesYUV::compute_external_speed_Fd(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel of the input image in the image X-axis or column index
 * \param y position of the current pixel of the input image in the image Y-axis or row index
 */
