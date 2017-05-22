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

#ifndef ACTIVECONTOUR_HPP
#define ACTIVECONTOUR_HPP

#include "linked_list.hpp"

namespace ofeli
{

class ActiveContour
{

public :

    //! Constructor to initialize the active contour from geometrical parameters of an unique shape, an ellipse or a rectangle.
    ActiveContour(int img_width1, int img_height1, bool hasEllipse1, double init_width1, double init_height1, double center_x1, double center_y1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1);

    //! Constructor to initialize the active contour from an initial phi level-set function.
    ActiveContour(int img_width1, int img_height1, const char* phi_init1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1);

    //! Destructor to delete #phi and #gaussian_kernel.
    ~ActiveContour();

    //! Evolves the active contour of one iteration or one step. This function calls the functions #do_one_iteration_in_cycle1 or #do_one_iteration_in_cycle2.
    void evolve_one_iteration();

    //! Evolves directly the active contour at the final state, when #hasAlgoStopping egals to \c true.
    void evolve_to_final_state();

    //! Evolves the active contour \a n times or \a n iterations.
    void evolve_n_iterations(int n);

    //! Initialization for each new frame buffer, used for video tracking.
    virtual void initialize_for_each_frame();

    //! Getter function for the pointer #phi.
    const char* const get_phi() const;
    //! Getter function for the linked list #Lout.
    const ofeli::List& get_Lout() const;
    //! Getter function for the linked list #Lin.
    const ofeli::List& get_Lin() const;

    //! Getter function for the boolean #hasListsChanges.
    bool get_hasListsChanges() const;
    //! Getter function for the boolean #hasOscillation.
    bool get_hasOscillation() const;
    //! Getter function for #iterations.
    int get_iterations() const;
    //! Getter function for #max_iterations.
    int get_max_iterations() const;
    //! Getter function for the boolean #hasAlgoStopping.
    bool get_hasAlgoStopping() const;

protected :

    //! Image width, i.e. number of columns.
    const int img_width;
    //! Image height, i.e. number of rows.
    const int img_height;
    //! Image size, i.e. number of pixels. Obviously, it egals to #img_width × #img_height.
    const int img_size;
    //! Pointer on the level-set function buffer.
    char* const phi;

    //! General initialization used for video tracking.
    void do_general_initialization_for_each_frame();

    //! Gives the square of a value.
    template<typename type>
    inline static type square(type value) { return value*value; }

private :

    //! Initialization of variables in the both constructors.
    void initialize();

    //! Does a normalised gaussian kernel without to divide by \a π
    static const int* const make_gaussian_kernel(int kernel_length1, double sigma1);



    //! Function called by evolve_one_iteration() for external or data dependant evolution with \a Fd speed.
    void do_one_iteration_in_cycle1();

    //! Function called by evolve_one_iteration() for a curve smoothing or internal evolution with \a Fint speed.
    void do_one_iteration_in_cycle2();



    //! Outward local movement of the curve for a current point (\a x,\a y) of #Lout that is switched in #Lin.
    inline void switch_in(int i, int x, int y);

    //! Second step of procedure #switch_in.
    inline void add_Rout_neighbor_to_Lout(int i_neighbor);

    //! Inward local movement of the curve for a current point (\a x,\a y) of #Lin that is switched in #Lout.
    inline void switch_out(int i, int x, int y);

    //! Second step of procedure #switch_out.
    inline void add_Rin_neighbor_to_Lin(int i_neighbor);

    //! Computes the internal speed  Fint for a current point (\a x,\a y) of #Lout or #Lin.
    inline int compute_internal_speed_Fint(int i, int x, int y);

    //! Gives the sign of a char. Return the integer -1 or 1.
    static inline int signum_function(char i);

    //! Computes the external speed \a Fd for a current point (\a x,\a y) of #Lout or #Lin.
    virtual inline int compute_external_speed_Fd(int i, int x, int y);

    //! Finds if a current point (\a x,\a y) of #Lin is redundant.
    inline bool find_isRedundantPointOfLin(int x, int y) const;

    //! Eliminates redundant points in #Lin.
    void eliminate_redundant_in_Lin();

    //! Finds if a current point (\a x,\a y) of #Lout is redundant.
    inline bool find_isRedundantPointOfLout(int x, int y) const;

    //! Eliminates redundant points in #Lout.
    void eliminate_redundant_in_Lout();



    //! Specific step reimplemented in the children active contours ACwithoutEdges and ACwithoutEdgesYUV to calculate the means \a Cout and \a Cin in \a O(1) or \a O(#lists_length) with updates counting and not in \a O(#img_size).
    virtual inline void do_specific_beginning_step();

    //! Specific step reimplemented in the children active contours ACwithoutEdges and ACwithoutEdgesYUV to update the variables to calculate the means \a Cout and \a Cin before each #switch_in, in the cycle 1.
    virtual inline void do_specific_in_step(int i, int x, int y);

    //! Specific step reimplemented in the children active contours ACwithoutEdges and ACwithoutEdgesYUV to update the variables to calculate the means \a Cout and \a Cin before each #switch_in, in the cycle 2.
    virtual inline void do_specific_in_step2(int i, int x, int y);

    //! Specific step reimplemented in the children active contours ACwithoutEdges and ACwithoutEdgesYUV to update the variables to calculate the means \a Cout and \a Cin before each #switch_out, in the cycle 1.
    virtual inline void do_specific_out_step(int i, int x, int y);

    //! Specific step reimplemented in the children active contours ACwithoutEdges and ACwithoutEdgesYUV to update the variables to calculate the means \a Cout and \a Cin before each #switch_out, in the cycle 2.
    virtual inline void do_specific_out_step2(int i, int x, int y);



    //! First stopping condition called at the end of each iteration of the cycle 1. Update of #hasListsChanges. Two tests on #hasListsChanges and #iterations are done. If one test is positive, #hasCycle1 egals to \c false if #hasSmoothingCycle is \c true, otherwise #hasAlgoStopping egals to \c true and the active contour stops.
    void calculate_stopping_conditions1();

    //! Second stopping condition called at the end of the cycle 2. Update of #hasOscillation. Two tests on #hasOscillation and #iterations are done. If one test is positive, #hasAlgoStopping egals to \c true and the active contour stops.
    void calculate_stopping_conditions2();

    //! Calculates #max_iterations.
    static inline int calculate_max_iterations(int img_width1, int img_height1);

    //! Boolean egals to \c true to have the curve smoothing, evolutions in the cycle 2 with the internal speed Fint.
    const bool hasSmoothingCycle;
    //! Kernel length of the gaussian filter for the curve smoothing.
    const int kernel_length;
    //! Standard deviation of the gaussian kernel for the curve smoothing.
    const double sigma;
    //! Boolean egals to \c true to eliminate the redundant points in #Lin and #Lout lists.
    const bool hasCleaning;

    //! Pointer on the gaussian kernel buffer used to calculate Fint.
    const int* const gaussian_kernel;

    //! Number of times the active contour evolves.
    int iterations;
    //! Maximum number of times the active contour can evolve.
    const int max_iterations;
    //! Boolean egals to \c true to stop the algorithm.
    bool hasAlgoStopping;

    //! List of points belong to the outside boundary.
    ofeli::List Lout;
    //! List of points belong to the inside boundary.
    ofeli::List Lin;

    //! \a kernel_radius = (  #kernel_length - 1) / 2 with #kernel_length impair.
    const int kernel_radius;


    //! Boolean egals to \c true if one point of #Lout at least is switched in during the scan through the #Lout list.
    bool hasOutwardEvolution;
    //! Boolean egals to \c true if one point of #Lin at least is switched out during the scan through the #Lin list.
    bool hasInwardEvolution;

    //! Number of iterations the active contour evolves in the cycle 1 with \a Fd speed.
    int Na;
    //! Number of maximum iterations the active contour evolves in the cycle 1 with \a Fd speed.
    const int Na_max;
    //! Number of iterations the active contour evolves in the cycle 2 with \a Fint speed.
    int Ns;
    //! Number of maximum iterations the active contour evolves in the cycle 2 with \a Fint speed.
    const int Ns_max;
    //! Number of iterations in the last cycle 2.
    int Ns_last;

    //! Boolean egals to true if #hasOutwardEvolution egals to \c true or #hasInwardEvolution egals to \c true.
    bool hasListsChanges;
    //! Length of #Lin + length of #Lout.
    int lists_length;
    //! Length of #Lin + length of #Lout of the previous iteration.
    int previous_lists_length;
    //! Boolean egals to \c true if oscillations are detected. In this case, #hasAlgoStopping egals to \c true
    bool hasOscillation;
    //! To count the number of times the relative gap of the lists length are less of a constant, fixed at 0.5%.
    int counter;
    //! Boolean egals to \c false to go away the cycle 1 and never to go back into.
    bool hasCycle1;
};

}

#endif // ACTIVECONTOUR_HPP



//! \class ofeli::ActiveContour
//! The class ActiveContour contains the implementation of the Fast-Two-Cycle (FTC) algorithm of Shi and Karl as it describes in their article "A real-time algorithm for the approximation of level-set based curve evolution" published in IEEE Transactions on Image Processing in may 2008.
//! This class should be never instantiated because the function to calculate the external speed \a Fd is too general. The 3 classes #ofeli::ACwithoutEdges, #ofeli::ACwithoutEdgesYUV and #ofeli::GeodesicAC, inherit of the class ActiveContour, with for each class it own implementation of this function.

/**
 * \fn ActiveContour::ActiveContour(int img_width1, int img_height1, bool hasEllipse1, double init_width1, double init_height1, double center_x1, double center_y1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1)
 * \param img_width1 Image width, i.e. number of columns. Passed to #img_width.
 * \param img_height1 Image height, i.e. number of rows. Passed to #img_height.
 * \param hasEllipse1 Boolean to choose the shape of the active contour initialization, \c true for an ellipse or \c false for a rectangle.
 * \param init_width1 Width of the shape divided by the image #img_width.
 * \param init_height1 Height of the shape divided by the image #img_height.
 * \param center_x1 X-axis position or column index of the center of the shape divided by the image #img_width subtracted by 0.5.
 * \param center_y1 Y-axis position or row index of the center of the shape divided by the image #img_height subtracted by 0.5.\n
          To have the center of the shape into the image : -0.5 < center_x1 < 0.5 and -0.5 < center_y1 < 0.5.
 * \param hasSmoothingCycle1 Boolean to have or not the curve smoothing, evolutions in the cycle 2 with an internal speed Fint. Passed to #hasSmoothingCycle.
 * \param kernel_length1 Kernel length of the gaussian filter for the curve smoothing. Passed to #kernel_length.
 * \param sigma1 Standard deviation of the gaussian kernel for the curve smoothing. Passed to #sigma.
 * \param hasCleaning1 Boolean to eliminate or not redundant points in #Lin and #Lout lists. Passed to #hasCleaning.
 * \param Na1 Number of maximum iterations the active contour evolves in the cycle 1, external or data dependant evolutions with \a Fd speed. Passed to #Na_max.
 * \param Ns1 Number of maximum iterations the active contour evolves in the cycle 2, curve smoothing or internal evolutions with \a Fint speed. Passed to #Ns_max.
 */

/**
 * \fn ActiveContour::ActiveContour(int img_width1, int img_height1, const char* phi_init1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1)
 * \param img_width1 Image width, i.e. number of columns. Passed to #img_width.
 * \param img_height1 Image height, i.e. number of rows. Passed to #img_height.
 * \param phi_init1 Pointer on the initialized level-set function buffer. Copied to #phi.
 * \param hasSmoothingCycle1 Boolean to have or not the curve smoothing, evolutions in the cycle 2 with an internal speed Fint. Passed to #hasSmoothingCycle.
 * \param kernel_length1 Kernel length of the gaussian filter for the curve smoothing. Passed to #kernel_length.
 * \param sigma1 Standard deviation of the gaussian kernel for the curve smoothing. Passed to #sigma.
 * \param hasCleaning1 Boolean to eliminate or not redundant points in #Lin and #Lout lists. Passed to #hasCleaning.
 * \param Na1 Number of maximum iterations the active contour evolves in the cycle 1, external or data dependant evolutions with \a Fd speed. Passed to #Na_max.
 * \param Ns1 Number of maximum iterations the active contour evolves in the cycle 2, curve smoothing or internal evolutions with \a Fint speed. Passed to #Ns_max.
 */

/**
 * \fn inline void ActiveContour::switch_in(int i, int x, int y)
 * \param i offset of the buffer pointed by #phi with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel of #phi in the image X-axis or column index
 * \param y position of the current pixel of #phi in the image Y-axis or row index
 */

/**
 * \fn inline void ActiveContour::switch_out(int i, int x, int y)
 * \param i offset of the buffer pointed by #phi with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel of #phi in the image X-axis or column index
 * \param y position of the current pixel of #phi in the image Y-axis or row index
 */

/**
 * \fn inline int ActiveContour::compute_internal_speed_Fint(int i, int x, int y)
 * \param i offset of the buffer pointed by #phi with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel of #phi in the image X-axis or column index
 * \param y position of the current pixel of #phi in the image Y-axis or row index
 * \return Fint, the internal speed for the regularization of the active contour used in the cycle 2
 */

/**
 * \fn virtual inline int ActiveContour::compute_external_speed_Fd(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel of the input image in the image X-axis or column index
 * \param y position of the current pixel of the input image in the image Y-axis or row index
 * \return Fd, the external speed for the data dependant evolution of the active contour used in the cycle 1
 */

/**
 * \fn inline bool ActiveContour::find_isRedundantPointOfLin(int x, int y) const
 * \param x position of the current pixel of #phi in the image X-axis or column index
 * \param y position of the current pixel of #phi in the image Y-axis or row index
 * \return \c true if the point (\a x,\a y) of #Lin is redundant, otherwise, \c false
 */

/**
 * \fn inline bool ActiveContour::find_isRedundantPointOfLout(int x, int y) const
 * \param x position of the current pixel of #phi in the image X-axis or column index
 * \param y position of the current pixel of #phi in the image Y-axis or row index
 * \return \c true if the point (\a x,\a y) of #Lout is redundant, otherwise, \c false
 */

/**
 * \fn virtual inline void ActiveContour::do_specific_in_step(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel in the image X-axis or column index
 * \param y position of the current pixel in the image Y-axis or row index
 */

/**
 * \fn virtual inline void ActiveContour::do_specific_in_step2(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel in the image X-axis or column index
 * \param y position of the current pixel in the image Y-axis or row index
 */

/**
 * \fn virtual inline void ActiveContour::do_specific_out_step(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel in the image X-axis or column index
 * \param y position of the current pixel in the image Y-axis or row index
 */

/**
 * \fn virtual inline void ActiveContour::do_specific_out_step2(int i, int x, int y)
 * \param i offset of the image data buffer with \a i = \a x + \a y × #img_width
 * \param x position of the current pixel in the image X-axis or column index
 * \param y position of the current pixel in the image Y-axis or row index
 */

/**
 * \example interface.cpp
 * \code
 * // How to use the evolve_one_iteration() function in an interface ?
 *
 * ofeli::ACwithoutEdges acWithoutEdges1(img1, img_width1, img_height1, hasEllipse1, init_width1, init_height1, center_x1, center_y1, hasSmoothingCycle1, kernel_length1, sigma1, hasCleaning1, Na1, Ns1);
 *
 * // Declarations
 * const ofeli::List* Lout1 = &acWithoutEdges1.get_Lout();
 * const ofeli::List* Lin1 = &acWithoutEdges1.get_Lin();
 * int i // offset of the current pixel
 * unsigned char I // intensity of the current pixel

 * // Lout1 displayed in blue
 * unsigned char Rout = 0;
 * unsigned char Gout = 0;
 * unsigned char Bout = 255;
 *
 * // Lin1 displayed in red
 * unsigned char Rin = 255;
 * unsigned char Gin = 0;
 * unsigned char Bin = 0;

 * // Loop for the evolution of the active contour
 * do {
 *
 *     // erase the previous lists Lout1 and Lin1 of the displayed buffer
 *     for( List::const_Link iterator = Lout1->get_begin(); !iterator->end(); iterator = iterator->get_next() )
 *
 *         i = iterator->get_elem();
 *
 *         I = img1[i];
 *
 *         img_displayed[3*i] = I;
 *         img_displayed[3*i+1] = I;
 *         img_displayed[3*i+2] = I;
 *     }
 *
 *     for( List::const_Link iterator = Lin1->get_begin(); !iterator->end(); iterator = iterator->get_next() )
 *
 *         i = iterator->get_elem();
 *
 *         I = img1[i];
 *
 *         img_displayed[3*i] = I;
 *         img_displayed[3*i+1] = I;
 *         img_displayed[3*i+2] = I;
 *     }
 *
 *     // to evolve the active contour of one iteration or one step
 *     acWithoutEdges1.evolve_one_iteration();
 *
 *     // to get the temporary result
 *     Lout1 = &acWithoutEdges1.get_Lout();
 *     Lin1 = &acWithoutEdges1.get_Lin();
 *
 *     // put the color of lists into the displayed buffer
 *     for( List::const_Link iterator = Lout1->get_begin(); !iterator->end(); iterator = iterator->get_next() )
 *
 *         i = 3*iterator->get_elem();
 *
 *         img_displayed[i] = Rout;
 *         img_displayed[i+1] = Gout;
 *         img_displayed[i+2] = Bout;
 *     }
 *
 *     for( List::const_Link iterator = Lin1->get_begin(); !iterator->end(); iterator = iterator->get_next() )
 *
 *         i = 3*iterator->get_elem();
 *
 *         img_displayed[i] = Rin;
 *         img_displayed[i+1] = Gin;
 *         img_displayed[i+2] = Bin;
 *     }
 *
 *     // paint event, refresh the widget witch display the buffer img_displayed
 *     update();
 *
 * } while( !acWithoutEdges1.get_hasAlgoStopping() );
 * \endcode
 */
