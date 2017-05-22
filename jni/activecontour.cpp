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

#include "activecontour.hpp"
#include <cmath> // for the function "exp"
#include <cstdlib> // for the function "abs"
#include <cstring> // for the function "memcpy"
#include <algorithm> // for the function "std::max"
#include <iostream> // for the object "std::cerr"

namespace ofeli
{

ActiveContour::ActiveContour(int img_width1, int img_height1, bool hasEllipse1, double init_width1, double init_height1, double center_x1, double center_y1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1) : img_width(img_width1), img_height(img_height1), hasSmoothingCycle(hasSmoothingCycle1), kernel_length(kernel_length1), sigma(sigma1), hasCleaning(hasCleaning1), Na_max(Na1), Ns_max(Ns1), img_size(img_width1*img_height1), kernel_radius((kernel_length1-1)/2), phi(new char[img_width1*img_height1]), gaussian_kernel(make_gaussian_kernel(kernel_length1, sigma1)), max_iterations(calculate_max_iterations(img_width1, img_height1))
{
    // active contour initialization with phi, Lout and Lin

    int x, y; // position of the current pixel

    // do an ellipse
    if( hasEllipse1 )
    {
        for( int i = 0; i < img_size; i++ )
        {
            // i = x+y*img_width so
            y = i/img_width;
            x = i-y*img_width;

            // ellipse inequation
            if( square( double(y)-(1.0+2.0*center_y1)*double(img_height)/2.0 ) / square( init_height1*double(img_height)/2.0 ) + square( double(x)-(1.0+2.0*center_x1)*double(img_width)/2.0 ) / square( init_width1*double(img_width)/2.0 ) > 1.0 )
            {
                phi[i] = 1; // outside
                Lout.push_front(i);
            }
            else
            {
                phi[i] = -1; // inside
                Lin.push_front(i);
            }
        }
    }

    // do a rectangle
    else
    {
        for( int i = 0; i < img_size; i++ )
        {
            // i = x+y*img_width so
            y = i/img_width;
            x = i-y*img_width;

            if( double(y) > ((1.0-init_height1)*double(img_height)/2.0 + center_y1*double(img_height)) && double(y) < ((double(img_height)-(1.0-init_height1)*double(img_height)/2.0) + center_y1*double(img_height)) && double(x) > ((1.0-init_width1)*double(img_width)/2.0 + center_x1*double(img_width)) && double(x) < ((double(img_width)-(1.0-init_width1)*double(img_width)/2.0) + center_x1*double(img_width)) )
            {
                phi[i] = -1; // inside
                Lin.push_front(i);
            }
            else
            {
                phi[i] = 1; // outside
                Lout.push_front(i);
            }
        }
    }

    eliminate_redundant_in_Lout(); // clean Lout boundary
    eliminate_redundant_in_Lin(); // clean Lin boundary

    // variables initialization
    initialize();
}

ActiveContour::ActiveContour(int img_width1, int img_height1, const char* phi_init1, bool hasSmoothingCycle1, int kernel_length1, double sigma1, bool hasCleaning1, int Na1, int Ns1) : img_width(img_width1), img_height(img_height1), hasSmoothingCycle(hasSmoothingCycle1), kernel_length(kernel_length1), sigma(sigma1), hasCleaning(hasCleaning1), Na_max(Na1), Ns_max(Ns1), img_size(img_width1*img_height1), kernel_radius((kernel_length1-1)/2), phi(new char[img_width1*img_height1]), gaussian_kernel(make_gaussian_kernel(kernel_length1, sigma1)), max_iterations(calculate_max_iterations(img_width1, img_height1))
{
    if( phi_init1 == NULL )
    {
        std::cerr << std::endl << "The pointer phi_init1 must be a non-null pointer, it must be allocated." << std::endl;
    }
    else
    {
        // active contour initialization with phi, Lout and Lin
        for( int i = 0; i < img_size; i++ )
        {
            phi[i] = phi_init1[i];

            if( phi_init1[i] == 1 )
            {
                Lout.push_front(i);
            }
            else if( phi_init1[i] == -1 )
            {
                Lin.push_front(i);
            }
        }

        // variables initialization
        initialize();
    }
}

void ActiveContour::initialize()
{
    // initialization of 4 variables for the oscillation stopping condition
    previous_lists_length = 999999;
    lists_length = 0;
    counter = 0;

    do_general_initialization_for_each_frame();

    return;
}

void ActiveContour::do_general_initialization_for_each_frame()
{
    Na = 0; // in the cycle 1
    Ns = 0; // in the cycle 2
    Ns_last = 0; // in the last cycle 2
    hasCycle1 = true;

    // 3 variables of 3 stopping conditions
    hasListsChanges = true;
    hasOscillation = false;
    iterations = 0;

    hasAlgoStopping = false; // to start the algorithm

    return;
}

void ActiveContour::initialize_for_each_frame()
{
    do_general_initialization_for_each_frame();

    return;
}

const int* const ActiveContour::make_gaussian_kernel(int kernel_length1, double sigma1)
{
    // kernel_length impair and strictly positive
    if( kernel_length1 % 2 == 0 )
    {
        kernel_length1--;
    }

    if( kernel_length1 < 1 )
    {
        kernel_length1 = 1;
    }

    // protection against /0
    if( sigma1 < 0.000000001 )
    {
        sigma1 = 0.000000001;
    }

    int x, y; // position of the current pixel

    const int kernel_size1 = kernel_length1*kernel_length1;
    const int kernel_radius1 = (kernel_length1-1)/2;

    int* const gaussian_kernel1 = new int[kernel_size1];

    for( int i = 0; i < kernel_size1; i++ )
    {
        // i = x+y*kernel_length so
        y = i/kernel_length1;
        x = i-y*kernel_length1;

        gaussian_kernel1[i] = int( 0.5+100000.0*exp( -( square(double(y)-double(kernel_radius1)) + square(double(x)-double(kernel_radius1)) ) / (2*square(sigma1)) ) );
    }

    return gaussian_kernel1;
}

ActiveContour::~ActiveContour()
{
    delete[] gaussian_kernel;
    delete[] phi;
}

void ActiveContour::do_one_iteration_in_cycle1()
{
    int i, x, y; // offset and position of the current point of the active contour

    hasOutwardEvolution = false;

    do_specific_beginning_step(); // to calculate Cout and Cin means in O(1) in Chan-Vese model.

    // scan through Lout
    for( Lout.begin(); !Lout.end();  )
    {
        i = Lout.get_current();

        // i = x+y*img_width so
        y = i/img_width;
        x = i-y*img_width;

        if( compute_external_speed_Fd(i,x,y) > 0 )
        {
            do_specific_in_step(i,x,y); // updates of the variables to calculate the means Cout and Cin

            switch_in(i,x,y); // outward local movement

            if( !hasOutwardEvolution )
            {
                hasOutwardEvolution = true;
            }
        }
        else
        {
            Lout.next();
        }
    }

    if( hasOutwardEvolution && hasCleaning )
    {
        eliminate_redundant_in_Lin(); // Lin boundary cleaning
    }

    hasInwardEvolution = false;

    // scan through Lin
    for( Lin.begin(); !Lin.end();  )
    {
        i = Lin.get_current();

        // i = x+y*img_width so
        y = i/img_width;
        x = i-y*img_width;

        if( compute_external_speed_Fd(i,x,y) < 0 )
        {
            do_specific_out_step(i,x,y); // updates of the variables to calculate the means Cout and Cin

            switch_out(i,x,y); // inward local movement

            if( !hasInwardEvolution )
            {
                hasInwardEvolution = true;
            }
        }
        else
        {
            Lin.next();
        }
    }

    if( hasInwardEvolution && hasCleaning )
    {
        eliminate_redundant_in_Lout(); // Lout boundary cleaning
    }

    calculate_stopping_conditions1();

    return;
}

void ActiveContour::do_one_iteration_in_cycle2()
{
    int i, x, y; // offset and position of the current point of the active contour

    // scan through Lout
    for( Lout.begin(); !Lout.end();  )
    {
        i = Lout.get_current();

        // i = x+y*img_width so
        y = i/img_width;
        x = i-y*img_width;

        if( Ns == Ns_max-1 )
        {
            lists_length++;
        }

        if( compute_internal_speed_Fint(i,x,y) < 0 )
        {
            do_specific_in_step2(i,x,y); // updates of the variables to calculate the means Cout and Cin

            switch_in(i,x,y); // outward local movement
        }
        else
        {
            Lout.next();
        }
    }

    if( hasCleaning )
    {
        eliminate_redundant_in_Lin(); // Lin boundary cleaning
    }

    // scan through Lin
    for( Lin.begin(); !Lin.end();  )
    {
        i = Lin.get_current();

        // i = x+y*img_width so
        y = i/img_width;
        x = i-y*img_width;

        if( Ns == Ns_max-1 )
        {
            lists_length++;
        }

        if( compute_internal_speed_Fint(i,x,y) > 0 )
        {
            do_specific_out_step2(i,x,y); // updates of the variables to calculate the means Cout and Cin

            switch_out(i,x,y); // inward local movement
        }
        else
        {
            Lin.next();
        }
    }

    if( hasCleaning )
    {
        eliminate_redundant_in_Lout(); // Lout boundary cleaning
    }

    iterations++;

    return;
}

void ActiveContour::evolve_one_iteration()
{
    if( !hasAlgoStopping )
    {
        // Na_max iterations in the cycle 1
        if( hasCycle1 )
        {
            if( Na < Na_max-1 )
            {
                do_one_iteration_in_cycle1(); // data dependant evolution
                Na++;
                return;
            }

            // last iteration of the cycle 1
            if( Na == Na_max-1 )
            {
                do_one_iteration_in_cycle1(); // data dependant evolution

                if( hasSmoothingCycle )
                {
                    Na++; // in order to do the cycle2, now Na = Na_max
                }
                else
                {
                    Na = 0; // in order to restart the cycle1
                }

                return;
            }
        }

        // Ns_max iterations in the cycle 2
        if( hasSmoothingCycle )
        {
            if( Ns < Ns_max-1 )
            {
                do_one_iteration_in_cycle2(); // ac regularization
                Ns++;
                return;
            }

            // last iteration of the cycle 2
            if( Ns == Ns_max-1 )
            {
                do_one_iteration_in_cycle2(); // ac regularization
                calculate_stopping_conditions2();
                // in order to restart the both cycles
                Na = 0;
                Ns = 0;
                return;
            }
        }
    }

    return;
}

void ActiveContour::evolve_n_iterations(int n)
{
    const int N = iterations+n;

    while( !hasAlgoStopping && iterations < N )
    {
        for(  ; Na < Na_max && hasCycle1 && iterations < N; Na++ )
        {
            do_one_iteration_in_cycle1();
        }
        Na = 0;

        if( hasSmoothingCycle )
        {
            for(  ; Ns < Ns_max && iterations < N; Ns++ )
            {
                do_one_iteration_in_cycle2();
            }
            Ns = 0;
            calculate_stopping_conditions2();
        }
    }

    return;
}

void ActiveContour::evolve_to_final_state()
{
    while( !hasAlgoStopping )
    {
        for(  ; Na < Na_max && hasCycle1; Na++ )
        {
            do_one_iteration_in_cycle1();
        }
        Na = 0;

        if( hasSmoothingCycle )
        {
            for(  ; Ns < Ns_max; Ns++ )
            {
                do_one_iteration_in_cycle2();
            }
            Ns = 0;

            calculate_stopping_conditions2();
        }
    }

    return;
}

inline void ActiveContour::switch_in(int i, int x, int y)
{
    // Outward local movement

    // current point (x,y) of Lout is switched in Lin
    Lin.switch_current(Lout);
    // <==> Lout.erase(); Lin.push_front(i); but without use of operator delete and operator new.

    phi[i] = -1; // value is changed from 1 to -1


    //==============   4-connex version   =============

    if( y-1 >= 0 )
    {
        add_Rout_neighbor_to_Lout( x+(y-1)*img_width );
    }
    if( x-1 >= 0 )
    {
        add_Rout_neighbor_to_Lout( (x-1)+y*img_width );
    }
    if( x+1 < img_width )
    {
        add_Rout_neighbor_to_Lout( (x+1)+y*img_width );
    }
    if( y+1 < img_height )
    {
        add_Rout_neighbor_to_Lout( x+(y+1)*img_width );
    }

    //=================================================


    //==============   8-connex version   =============

    // if not in the border of the image, no neighbor test
    /*if( x > 0 && x < img_width-1 && y > 0 && y < img_height-1 )
    {
        // scan through a 8-connex neighborhood
        for( int dy = -1; dy <= 1; dy++ )
        {
            for( int dx = -1; dx <= 1; dx++ )
            {
                if( !( dx == 0 && dy == 0 ) )
                {
                    add_Rout_neighbor_to_Lout( (x+dx)+(y+dy)*img_width );
                }
            }
        }
    }

    // if in the border, neighbors tests
    else
    {
        // scan through a 8-connex neighborhood
        for( int dy = -1; dy <= 1; dy++ )
        {
            for( int dx = -1; dx <= 1; dx++ )
            {
                if( !( dx == 0 && dy == 0 ) )
                {
                    // existence tests
                    if( x+dx >= 0 && x+dx < img_width && y+dy >= 0 && y+dy < img_height )
                    {
                        add_Rout_neighbor_to_Lout( (x+dx)+(y+dy)*img_width );
                    }
                }
            }
        }
    }*/

    //=================================================

    return;
}

inline void ActiveContour::add_Rout_neighbor_to_Lout(int i_neighbor)
{
    // if a neighbor ∈ Rout
    if( phi[i_neighbor] == 3 )
    {
        phi[i_neighbor] = 1;

        // neighbor ∈ Rout --> ∈ Lout
        Lout.push_front(i_neighbor);
    }

    return;
}

inline void ActiveContour::switch_out(int i, int x, int y)
{
    // Inward local movement

    // current point (x,y) of Lin is switched in Lout
    Lout.switch_current(Lin);
    // <==> Lin.erase(); Lout.push_front(i); but without use of operator delete and operator new.

    phi[i] = 1; // value is changed from -1 to 1


    //==============   4-connex version   =============

    if( y-1 >= 0 )
    {
        add_Rin_neighbor_to_Lin( x+(y-1)*img_width );
    }
    if( x-1 >= 0 )
    {
        add_Rin_neighbor_to_Lin( (x-1)+y*img_width );
    }
    if( x+1 < img_width )
    {
        add_Rin_neighbor_to_Lin( (x+1)+y*img_width );
    }
    if( y+1 < img_height )
    {
        add_Rin_neighbor_to_Lin( x+(y+1)*img_width );
    }

    //=================================================


    //==============   8-connex version   =============

    // if not in the border of the image, no neighbor test
    /*if( x > 0 && x < img_width-1 && y > 0 && y < img_height-1 )
    {
        // scan through a 8-connex neighborhood
        for( int dy = -1; dy <= 1; dy++ )
        {
            for( int dx = -1; dx <= 1; dx++ )
            {
                if( !( dx == 0 && dy == 0 ) )
                {
                    add_Rin_neighbor_to_Lin( (x+dx)+(y+dy)*img_width );
                }
            }
        }
    }

    // if in the border, neighbors tests
    else
    {
        // scan through a 8-connex neighborhood
        for( int dy = -1; dy <= 1; dy++ )
        {
            for( int dx = -1; dx <= 1; dx++ )
            {
                if( !( dx == 0 && dy == 0 ) )
                {
                    // existence tests
                    if( x+dx >= 0 && x+dx < img_width && y+dy >= 0 && y+dy < img_height )
                    {
                        add_Rin_neighbor_to_Lin( (x+dx)+(y+dy)*img_width );
                    }
                }
            }
        }
    }*/

    //=================================================

    return;
}

inline void ActiveContour::add_Rin_neighbor_to_Lin(int i_neighbor)
{
    // if a neighbor ∈ Rin
    if( phi[i_neighbor] == -3 )
    {
        phi[i_neighbor] = -1;

        // neighbor ∈ Rin --> ∈ Lin
        Lin.push_front(i_neighbor);
    }

    return;
}

inline int ActiveContour::compute_internal_speed_Fint(int i, int x, int y)
{
    int Fint = 0;

    // if not in the border of the image, no neighbor test
    if( x > kernel_radius-1 && x < img_width-kernel_radius && y > kernel_radius-1 && y < img_height-kernel_radius )
    {
        for( int dy = -kernel_radius; dy <= kernel_radius; dy++ )
        {
            for( int dx = -kernel_radius; dx <= kernel_radius; dx++ )
            {
                Fint += gaussian_kernel[ (kernel_radius+dx)+(kernel_radius+dy)*kernel_length ]*signum_function( phi[(x+dx)+(y+dy)*img_width] );
            }
        }
    }
    // if in the border of the image, tests of neighbors
    else
    {
        for( int dy = -kernel_radius; dy <= kernel_radius; dy++ )
        {
            for(int dx = -kernel_radius; dx <= kernel_radius; dx++ )
            {
                if( x+dx >= 0 && x+dx < img_width && y+dy >= 0 && y+dy < img_height )
                {
                    Fint += gaussian_kernel[ (kernel_radius+dx)+(kernel_radius+dy)*kernel_length ]*signum_function( phi[(x+dx)+(y+dy)*img_width] );
                }
                else
                {
                    Fint += gaussian_kernel[ (kernel_radius+dx)+(kernel_radius+dy)*kernel_length ]*signum_function( phi[i] );
                }
            }
        }
    }

    return Fint;
}

inline int ActiveContour::signum_function(char i)
{
    if( i < 0 )
    {
        return -1;
    }
    else
    {
        return 1;
    }
}

inline bool ActiveContour::find_isRedundantPointOfLin(int x, int y) const
{
    //==============   4-connex version   =============

    // if ∃ a neighbor ∈ Lout | ∈ Rout
    if( y-1 >= 0 )
    {
        if( phi[ x+(y-1)*img_width ] >= 0 )
        {
            return false; // is not redundant point of Lin
        }
    }
    if( x-1 >= 0 )
    {
        if( phi[ (x-1)+y*img_width ] >= 0 )
        {
            return false; // is not redundant point of Lin
        }
    }
    if( x+1 < img_width )
    {
        if( phi[ (x+1)+y*img_width ] >= 0 )
        {
            return false; // is not redundant point of Lin
        }
    }
    if( y+1 < img_height )
    {
        if( phi[ x+(y+1)*img_width ] >= 0 )
        {
            return false; // is not redundant point of Lin
        }
    }

    //=================================================


    //==============   8-connex version   =============

    // if not in the border of the image, no neighbor test
    /*if( x > 0 && x < img_width-1 && y > 0 && y < img_height-1 )
    {
        // scan through a 8-connex neighborhood
        for( int dy = -1; dy <= 1; dy++ )
        {
            for( int dx = -1; dx <= 1; dx++ )
            {
                if( !( dx == 0 && dy == 0 ) )
                {
                    // if ∃ a neighbor ∈ Lout | ∈ Rout
                    if( phi[ (x+dx)+(y+dy)*img_width ] >= 0 )
                    {
                        return false; // is not redundant point of Lin
                    }
                }
            }
        }
    }

    // if in the border of the image, tests of neighbors
    else
    {
        // scan through a 8-connex neighborhood
        for( int dy = -1; dy <= 1; dy++ )
        {
            for( int dx = -1; dx <= 1; dx++ )
            {
                if( !( dx == 0 && dy == 0 ) )
                {
                    // neighbors tests
                    if( x+dx >= 0 && x+dx < img_width && y+dy >= 0 && y+dy < img_height )
                    {
                        // if ∃ a neighbor ∈ Lout | ∈ Rout
                        if( phi[ (x+dx)+(y+dy)*img_width ] >= 0 )
                        {
                            return false; // is not redundant point of Lin
                        }
                    }
                }
            }
        }
    }*/

    //=================================================

    // ==> ∀ neighbors ∈ Lin | ∈ Rin
    return true; // is redundant point of Lin
}

inline bool ActiveContour::find_isRedundantPointOfLout(int x, int y) const
{
    //==============   4-connex version   =============

    // if ∃ a neighbor ∈ Lin | ∈ Rin
    if( y-1 >= 0 )
    {
        if( phi[ x+(y-1)*img_width ] <= 0 )
        {
            return false; // is not redundant point of Lout
        }
    }
    if( x-1 >= 0 )
    {
        if( phi[ (x-1)+y*img_width ] <= 0 )
        {
            return false; // is not redundant point of Lout
        }
    }
    if( x+1 < img_width )
    {
        if( phi[ (x+1)+y*img_width ] <= 0 )
        {
            return false; // is not redundant point of Lout
        }
    }
    if( y+1 < img_height )
    {
        if( phi[ x+(y+1)*img_width ] <= 0 )
        {
            return false; // is not redundant point of Lout
        }
    }

    //=================================================


    //==============   8-connex version   =============

    // if not in the border of the image, no neighbor test
    /*if( x > 0 && x < img_width-1 && y > 0 && y < img_height-1 )
    {
        // scan through a 8-connex neighborhood
        for( int dy = -1; dy <= 1; dy++ )
        {
            for( int dx = -1; dx <= 1; dx++ )
            {
                if( !( dx == 0 && dy == 0 ) )
                {
                    // if ∃ a neighbor ∈ Lin | ∈ Rin
                    if( phi[ (x+dx)+(y+dy)*img_width ] <= 0 )
                    {
                        return false; // is not redundant point of Lout
                    }
                }
            }
        }
    }

    // if in the border of the image, tests of neighbors
    else
    {
        // scan through a 8-connex neighborhood
        for( int dy = -1; dy <= 1; dy++ )
        {
            for( int dx = -1; dx <= 1; dx++ )
            {
                if( !( dx == 0 && dy == 0 ) )
                {
                    // neighbors tests
                    if( x+dx >= 0 && x+dx < img_width && y+dy >= 0 && y+dy < img_height )
                    {
                        // if ∃ a neighbor ∈ Lin | ∈ Rin
                        if( phi[ (x+dx)+(y+dy)*img_width ] <= 0 )
                        {
                            return false; // is not redundant point of Lout
                        }
                    }
                }
            }
        }
    }*/

    //=================================================

    // ==> ∀ neighbors ∈ Lout | ∈ Rout
    return true; // is redundant point of Lout
}

void ActiveContour::eliminate_redundant_in_Lin()
{
    int i, x, y; // offset and position of the current point of the active contour

    // scan through Lin
    for( Lin.begin(); !Lin.end();  )
    {
        i = Lin.get_current();

        // i = x+y*img_width so
        y = i/img_width;
        x = i-y*img_width;

        // if ∀ neighbors ∈ Lin | ∈ Rin
        if( find_isRedundantPointOfLin(x,y) )
        {
            phi[i] = -3; // phi(point) = -1 --> -3
            Lin.erase(); // point ∈ Lin --> ∈ Rin
        }
        else
        {
            Lin.next();
        }
    }

    return;
}

void ActiveContour::eliminate_redundant_in_Lout()
{
    int i, x, y; // offset and position of the current point of the active contour

    // scan through Lout
    for( Lout.begin(); !Lout.end();  )
    {
        i = Lout.get_current();

        // i = x+y*img_width so
        y = i/img_width;
        x = i-y*img_width;

        // if ∀ neighbors ∈ Lout | ∈ Rout
        if( find_isRedundantPointOfLout(x,y) )
        {
            phi[i] = 3; // phi(point) = 1 --> 3
            Lout.erase(); // point ∈ Lout --> ∈ Rout
        }
        else
        {
            Lout.next();
        }
    }

    return;
}


inline int ActiveContour::compute_external_speed_Fd(int i, int x, int y)
{
    // this class should never be instantiated

    return -1; // always an inward movement of the active contour, not a very discriminant speed...

    // reimplement a better and data-dependant function in a child class
}

// at the end of each iteration in the cycle 1
void ActiveContour::calculate_stopping_conditions1()
{
    if( !hasInwardEvolution && !hasOutwardEvolution )
    {
        hasListsChanges = false;
    }

    iterations++;

    if( !hasListsChanges || iterations >= max_iterations )
    {
        hasCycle1 = false;

        if( !hasSmoothingCycle )
        {
            hasAlgoStopping = true;
        }
    }

    return;
}

// at the end of the last iteration of the cycle 2
void ActiveContour::calculate_stopping_conditions2()
{
    // if the relative difference of active contour length between two cycle2 is less of 0.01
    if( double(abs(previous_lists_length-lists_length))/double(std::max(lists_length,previous_lists_length)) < 0.01 )
    {
        counter++;
    }
    // not consecutivly ==> reinitialization
    else
    {
        if( counter != 0 )
        {
            counter = 0;
        }
    }

    // if 3 times consecutivly
    if( counter == 3 )
    {
        hasOscillation = true;
    }

    if( hasOscillation || iterations >= max_iterations || !hasCycle1 )
    {
        hasAlgoStopping = true;
    }

    // keep last length to compare after
    previous_lists_length = lists_length;

    // reinitialization
    lists_length = 0;

    return;
}

inline int ActiveContour::calculate_max_iterations(int width1, int height1)
{
    if( width1 > height1 )
    {
        return 5*height1;
    }
    else
    {
        return 5*width1;
    }
}

// template functions square is defined in the header file

// 3 getters to see the result
const char* const ActiveContour::get_phi() const
{
    return phi;
}

const ofeli::List& ActiveContour::get_Lout() const
{
    return Lout;
}

const ofeli::List& ActiveContour::get_Lin() const
{
    return Lin;
}



// 5 getters to see the variables of the stopping conditions
bool ActiveContour::get_hasListsChanges() const
{
    return hasListsChanges;
}

bool ActiveContour::get_hasOscillation() const
{
    return hasOscillation;
}

int ActiveContour::get_iterations() const
{
    return iterations;
}

int ActiveContour::get_max_iterations() const
{
    return max_iterations;
}

bool ActiveContour::get_hasAlgoStopping() const
{
    return hasAlgoStopping;
}

void ActiveContour::do_specific_beginning_step()
{
    return;
}

inline void ActiveContour::do_specific_in_step(int i, int x, int y)
{
    return;
}

inline void ActiveContour::do_specific_out_step(int i, int x, int y)
{
    return;
}

inline void ActiveContour::do_specific_in_step2(int i, int x, int y)
{
    return;
}


inline void ActiveContour::do_specific_out_step2(int i, int x, int y)
{
    return;
}

}
