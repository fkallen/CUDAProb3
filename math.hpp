/*
This file is part of CUDAProb3++.

CUDAProb3++ is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CUDAProb3++ is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with CUDAProb3++.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUDAPROB3_MATH_HPP
#define CUDAPROB3_MATH_HPP

#include "hpc_helpers.cuh"

namespace cudaprob3{

    namespace math{

        template<typename T>
        struct ComplexNumber{
            T re;
            T im;
        };

        template<typename T>
        HOSTDEVICEQUALIFIER
        constexpr T ct_sqr(T x){
            return x * x;
        }

        template<typename T>
        HOSTDEVICEQUALIFIER
        constexpr T ct_cube(T x){
            return x * x * x;
        }


        /*
        *   multiply complex 3x3 matrix
        *        C = A X B
        */
        template<typename FLOAT_T>
        HOSTDEVICEQUALIFIER
        void multiply_complex_matrix(ComplexNumber<FLOAT_T> A[][3], ComplexNumber<FLOAT_T> B[][3], ComplexNumber<FLOAT_T> C[][3]){

            for (int i=0; i<3; i++) {

                for (int j=0; j<3; j++) {

                    for (int k=0; k<3; k++) {
                        C[i][j].re += A[i][k].re*B[k][j].re-A[i][k].im*B[k][j].im;
                        C[i][j].im += A[i][k].im*B[k][j].re+A[i][k].re*B[k][j].im;
                    }
                }
            }
        }

        /*
        *   multiply complex 3x3 matrix and 3 vector
        *        W = A X V
        */
        template<typename FLOAT_T>
        HOSTDEVICEQUALIFIER
        void multiply_complex_matvec(ComplexNumber<FLOAT_T> A[][3], ComplexNumber<FLOAT_T> V[3], ComplexNumber<FLOAT_T> W[3]){

            for(int i=0;i<3;i++) {
                W[i].re = A[i][0].re*V[0].re-A[i][0].im*V[0].im+
                    A[i][1].re*V[1].re-A[i][1].im*V[1].im+
                    A[i][2].re*V[2].re-A[i][2].im*V[2].im ;
                W[i].im = A[i][0].re*V[0].im+A[i][0].im*V[0].re+
                    A[i][1].re*V[1].im+A[i][1].im*V[1].re+
                    A[i][2].re*V[2].im+A[i][2].im*V[2].re ;
            }
        }

        /*
        *   copy complex 3x3 matrix
        *        A --> B
        */
        template<typename FLOAT_T>
        HOSTDEVICEQUALIFIER
        void copy_complex_matrix(ComplexNumber<FLOAT_T> A[][3], ComplexNumber<FLOAT_T> B[][3]){
            //memcpy(B,A,sizeof(ComplexNumber<FLOAT_T>)*9);

            for(int i = 0; i < 3; i++){
                for(int j = 0; j < 3; j++){
                    B[i][j].re = A[i][j].re;
                    B[i][j].im = A[i][j].im;
                }
            }
        }

        /*
        *   clear complex 3x3 matrix
        *
        */
        template<typename FLOAT_T>
        HOSTDEVICEQUALIFIER
        void clear_complex_matrix(ComplexNumber<FLOAT_T> A[][3]){
            A[0][0].re = 0;
            A[0][0].im = 0;
            A[0][1].re = 0;
            A[0][1].im = 0;
            A[0][2].re = 0;
            A[0][2].im = 0;
            A[1][0].re = 0;
            A[1][0].im = 0;
            A[1][1].re = 0;
            A[1][1].im = 0;
            A[1][2].re = 0;
            A[1][2].im = 0;
            A[2][0].re = 0;
            A[2][0].im = 0;
            A[2][1].re = 0;
            A[2][1].im = 0;
            A[2][2].re = 0;
            A[2][2].im = 0;
            //memset(A,0,sizeof(ComplexNumber<FLOAT_T>)*9);
        }

    }

}


#endif
