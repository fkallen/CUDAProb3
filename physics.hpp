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

#ifndef CUDAPROB3_PHYSICS_HPP
#define CUDAPROB3_PHYSICS_HPP

#include "constants.hpp"
#include "math.hpp"

#include <string.h>
#include <stdio.h>
//#include <math.h>
//#include <algorithm>
#include <assert.h>
#include <omp.h>


/*
 * This file contains the Barger et al physics which are used by Prob3++ to compute oscillation probabilities.
 *
 * Core function to loop over energys and cosine is function:
 *
 * template<typename FLOAT_T>
 * __host__ __device__
 * void calculate(NeutrinoType type, const FLOAT_T* const cosinelist, int n_cosines, const FLOAT_T* const energylist, int n_energies,
 *                       const FLOAT_T* const radii, const FLOAT_T* const rhos, const int* const maxlayers, FLOAT_T ProductionHeightinCentimeter, FLOAT_T* const result)
 *
 * It can either be called directly on the CPU, or on the GPU via kernel
 *
 * template<typename FLOAT_T>
 * __global__
 * void calculateKernel(NeutrinoType type, const FLOAT_T* const cosinelist, int n_cosines, const FLOAT_T* const energylist, int n_energies,
 *                       const FLOAT_T* const radii, const FLOAT_T* const rhos, const int* const maxlayers, FLOAT_T ProductionHeightinCentimeter, FLOAT_T* const result)
 *
 *
 * Both host and device code is combined in function void calculate(..), such that only one function has to be maintained for host and device.
 *
 *
 * Before using function void calculate(..) (or the kernel), neutrino mixing matrix and neutrino mass differences have to be set.
 * Use
 *
 * template<typename FLOAT_T>
 * void setMixMatrix(math::ComplexNumber<FLOAT_T>* U);
 *
 * and
 *
 * template<typename FLOAT_T>
 * void setMassDifferences(FLOAT_T* dm);
 *
 * before GPU calculation.
 *
 * Use
 *
 * template<typename FLOAT_T>
 * void setMixMatrix_host(math::ComplexNumber<FLOAT_T>* U);
 *
 * and
 *
 * template<typename FLOAT_T>
 * void setMassDifferences_host(FLOAT_T* dm);
 *
 * before CPU calculation.
 *
 *
 *
 *
 * NVCC macro __CUDA_ARCH__ is used for gpu exclusive code inside __host__ __device__ functions
 *
 */





// in device code, we need to access the device global constants instead of host global constants
#ifdef __CUDA_ARCH__

    #define U(i,j) ((math::ComplexNumber<FLOAT_T>*)mix_data_device)[( i * 3 + j)]
    #define DM(i,j) ((FLOAT_T*)mass_data_device)[( i * 3 + j)]
    #define AXFAC(a,b,c,d,e) ((FLOAT_T*)A_X_factor_device)[a * 3 * 3 * 3 * 4 + b * 3 * 3 * 4 + c * 3 * 4 + d * 4 + e]
    #define ORDER(i) mass_order_device[i]

#else

    #define U(i,j) ((math::ComplexNumber<FLOAT_T>*)mix_data)[( i * 3 + j)]
    #define DM(i,j) ((FLOAT_T*)mass_data)[( i * 3 + j)]
    #define AXFAC(a,b,c,d,e) ((FLOAT_T*)A_X_factor)[a * 3 * 3 * 3 * 4 + b * 3 * 3 * 4 + c * 3 * 4 + d * 4 + e]
    #define ORDER(i) mass_order[i]
#endif


namespace cudaprob3{

        namespace physics{

            /*
            * Constant global data
            */

            #ifdef __NVCC__
                __constant__ double mix_data_device [9 * sizeof(math::ComplexNumber<double>)] ;
                __constant__ double mass_data_device[9];
                __constant__ double A_X_factor_device[81 * 4]; //precomputed factors which only depend on the mixing matrix for faster calculation
                __constant__ int mass_order_device[3];
            #endif

            double mix_data [9 * sizeof(math::ComplexNumber<double>)] ;
            double mass_data[9];
            double A_X_factor[81 * 4]; //precomputed factors for faster calculation
            int mass_order[3];


            /*
             * Set global 3x3 pmns mixing matrix
             */
            template<typename FLOAT_T>
            void setMixMatrix(math::ComplexNumber<FLOAT_T>* U){
                memcpy((FLOAT_T*)mix_data, U, sizeof(math::ComplexNumber<FLOAT_T>) * 9);

                //precomputed factors for faster calculation
                for (int n=0; n<3; n++) {
                    for (int m=0; m<3; m++) {
                        for (int i=0; i<3; i++) {
                            for (int j=0; j<3; j++) {
                                AXFAC(n,m,i,j,0) = U[n * 3 + i].re * U[m * 3 + j].re + U[n * 3 + i].im * U[m * 3 + j].im;
                                AXFAC(n,m,i,j,1) = U[n * 3 + i].re * U[m * 3 + j].im - U[n * 3 + i].im * U[m * 3 + j].re;
                                AXFAC(n,m,i,j,2) = U[n * 3 + i].im * U[m * 3 + j].im + U[n * 3 + i].re * U[m * 3 + j].re;
                                AXFAC(n,m,i,j,3) = U[n * 3 + i].im * U[m * 3 + j].re - U[n * 3 + i].re * U[m * 3 + j].im;
                            }
                        }
                    }
                }
                #ifdef __NVCC__
                    //copy to constant memory on GPU
                    cudaMemcpyToSymbol(mix_data_device, U, sizeof(math::ComplexNumber<FLOAT_T>) * 9, 0, H2D); CUERR;
                    cudaMemcpyToSymbol(A_X_factor_device, A_X_factor, sizeof(FLOAT_T) * 81 * 4, 0, H2D); CUERR;
                #endif
            }

            /*
             * Set global 3x3 pmns mixing matrix on host only
             */
            template<typename FLOAT_T>
            void setMixMatrix_host(math::ComplexNumber<FLOAT_T>* U){
                memcpy((FLOAT_T*)mix_data, U, sizeof(math::ComplexNumber<FLOAT_T>) * 9);

                //precomputed factors for faster calculation
                for (int n=0; n<3; n++) {
                    for (int m=0; m<3; m++) {
                        for (int i=0; i<3; i++) {
                            for (int j=0; j<3; j++) {
                                AXFAC(n,m,i,j,0) = U[n * 3 + i].re * U[m * 3 + j].re + U[n * 3 + i].im * U[m * 3 + j].im;
                                AXFAC(n,m,i,j,1) = U[n * 3 + i].re * U[m * 3 + j].im - U[n * 3 + i].im * U[m * 3 + j].re;
                                AXFAC(n,m,i,j,2) = U[n * 3 + i].im * U[m * 3 + j].im + U[n * 3 + i].re * U[m * 3 + j].re;
                                AXFAC(n,m,i,j,3) = U[n * 3 + i].im * U[m * 3 + j].re - U[n * 3 + i].re * U[m * 3 + j].im;
                            }
                        }
                    }
                }
            }

            /*
             * Set global 3x3 neutrino mass difference matrix
             */
            /// \brief set mass differences to constant memory
            template<typename FLOAT_T>
            void setMassDifferences(FLOAT_T* dm){
                memcpy((FLOAT_T*)mass_data, dm, sizeof(FLOAT_T) * 9);
                #ifdef __NVCC__
                cudaMemcpyToSymbol(mass_data_device, dm, sizeof(FLOAT_T) * 9 , 0, cudaMemcpyHostToDevice); CUERR;
                #endif
            }

            /*
             * Set global 3x3 neutrino mass difference matrix on host only
             */
            template<typename FLOAT_T>
            void setMassDifferences_host(FLOAT_T* dm){
                memcpy((FLOAT_T*)mass_data, dm, sizeof(FLOAT_T) * 9);
            }


            //
            template<typename FLOAT_T>
            void prepare_getMfast(NeutrinoType type) {
                FLOAT_T alphaV, betaV, gammaV, argV, tmpV;
                FLOAT_T theta0V, theta1V, theta2V;
                FLOAT_T mMatV[3];

                /* The strategy to sort out the three roots is to compute the vacuum
                * mass the same way as the "matter" masses are computed then to sort
                * the results according to the input vacuum masses
                */
                alphaV = DM(0,1) + DM(0,2);

                betaV = DM(0,1) * DM(0,2);

                gammaV = 0.0;

                /* Compute the argument of the arc-cosine */
                tmpV = alphaV*alphaV-3.0*betaV;

                /* Equation (21) */
                argV = (2.0*alphaV*alphaV*alphaV-9.0*alphaV*betaV+27.0*gammaV)/
                    (2.0*sqrt(tmpV*tmpV*tmpV));
                if (fabs(argV)>1.0) argV = argV/fabs(argV);

                /* These are the three roots the paper refers to */
                theta0V = acos(argV)/3.0;
                theta1V = theta0V-(2.0*M_PI/3.0);
                theta2V = theta0V+(2.0*M_PI/3.0);

                mMatV[0] = mMatV[1] = mMatV[2] = -(2.0/3.0)*sqrt(tmpV);
                mMatV[0] *= cos(theta0V); mMatV[1] *= cos(theta1V); mMatV[2] *= cos(theta2V);
                tmpV = DM(0,0) - alphaV/3.0;
                mMatV[0] += tmpV; mMatV[1] += tmpV; mMatV[2] += tmpV;

                /* Sort according to which reproduce the vaccum eigenstates */
                int order[3];

                for (int i=0; i<3; i++) {
                    tmpV = fabs(DM(i,0)-mMatV[0]);
                    int k = 0;

                    for (int j=1; j<3; j++) {
                        FLOAT_T tmp = fabs(DM(i,0)-mMatV[j]);
                        if (tmp<tmpV) {
                            k = j;
                            tmpV = tmp;
                        }
                    }
                    order[i] = k;
                }
                memcpy(mass_order, order, sizeof(int) * 3);

                #ifdef __NVCC__
                cudaMemcpyToSymbol(mass_order_device, order, sizeof(int) * 3, 0, cudaMemcpyHostToDevice); CUERR;
                #endif
            }

           /*
            * Return induced neutrino mass difference matrix d_dmMatMat,
            * and d_dmMatVac, which is the mass difference matrix between induced masses and vacuum masses
            *
            * The strategy to sort out the three roots is to compute the vacuum
            * mass the same way as the "matter" masses are computed then to sort
            * the results according to the input vacuum masses. Subsequently, the "matter" masses
            * are calculated, using the found sorting for vacuum masses
            *
            * In the original implementation the order of vacuum masses is computed for each bin.
            * However, the ordering of vacuum masses does only depend on the constant neutrino mixing matrix.
            * Thus, the ordering can be precomputed, which is done in prepare_getMfast
            */
            template<typename FLOAT_T>
            HOSTDEVICEQUALIFIER
            void getMfast(const FLOAT_T Enu, const FLOAT_T rho,
                const NeutrinoType type,
                FLOAT_T d_dmMatMat[][3], FLOAT_T d_dmMatVac[][3]) {

                FLOAT_T mMatU[3], mMat[3];

                /* Equations (22) fro Barger et.al.*/
                const FLOAT_T fac = [&](){
                    if(type == Antineutrino)
                        return Constants<FLOAT_T>::tworttwoGf()*Enu*rho;
                    else
                        return -Constants<FLOAT_T>::tworttwoGf()*Enu*rho;
                }();

                const FLOAT_T alpha  = fac + DM(0,1) + DM(0,2);

                const FLOAT_T beta = DM(0,1)*DM(0,2) +
                    fac*(DM(0,1)*(1.0 -
                            U(0,1).re*U(0,1).re -
                            U(0,1).im*U(0,1).im ) +
                    DM(0,2)*(1.0-
                            U(0,2).re*U(0,2).re -
                            U(0,2).im*U(0,2).im));


                const FLOAT_T gamma = fac*DM(0,1)*DM(0,2)*(U(0,0).re * U(0,0).re + U(0,0).im * U(0,0).im);

                /* Compute the argument of the arc-cosine */
                const FLOAT_T tmp = alpha*alpha-3.0*beta < 0 ? 0 : alpha*alpha-3.0*beta;

                /* Equation (21) */
                const FLOAT_T argtmp = (2.0*alpha*alpha*alpha-9.0*alpha*beta+27.0*gamma)/
                    (2.0*sqrt(tmp*tmp*tmp));
                const FLOAT_T arg = [&](){
                    if (fabs(argtmp)>1.0)
                        return argtmp/fabs(argtmp);
                    else
                        return argtmp;
                }();

                /* These are the three roots the paper refers to */
                const FLOAT_T theta0 = acos(arg)/3.0;
                const FLOAT_T theta1 = theta0-(2.0*M_PI/3.0);
                const FLOAT_T theta2 = theta0+(2.0*M_PI/3.0);

                mMatU[0] = -(2.0/3.0)*sqrt(tmp);
                mMatU[1] = -(2.0/3.0)*sqrt(tmp);
                mMatU[2] = -(2.0/3.0)*sqrt(tmp);
                mMatU[0] *= cos(theta0);
                mMatU[1] *= cos(theta1);
                mMatU[2] *= cos(theta2);
                const FLOAT_T tmp2 = DM(0,0) - alpha/3.0;
                mMatU[0] += tmp2;
                mMatU[1] += tmp2;
                mMatU[2] += tmp2;

                /* Sort according to which reproduce the vaccum eigenstates */

                UNROLLQUALIFIER
                for (int i=0; i<3; i++) {
                    mMat[i] = mMatU[ORDER(i)];
                }

                UNROLLQUALIFIER
                for (int i=0; i<3; i++) {
                    UNROLLQUALIFIER
                    for (int j=0; j<3; j++) {
                        d_dmMatMat[i][j] = mMat[i] - mMat[j];
                        d_dmMatVac[i][j] = mMat[i] - DM(j,0);
                    }
                }
            }

            /*
                Calculate the product of Eq. (11)
            */
            template<typename FLOAT_T>
            HOSTDEVICEQUALIFIER
            void get_product(const FLOAT_T L, const FLOAT_T E, const FLOAT_T rho, const FLOAT_T d_dmMatVac[][3], const FLOAT_T d_dmMatMat[][3],
                const NeutrinoType type, math::ComplexNumber<FLOAT_T> product[][3][3]){

                math::ComplexNumber<FLOAT_T> twoEHmM[3][3][3];

                const FLOAT_T fac = [&](){
                    if(type == Antineutrino)
                        return Constants<FLOAT_T>::tworttwoGf()*E*rho;
                    else
                        return -Constants<FLOAT_T>::tworttwoGf()*E*rho;
                }();

                /* Calculate the matrix 2EH-M_j */
                UNROLLQUALIFIER
                for (int n=0; n<3; n++) {
                    UNROLLQUALIFIER
                    for (int m=0; m<3; m++) {
                        twoEHmM[n][m][0].re = -fac*(U(0,n).re*U(0,m).re+U(0,n).im*U(0,m).im);
                        twoEHmM[n][m][0].im = -fac*(U(0,n).re*U(0,m).im-U(0,n).im*U(0,m).re);
                        twoEHmM[n][m][1].re = -fac*(U(0,n).re*U(0,m).re+U(0,n).im*U(0,m).im);
                        twoEHmM[n][m][1].im = -fac*(U(0,n).re*U(0,m).im-U(0,n).im*U(0,m).re);
                        twoEHmM[n][m][2].re = -fac*(U(0,n).re*U(0,m).re+U(0,n).im*U(0,m).im);
                        twoEHmM[n][m][2].im = -fac*(U(0,n).re*U(0,m).im-U(0,n).im*U(0,m).re);
                    }
                }

                UNROLLQUALIFIER
                for (int j=0; j<3; j++){
                    twoEHmM[0][0][j].re-= d_dmMatVac[j][0];
                    twoEHmM[1][1][j].re-= d_dmMatVac[j][1];
                    twoEHmM[2][2][j].re-= d_dmMatVac[j][2];
                }

                /* Calculate the product in eq.(11) of twoEHmM for j!=k */
                //memset(product, 0, 3*3*3*sizeof(math::ComplexNumber<FLOAT_T>));
                UNROLLQUALIFIER
                for (int i=0; i<3; i++) {
                    UNROLLQUALIFIER
                    for (int j=0; j<3; j++) {
                        UNROLLQUALIFIER
                        for (int k=0; k<3; k++) {
                            product[i][j][k].re = 0;
                            product[i][j][k].im = 0;
                        }
                    }
                }

                UNROLLQUALIFIER
                for (int i=0; i<3; i++) {
                    UNROLLQUALIFIER
                    for (int j=0; j<3; j++) {
                        UNROLLQUALIFIER
                        for (int k=0; k<3; k++) {
                            product[i][j][0].re +=
                                twoEHmM[i][k][1].re*twoEHmM[k][j][2].re -
                                twoEHmM[i][k][1].im*twoEHmM[k][j][2].im;
                            product[i][j][0].im +=
                                twoEHmM[i][k][1].re*twoEHmM[k][j][2].im +
                                twoEHmM[i][k][1].im*twoEHmM[k][j][2].re;

                            product[i][j][1].re +=
                                twoEHmM[i][k][2].re*twoEHmM[k][j][0].re -
                                twoEHmM[i][k][2].im*twoEHmM[k][j][0].im;
                            product[i][j][1].im +=
                                twoEHmM[i][k][2].re*twoEHmM[k][j][0].im +
                                twoEHmM[i][k][2].im*twoEHmM[k][j][0].re;

                            product[i][j][2].re +=
                                twoEHmM[i][k][0].re*twoEHmM[k][j][1].re -
                                twoEHmM[i][k][0].im*twoEHmM[k][j][1].im;
                            product[i][j][2].im +=
                                twoEHmM[i][k][0].re*twoEHmM[k][j][1].im +
                                twoEHmM[i][k][0].im*twoEHmM[k][j][1].re;
                        }

                        product[i][j][0].re /= (d_dmMatMat[0][1]*d_dmMatMat[0][2]);
                        product[i][j][0].im /= (d_dmMatMat[0][1]*d_dmMatMat[0][2]);
                        product[i][j][1].re /= (d_dmMatMat[1][2]*d_dmMatMat[1][0]);
                        product[i][j][1].im /= (d_dmMatMat[1][2]*d_dmMatMat[1][0]);
                        product[i][j][2].re /= (d_dmMatMat[2][0]*d_dmMatMat[2][1]);
                        product[i][j][2].im /= (d_dmMatMat[2][0]*d_dmMatMat[2][1]);
                    }
                }
            }


            template<typename FLOAT_T>
            HOSTDEVICEQUALIFIER
            void getA(const FLOAT_T L, const FLOAT_T E, const FLOAT_T rho, const FLOAT_T d_dmMatVac[][3], const FLOAT_T d_dmMatMat[][3],
                const NeutrinoType type, math::ComplexNumber<FLOAT_T> A[3][3], const FLOAT_T phase_offset){

                math::ComplexNumber<FLOAT_T> X[3][3];
                math::ComplexNumber<FLOAT_T> product[3][3][3];
                /* (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2-km) */
                const FLOAT_T LoEfac = 2.534;

                if (phase_offset == 0.0) {
                    get_product(L, E, rho, d_dmMatVac, d_dmMatMat, type, product);
                }


                /* Make the sum with the exponential factor in Eq. (11) */
                //memset(X, 0, 3*3*sizeof(math::ComplexNumber<FLOAT_T>));
                UNROLLQUALIFIER
                for (int i=0; i<3; i++) {
                    UNROLLQUALIFIER
                    for (int j=0; j<3; j++) {
                        X[i][j].re = 0;
                        X[i][j].im = 0;
                    }
                }

                UNROLLQUALIFIER
                for (int k=0; k<3; k++) {
                    const FLOAT_T arg = [&](){
                        if( k == 2)
                            return -LoEfac * d_dmMatVac[k][0] * L/E + phase_offset;
                        else
                            return -LoEfac * d_dmMatVac[k][0] * L/E;
                    }();

#ifdef __CUDACC__
                    FLOAT_T c,s;
                    sincos(arg, &s, &c);
#else
                    const FLOAT_T s = sin(arg);
                    const FLOAT_T c = cos(arg);
#endif
                    UNROLLQUALIFIER
                    for (int i=0; i<3; i++) {
                        UNROLLQUALIFIER
                        for (int j=0; j<3; j++) {
                            X[i][j].re += c*product[i][j][k].re - s*product[i][j][k].im;
                            X[i][j].im += c*product[i][j][k].im + s*product[i][j][k].re;
                        }
                    }
                }

                /* Eq. (10)*/
                //memset(A, 0, 3*3*2*sizeof(FLOAT_T));

                UNROLLQUALIFIER
                for (int n=0; n<3; n++) {
                    UNROLLQUALIFIER
                    for (int m=0; m<3; m++) {
                        A[n][m].re = 0;
                        A[n][m].im = 0;
                    }
                }

                UNROLLQUALIFIER
                for (int n=0; n<3; n++) {
                    UNROLLQUALIFIER
                    for (int m=0; m<3; m++) {
                        UNROLLQUALIFIER
                        for (int i=0; i<3; i++) {
                            UNROLLQUALIFIER
                            for (int j=0; j<3; j++) {
                                // use precomputed factors
                                A[n][m].re +=
                                    AXFAC(n,m,i,j,0) * X[i][j].re +
                                    AXFAC(n,m,i,j,1) * X[i][j].im;
                                A[n][m].im +=
                                    AXFAC(n,m,i,j,2) * X[i][j].im +
                                    AXFAC(n,m,i,j,3) * X[i][j].re;
                            }
                        }
                    }
                }
            }

            /*
             * Get 3x3 transition amplitude Aout for neutrino with energy E travelling Len kilometers through matter of constant density rho
             */
            template<typename FLOAT_T>
            HOSTDEVICEQUALIFIER
            void get_transition_matrix(const NeutrinoType type, const FLOAT_T Enu, const FLOAT_T rho, const FLOAT_T Len,
                                        math::ComplexNumber<FLOAT_T> Aout[][3], const FLOAT_T phase_offset){

                FLOAT_T d_dmMatVac[3][3], d_dmMatMat[3][3];

                getMfast(Enu, rho, type, d_dmMatMat, d_dmMatVac);
                getA(Len, Enu, rho, d_dmMatVac, d_dmMatMat, type, Aout,phase_offset);
            }

            /*
                Find density in layer
            */
            template<typename FLOAT_T>
            HOSTDEVICEQUALIFIER
            FLOAT_T getDensityOfLayer(const FLOAT_T* const rhos, int layer, int max_layer){
                if(layer == 0) return 0.0;
                int i;
                if(layer <= max_layer){
                    i = layer-1;
                }else{
                    i = 2 * max_layer - layer - 1;
                }

                return rhos[i];
            }

            /*
                Find distance in layer
            */
            template<typename FLOAT_T>
            HOSTDEVICEQUALIFIER
            FLOAT_T getTraversedDistanceOfLayer(const FLOAT_T* const radii,
                                                int layer,
                                                int max_layer,
                                                FLOAT_T PathLength,
                                                FLOAT_T TotalEarthLength,
                                                FLOAT_T cosine_zenith){

                if(cosine_zenith >= 0) return PathLength;
                if(layer == 0) return PathLength - TotalEarthLength;

                int i;
                if(layer >= max_layer)
                    i = -layer - 1 + 2 * max_layer;
                else{
                    i = layer-1;
                }

                const FLOAT_T CrossThis = 2.0*sqrt( radii[i] * radii[i]  - (Constants<FLOAT_T>::REarth())*(Constants<FLOAT_T>::REarth())*( 1 - cosine_zenith*cosine_zenith ) );
                const FLOAT_T CrossNext = 2.0*sqrt( radii[i+1] * radii[i+1] - (Constants<FLOAT_T>::REarth())*((FLOAT_T)Constants<FLOAT_T>::REarth())*( 1 -cosine_zenith*cosine_zenith ) );

                if(i < max_layer - 1){
                    return 0.5*( CrossThis-CrossNext )*(Constants<FLOAT_T>::km2cm());
                }else{
                    return CrossThis*(Constants<FLOAT_T>::km2cm());
                }
            }


            template<typename FLOAT_T>
            HOSTDEVICEQUALIFIER
            void calculate(NeutrinoType type,
                            const FLOAT_T* const cosinelist,
                            int n_cosines,
                            const FLOAT_T* const energylist,
                            int n_energies,
                            const FLOAT_T* const radii,
                            const FLOAT_T* const rhos,
                            const int* const maxlayers,
                            FLOAT_T ProductionHeightinCentimeter,
                            FLOAT_T* const result){

            //prepare more constant data. For the kernel, this is done by the wrapper function callCalculateKernelAsync
            #ifndef __CUDA_ARCH__
                prepare_getMfast<FLOAT_T>(type);
            #endif

            #ifdef __CUDA_ARCH__
                // on the device, we use the global thread Id to index the data
                const int max_energies_per_path = SDIV(n_energies, blockDim.x) * blockDim.x;
                for(unsigned index = blockIdx.x * blockDim.x + threadIdx.x; index < n_cosines * max_energies_per_path; index += blockDim.x * gridDim.x){
                    const unsigned index_energy = index % max_energies_per_path;
                    const unsigned index_cosine = index / max_energies_per_path;
            #else
                // on the host, we use OpenMP to parallelize looping over cosines
                #pragma omp parallel for schedule(dynamic)
                for(int index_cosine = 0; index_cosine < n_cosines; index_cosine += 1){
            #endif

                    const FLOAT_T cosine_zenith = cosinelist[index_cosine];

                    const FLOAT_T PathLength = sqrt((Constants<FLOAT_T>::REarthcm() + ProductionHeightinCentimeter )*(Constants<FLOAT_T>::REarthcm() + ProductionHeightinCentimeter)
                                                - (Constants<FLOAT_T>::REarthcm()*Constants<FLOAT_T>::REarthcm())*( 1 - cosine_zenith*cosine_zenith)) - Constants<FLOAT_T>::REarthcm()*cosine_zenith;

                    const FLOAT_T TotalEarthLength =  -2.0*cosine_zenith*Constants<FLOAT_T>::REarthcm(); // in [cm]
                    const int MaxLayer = maxlayers[index_cosine];

                    math::ComplexNumber<FLOAT_T> TransitionMatrix[3][3];
                    math::ComplexNumber<FLOAT_T> TransitionMatrixCoreToMantle[3][3];
                    math::ComplexNumber<FLOAT_T> finalTransitionMatrix[3][3];
                    math::ComplexNumber<FLOAT_T> TransitionTemp[3][3];

                #ifndef __CUDA_ARCH__
                    for(int index_energy = 0; index_energy < n_energies; index_energy += 1){
                #else
                    if(index_energy < n_energies){
                #endif

                        const FLOAT_T energy = energylist[index_energy];

                        // set TransitionMatrixCoreToMantle to unit matrix
                        UNROLLQUALIFIER
                        for(int i = 0; i < 3; i++){
                            UNROLLQUALIFIER
                            for(int j = 0; j < 3; j++){
                                    TransitionMatrixCoreToMantle[i][j].re = (i == j ? 1.0 : 0.0);
                                    TransitionMatrixCoreToMantle[i][j].im = 0.0;
                            }
                        }

                        // loop from vacuum layer to innermost crossed layer
                        for (int i = 0; i <= MaxLayer ; i++ ){
                            const FLOAT_T distance = getTraversedDistanceOfLayer(radii, i, MaxLayer, PathLength, TotalEarthLength, cosine_zenith);
                            const FLOAT_T density = getDensityOfLayer(rhos, i, MaxLayer);

                            get_transition_matrix( type,
                                                    energy	,		   // in GeV
                                                    density  * Constants<FLOAT_T>::density_convert(),
                                                    distance / Constants<FLOAT_T>::km2cm(),
                                                    TransitionMatrix,			   // Output transition matrix
                                                    FLOAT_T(0.0)  					   // phase offset
                                                    );

                            if (i == 0){    // atmosphere
                                copy_complex_matrix( TransitionMatrix , finalTransitionMatrix );
                            }else if(i < MaxLayer){ // not the innermost layer, can reuse current TransitionMatrix
                                clear_complex_matrix( TransitionTemp );
                                multiply_complex_matrix( TransitionMatrix, finalTransitionMatrix, TransitionTemp );
                                copy_complex_matrix( TransitionTemp, finalTransitionMatrix );

                                clear_complex_matrix( TransitionTemp );
                                multiply_complex_matrix( TransitionMatrixCoreToMantle, TransitionMatrix, TransitionTemp );
                                copy_complex_matrix( TransitionTemp, TransitionMatrixCoreToMantle );
                            }else{ // innermost layer
                                clear_complex_matrix( TransitionTemp );
                                multiply_complex_matrix( TransitionMatrix, finalTransitionMatrix, TransitionTemp );
                                copy_complex_matrix( TransitionTemp, finalTransitionMatrix );
                            }
                        }

                        // calculate final transition matrix
                        clear_complex_matrix( TransitionTemp );
                        multiply_complex_matrix( TransitionMatrixCoreToMantle, finalTransitionMatrix, TransitionTemp );
                        copy_complex_matrix( TransitionTemp, finalTransitionMatrix );

                        // for oscillation probabilities where the initial wave function
                        // evaluates to 0+0i for two flavors and evaluates to 1+0i for the remaining third flavor,
                        // we don't need to perform full matrix vector multiplication
                        UNROLLQUALIFIER
                        for (int inflv = 0 ; inflv < 3 ; inflv++ ){
                            UNROLLQUALIFIER
                            for (int outflv = 0 ; outflv < 3 ; outflv++ ){
                                const FLOAT_T re = finalTransitionMatrix[outflv][inflv].re;
                                const FLOAT_T im = finalTransitionMatrix[outflv][inflv].im;

                            #ifdef __CUDA_ARCH__
                                const unsigned long long resultIndex = (unsigned long long)(n_energies) * (unsigned long long)(index_cosine) + (unsigned long long)(index_energy);
                                result[resultIndex + (unsigned long long)(n_energies) * (unsigned long long)(n_cosines) * (unsigned long long)((inflv * 3 + outflv))] = re * re + im * im;
                            #else
                                const unsigned long long resultIndex = (unsigned long long)(index_cosine) * (unsigned long long)(n_energies) * (unsigned long long)(9)
                                                    + (unsigned long long)(index_energy) * (unsigned long long)(9);
                                result[resultIndex + (unsigned long long)((inflv * 3 + outflv))] = re * re + im * im;
                            #endif

                            }
                        }
                    }
                }
            }


            #ifdef __NVCC__
            template<typename FLOAT_T>
            KERNEL
            __launch_bounds__( 64, 8 )
            void calculateKernel(NeutrinoType type,
                                const FLOAT_T* const cosinelist,
                                int n_cosines,
                                const FLOAT_T* const energylist,
                                int n_energies,
                                const FLOAT_T* const radii,
                                const FLOAT_T* const rhos,
                                const int* const maxlayers,
                                FLOAT_T ProductionHeightinCentimeter,
                                FLOAT_T* const result){

                calculate(type, cosinelist, n_cosines, energylist, n_energies, radii, rhos, maxlayers, ProductionHeightinCentimeter, result);
            }

            template<typename FLOAT_T>
            void callCalculateKernelAsync(dim3 grid,
                                        dim3 block,
                                        cudaStream_t stream,
                                        NeutrinoType type,
                                        const FLOAT_T* const cosinelist,
                                        int n_cosines,
                                        const FLOAT_T* const energylist,
                                        int n_energies,
                                        const FLOAT_T* const radii,
                                        const FLOAT_T* const rhos,
                                        const int* const maxlayers,
                                        FLOAT_T ProductionHeightinCentimeter,
                                        FLOAT_T* const result){

                prepare_getMfast<FLOAT_T>(type);

                calculateKernel<FLOAT_T><<<grid, block, 0, stream>>>(type, cosinelist, n_cosines, energylist, n_energies, radii, rhos, maxlayers, ProductionHeightinCentimeter, result);
                CUERR;
            }
            #endif

        } // namespace physics

} // namespace cudaprob3





#endif
