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

#ifndef MY_HPC_HELPERS_CUH
#define MY_HPC_HELPERS_CUH

#include <iostream>
#include <cstdint>


#include <chrono>



#define TIMERSTARTCPU(label)                                                  \
    std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
    a##label = std::chrono::system_clock::now();

#define TIMERSTOPCPU(label)                                                   \
    b##label = std::chrono::system_clock::now();                           \
    std::chrono::duration<double> delta##label = b##label-a##label;        \
    std::cout << "# elapsed time ("<< #label <<"): "                       \
                << delta##label.count()  << " s" << std::endl;



#ifndef __CUDACC__
    #define TIMERSTART(label)                                                  \
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
        a##label = std::chrono::system_clock::now();
#else
    #define TIMERSTART(label)                                                  \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);
#endif

#ifndef __CUDACC__
    #define TIMERSTOP(label)                                                   \
        b##label = std::chrono::system_clock::now();                           \
        std::chrono::duration<double> delta##label = b##label-a##label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta##label.count()  << " s" << std::endl;
#else
    #define TIMERSTOP(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms (" << #label << ")" \
                      << std::endl;
#endif


#ifdef __CUDACC__
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }
#else
	#define CUERR

#endif

#ifdef __CUDACC__
    // transfer constants
    #define H2D (cudaMemcpyHostToDevice)
    #define D2H (cudaMemcpyDeviceToHost)
    #define H2H (cudaMemcpyHostToHost)
    #define D2D (cudaMemcpyDeviceToDevice)
#endif

#ifdef __CUDACC__
    #define HOSTQUALIFIER __host__
    #define DEVICEQUALIFIER __device__
    #define HOSTDEVICEQUALIFIER __host__ __device__
    #define KERNEL __global__
#else
    #define HOSTQUALIFIER
    #define DEVICEQUALIFIER
    #define HOSTDEVICEQUALIFIER
    #define KERNEL
#endif


#ifdef __CUDA_ARCH__
    #define UNROLLQUALIFIER #pragma unroll
#else
    #define UNROLLQUALIFIER
#endif

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

#endif
