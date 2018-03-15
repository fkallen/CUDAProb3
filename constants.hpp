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

#ifndef CUDAPROB3_CONSTANTS_HPP
#define CUDAPROB3_CONSTANTS_HPP

#include "hpc_helpers.cuh"

namespace cudaprob3{

    template<typename FLOAT_T>
    struct Constants{
        HOSTDEVICEQUALIFIER
        static constexpr FLOAT_T tworttwoGf(){ return 1.52588e-4; }

        HOSTDEVICEQUALIFIER
        static constexpr FLOAT_T km2cm(){ return 1.0e5; }

        HOSTDEVICEQUALIFIER
        static constexpr FLOAT_T REarth(){ return 6371.0; }

        HOSTDEVICEQUALIFIER
        static constexpr FLOAT_T REarthcm(){ return REarth() * km2cm(); }

        HOSTDEVICEQUALIFIER
        static constexpr FLOAT_T density_convert(){ return 0.5; }
    };

}


#endif
