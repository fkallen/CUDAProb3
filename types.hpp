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

#ifndef CUDAPROB3_TYPES_HPP
#define CUDAPROB3_TYPES_HPP

namespace cudaprob3{

    enum ProbType : int{
        e_e = 0, e_m = 1, e_t = 2,
        m_e = 3, m_m = 4, m_t = 5,
        t_e = 6, t_m = 7, t_t = 8
    };

    enum NeutrinoType {Neutrino, Antineutrino};
}


#endif
