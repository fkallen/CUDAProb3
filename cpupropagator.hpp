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

#ifndef CUDAPROB3_CPUPROPAGATOR_HPP
#define CUDAPROB3_CPUPROPAGATOR_HPP

#include "constants.hpp"
#include "propagator.hpp"
#include "physics.hpp"

#include <omp.h>
#include <vector>


namespace cudaprob3{

    template<class FLOAT_T>
    class CpuPropagator : public Propagator<FLOAT_T>{
    public:
        CpuPropagator(int n_cosines_, int n_energies_, int threads) : Propagator<FLOAT_T>(n_cosines_, n_energies_){

            resultList.resize(std::uint64_t(n_cosines_) * std::uint64_t(n_energies_) * std::uint64_t(9));

            omp_set_num_threads(threads);
        }

        CpuPropagator(const CpuPropagator& other) : Propagator<FLOAT_T>(other){
            *this = other;
        }

        CpuPropagator(CpuPropagator&& other) : Propagator<FLOAT_T>(other){
            *this = std::move(other);
        }

        CpuPropagator& operator=(const CpuPropagator& other){
            Propagator<FLOAT_T>::operator=(other);

            resultList = other.resultList;

            return *this;
        }

        CpuPropagator& operator=(CpuPropagator&& other){
            Propagator<FLOAT_T>::operator=(std::move(other));

            resultList = std::move(other.resultList);

            return *this;
        }

    public:

        // calculate the probability of each cell
        void calculateProbabilities(NeutrinoType type) override{
            if(!this->isInit)
                throw std::runtime_error("CpuPropagator::calculateProbabilities. Object has been moved from.");
            if(!this->isSetProductionHeight)
                throw std::runtime_error("CpuPropagator::calculateProbabilities. production height was not set");

            physics::setMixMatrix_host(this->Mix_U.data());
            physics::setMassDifferences_host(this->dm.data());

            physics::calculate(type, this->cosineList.data(), this->cosineList.size(),
                this->energyList.data(), this->energyList.size(), this->radii.data(), this->rhos.data(), this->maxlayers.data(), this->ProductionHeightinCentimeter, resultList.data());
        }

        // get oscillation weight for specific cosine and energy
        FLOAT_T getProbability(int index_cosine, int index_energy, ProbType t) override{
            if(index_cosine >= this->n_cosines || index_energy >= this->n_energies)
                throw std::runtime_error("CpuPropagator::getProbability. Invalid indices");

            std::uint64_t index = std::uint64_t(index_cosine) * std::uint64_t(this->n_energies) * std::uint64_t(9)
                    + std::uint64_t(index_energy) * std::uint64_t(9);
            return resultList[index + int(t)];
        }

    private:
        std::vector<FLOAT_T> resultList;
    };



}

#endif
