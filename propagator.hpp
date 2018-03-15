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

#ifndef CUDAPROB3_PROPAGATOR_HPP
#define CUDAPROB3_PROPAGATOR_HPP

#include "constants.hpp"
#include "types.hpp"
#include "math.hpp"


#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <cmath>



namespace cudaprob3{


    /*
     * Base class which sets up input parameter on the host.
     * Concrete implementation of calcuations is provided in derived classes
     *
     *
     *
     * template parameter FLOAT_T is the floating point type to use for calculations, i.e float, double
     */
    template<class FLOAT_T>
    class Propagator{
    public:
        Propagator(int n_cosines_, int n_energies_) : n_cosines(n_cosines_), n_energies(n_energies_){
            energyList.resize(n_energies);
            cosineList.resize(n_cosines);
            maxlayers.resize(n_cosines);
        }

        Propagator(const Propagator& other){
            *this = other;
        }

        Propagator(Propagator&& other){
            *this = std::move(other);
        }

        Propagator& operator=(const Propagator& other){
            energyList = other.energyList;
            cosineList = other.cosineList;
            maxlayers = other.maxlayers;
            radii = other.radii;
            rhos = other.rhos;
            coslimit = other.coslimit;
            Mix_U = other.Mix_U;
            dm = other.dm;

            ProductionHeightinCentimeter = other.ProductionHeightinCentimeter;
            isSetCosine = other.isSetCosine;
            isSetProductionHeight = other.isSetProductionHeight;
            isInit = other.isInit;

            return *this;
        }

        Propagator& operator=(Propagator&& other){
            energyList = std::move(other.energyList);
            cosineList = std::move(other.cosineList);
            maxlayers = std::move(other.maxlayers);
            radii = std::move(other.radii);
            rhos = std::move(other.rhos);
            coslimit = std::move(other.coslimit);
            Mix_U = std::move(other.Mix_U);
            dm = std::move(other.dm);

            ProductionHeightinCentimeter = other.ProductionHeightinCentimeter;
            isSetCosine = other.isSetCosine;
            isSetProductionHeight = other.isSetProductionHeight;
            isInit = other.isInit;

            other.isInit = false;

            return *this;
        }


    public:
        // set density from arrays. radii_ and rhos_ must be same size. both radii_ and rhos_ must be sorted, in the same order.
        virtual void setDensity(const std::vector<FLOAT_T>& radii_, const std::vector<FLOAT_T>& rhos_){

            if(rhos_.size() != radii_.size()){
                throw std::runtime_error("setDensity : rhos.size() != radii.size()");
            }

            if(rhos_.size() == 0 || radii_.size() == 0){
                throw std::runtime_error("setDensity : vectors must not be empty");
            }

            bool needFlip = false;

            if(radii_.size() >= 2){
                int sign = (radii_[1] - radii_[0] > 0 ? 1 : -1);

                for(size_t i = 1; i < radii_.size(); i++){
                    if((radii_[i] - radii_[i-1]) * sign < 0)
                        throw std::runtime_error("radii order messed up");
                }

                if(sign == 1)
                    needFlip = true;
            }

            radii = radii_;
            rhos = rhos_;

            if(needFlip){
                std::reverse(radii.begin(), radii.end());
                std::reverse(rhos.begin(), rhos.end());
            }

            coslimit.clear();

            // first element of _Radii is largest radius!
            for(size_t i=0; i < radii.size() ; i++ )
            {
                // Using a cosine threshold instead! //
                FLOAT_T x = -1* sqrt( 1 - (radii[i] * radii[i] / ( Constants<FLOAT_T>::REarth()*Constants<FLOAT_T>::REarth())) );
                if ( i  == 0 ) x = 0;
                coslimit.push_back(x);
            }

            setMaxlayers();
        }

        // set density from file
        virtual void setDensityFromFile(const std::string& filename){
            std::ifstream file(filename);
            if(!file)
                throw std::runtime_error("could not open density file " + filename);

            std::vector<FLOAT_T> radii;
            std::vector<FLOAT_T> rhos;
            FLOAT_T r;
            FLOAT_T d;
            while (file >> r >> d){
                radii.push_back(r);
                rhos.push_back(d);
            }

            setDensity(radii, rhos);
        }

        // set mixing angles and cp phase. units are radians
        virtual void setMNSMatrix(FLOAT_T theta12, FLOAT_T theta13, FLOAT_T theta23, FLOAT_T dCP){

            const FLOAT_T s12 = sin(theta12);
            const FLOAT_T s13 = sin(theta13);
            const FLOAT_T s23 = sin(theta23);
            const FLOAT_T c12 = cos(theta12);
            const FLOAT_T c13 = cos(theta13);
            const FLOAT_T c23 = cos(theta23);

            const FLOAT_T sd  = sin(dCP);
            const FLOAT_T cd  = cos(dCP);

            U(0,0).re =  c12*c13;
            U(0,0).im =  0.0;
            U(0,1).re =  s12*c13;
            U(0,1).im =  0.0;
            U(0,2).re =  s13*cd;
            U(0,2).im = -s13*sd;
            U(1,0).re = -s12*c23-c12*s23*s13*cd;
            U(1,0).im =         -c12*s23*s13*sd;
            U(1,1).re =  c12*c23-s12*s23*s13*cd;
            U(1,1).im =         -s12*s23*s13*sd;
            U(1,2).re =  s23*c13;
            U(1,2).im =  0.0;
            U(2,0).re =  s12*s23-c12*c23*s13*cd;
            U(2,0).im =         -c12*c23*s13*sd;
            U(2,1).re = -c12*s23-s12*c23*s13*cd;
            U(2,1).im  =         -s12*c23*s13*sd;
            U(2,2).re =  c23*c13;
            U(2,2).im  =  0.0;
        }

        // set neutrino mass differences (m_i_j)^2 in electron volt. no assumptions about mass hierarchy are made
        virtual void setNeutrinoMasses(FLOAT_T dm12sq, FLOAT_T dm23sq){
            FLOAT_T mVac[3];

            mVac[0] = 0.0;
            mVac[1] = dm12sq;
            mVac[2] = dm12sq + dm23sq;

            const FLOAT_T delta = 5.0e-9;
            /* Break any degeneracies */
            if (dm12sq == 0.0) mVac[0] -= delta;
            if (dm23sq == 0.0) mVac[2] += delta;

            DM(0,0) = 0.0;
            DM(1,1) = 0.0;
            DM(2,2) = 0.0;
            DM(0,1) = mVac[0]-mVac[1];
            DM(1,0) = -DM(0,1);
            DM(0,2) = mVac[0]-mVac[2];
            DM(2,0) = -DM(0,2);
            DM(1,2) = mVac[1]-mVac[2];
            DM(2,1) = -DM(1,2);

        }

        // set the energy bins. energies are given in GeV
        virtual void setEnergyList(const std::vector<FLOAT_T>& list){
            if(list.size() != size_t(n_energies))
                throw std::runtime_error("Propagator::setEnergyList. Propagator was not created for this number of energy nodes");

            energyList = list;
        }

        // set cosine bins. cosines are given in radians
        virtual void setCosineList(const std::vector<FLOAT_T>& list){
            if(list.size() != size_t(n_cosines))
                throw std::runtime_error("Propagator::setCosineList. Propagator was not created for this number of cosine nodes");

            cosineList = list;

            if(isSetProductionHeight){
                setProductionHeight(ProductionHeightinCentimeter / 100000.0);
            }

            setMaxlayers();

            isSetCosine = true;
        }

        // set production height in km of neutrinos
        virtual void setProductionHeight(FLOAT_T heightKM){
            if(!isSetCosine)
                throw std::runtime_error("must set cosine list before production height");

            ProductionHeightinCentimeter = heightKM * 100000.0;

            isSetProductionHeight = true;
        }

        // calculate the probability of each cell
        virtual void calculateProbabilities(NeutrinoType type) = 0;

        // get oscillation weight for specific cosine and energy
        virtual FLOAT_T getProbability(int index_cosine, int index_energy, ProbType t) = 0;

    protected:
        virtual void setMaxlayers(){
            for(int index_cosine = 0; index_cosine < n_cosines; index_cosine++){
                FLOAT_T c = cosineList[index_cosine];
                const int maxLayer = std::count_if(coslimit.begin(), coslimit.end(), [c](FLOAT_T limit){ return c < limit;});
                maxlayers[index_cosine] = maxLayer;
            }
        }

        cudaprob3::math::ComplexNumber<FLOAT_T>& U(int i, int j){
            return Mix_U[( i * 3 + j)];
        }

        FLOAT_T& DM(int i, int j){
            return dm[( i * 3 + j)];
        }

        std::vector<FLOAT_T> energyList;
        std::vector<FLOAT_T> cosineList;
        std::vector<int> maxlayers;
        //std::vector<FLOAT_T> pathLengths;

        std::vector<FLOAT_T> radii;
        std::vector<FLOAT_T> rhos;
        std::vector<FLOAT_T> coslimit;

        std::array<cudaprob3::math::ComplexNumber<FLOAT_T>, 9> Mix_U; // MNS mixing matrix
        std::array<FLOAT_T, 9> dm; // mass differences;

        FLOAT_T ProductionHeightinCentimeter;

        bool isSetProductionHeight = false;
        bool isSetCosine = false;
        bool isInit = true;

        int n_cosines;
        int n_energies;
    };





} // namespace cudaprob3


#endif
