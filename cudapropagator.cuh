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

#ifdef __NVCC__

#ifndef CUDAPROB3_CUDAPROPAGATOR_HPP
#define CUDAPROB3_CUDAPROPAGATOR_HPP

#include "constants.hpp"
#include "propagator.hpp"
#include "physics.hpp"
#include "cuda_unique.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>


namespace cudaprob3{



    template<class FLOAT_T>
    class CudaPropagatorSingle : public Propagator<FLOAT_T>{
        template<typename>
        friend class CudaPropagator;
    public:

        CudaPropagatorSingle(int id, int n_cosines_, int n_energies_) : Propagator<FLOAT_T>(n_cosines_, n_energies_), deviceId(id){

            int nDevices;

            cudaGetDeviceCount(&nDevices);

            if(nDevices == 0) throw std::runtime_error("No GPU found");
            if(id >= nDevices){
                std::cout << "Available GPUs:" << std::endl;
                for(int j = 0; j < nDevices; j++){
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, j);
                    std::cout << "Id " << j << " : " << prop.name << std::endl;
                }
                throw std::runtime_error("The requested GPU Id " + std::to_string(id) + " is not available.");
            }

            cudaSetDevice(id); CUERR;
            cudaFree(0);

            cudaStreamCreate(&stream); CUERR;

            // resize data structures
            resultList = make_unique_pinned<FLOAT_T>(std::uint64_t(n_cosines_) * std::uint64_t(n_energies_) * std::uint64_t(9));


            d_energy_list = make_unique_dev<FLOAT_T>(deviceId, n_energies_); CUERR;
            d_cosine_list = make_unique_dev<FLOAT_T>(deviceId, n_cosines_); CUERR;
            d_result_list = make_shared_dev<FLOAT_T>(deviceId, std::uint64_t(n_cosines_) * std::uint64_t(n_energies_) * std::uint64_t(9)); CUERR;
            d_maxlayers = make_unique_dev<int>(deviceId, this->n_cosines);
        }


        CudaPropagatorSingle(int n_cosines_, int n_energies_) : CudaPropagatorSingle(0, n_cosines_, n_energies_){

        }

        ~CudaPropagatorSingle(){
            cudaSetDevice(deviceId);
            cudaStreamDestroy(stream);
        }

        CudaPropagatorSingle(const CudaPropagatorSingle& other) = delete;

        CudaPropagatorSingle(CudaPropagatorSingle&& other) : Propagator<FLOAT_T>(other){
            *this = std::move(other);

            cudaSetDevice(deviceId);
            cudaStreamCreate(&stream); CUERR;
        }

        CudaPropagatorSingle& operator=(const CudaPropagatorSingle& other) = delete;

        CudaPropagatorSingle& operator=(CudaPropagatorSingle&& other){
            Propagator<FLOAT_T>::operator=(std::move(other));

            resultList = std::move(other.resultList);
            d_rhos = std::move(other.d_rhos);
            d_radii = std::move(other.d_radii);
            d_maxlayers = std::move(other.d_maxlayers);
            d_energy_list = std::move(other.d_energy_list);
            d_cosine_list = std::move(other.d_cosine_list);
            d_result_list = std::move(other.d_result_list);

            deviceId = other.deviceId;
            resultsResideOnHost = other.resultsResideOnHost;

            return *this;
        }

    public:

        void setDensity(const std::vector<FLOAT_T>& radii_, const std::vector<FLOAT_T>& rhos_) override{
            // call parent function to set up host density data
            Propagator<FLOAT_T>::setDensity(radii_, rhos_);

            // copy host density data to device density data
            cudaSetDevice(deviceId); CUERR;

            int nDensityLayers = this->radii.size();

            d_rhos = make_unique_dev<FLOAT_T>(deviceId, 2 * nDensityLayers + 1);
            d_radii = make_unique_dev<FLOAT_T>(deviceId, 2 * nDensityLayers + 1);

            cudaMemcpy(d_rhos.get(), this->rhos.data(), sizeof(FLOAT_T) * nDensityLayers, H2D); CUERR;
            cudaMemcpy(d_radii.get(), this->radii.data(), sizeof(FLOAT_T) * nDensityLayers, H2D); CUERR;
        }

        void setEnergyList(const std::vector<FLOAT_T>& list) override{
            Propagator<FLOAT_T>::setEnergyList(list);

            cudaMemcpy(d_energy_list.get(), this->energyList.data(), sizeof(FLOAT_T) * this->n_energies, H2D); CUERR;
        }

        void setCosineList(const std::vector<FLOAT_T>& list) override{
            Propagator<FLOAT_T>::setCosineList(list);

            cudaMemcpy(d_cosine_list.get(), this->cosineList.data(), sizeof(FLOAT_T) * this->n_cosines, H2D); CUERR;
        }

        // calculate the probability of each cell
        void calculateProbabilities(NeutrinoType type) override{
            calculateProbabilitiesAsync(type);
            waitForCompletion();
        }

        // get oscillation weight for specific cosine and energy
        FLOAT_T getProbability(int index_cosine, int index_energy, ProbType t) override{
            if(index_cosine >= this->n_cosines || index_energy >= this->n_energies)
                throw std::runtime_error("CudaPropagatorSingle::getProbability. Invalid indices");

            if(!resultsResideOnHost){
                getResultFromDevice();
                resultsResideOnHost = true;
            }

            const std::uint64_t index = std::uint64_t(index_cosine) * std::uint64_t(this->n_energies) + std::uint64_t(index_energy);
            const std::uint64_t offset = std::uint64_t(t) * std::uint64_t(this->n_energies) * std::uint64_t(this->n_cosines);

            return resultList.get()[index + offset];
        }

    protected:
        void setMaxlayers() override{
            Propagator<FLOAT_T>::setMaxlayers();

            cudaMemcpy(d_maxlayers.get(), this->maxlayers.data(), sizeof(int) * this->n_cosines, H2D); CUERR;
        }

        void calculateProbabilitiesAsync(NeutrinoType type){
            if(!this->isInit)
                throw std::runtime_error("CudaPropagatorSingle::calculateProbabilities. Object has been moved from.");
            if(!this->isSetProductionHeight)
                throw std::runtime_error("CudaPropagatorSingle::calculateProbabilities. production height was not set");

            resultsResideOnHost = false;
            cudaSetDevice(deviceId); CUERR;

            physics::setMixMatrix(this->Mix_U.data());
            physics::setMassDifferences(this->dm.data());

            dim3 block(64, 1, 1);

            //const unsigned blocks = SDIV(this->energyList.size() * this->cosineList.size(), block.x);
            const unsigned blocks = SDIV(this->energyList.size(), block.x) * this->cosineList.size();

            dim3 grid(blocks, 1, 1);

            physics::callCalculateKernelAsync(grid, block, stream,
                            type,
                            d_cosine_list.get(), this->n_cosines,
                            d_energy_list.get(), this->n_energies,
                            d_radii.get(), d_rhos.get(),
                            d_maxlayers.get(),
                            this->ProductionHeightinCentimeter, d_result_list.get());

            CUERR;
        }

        // wait for calculateProbabilitiesAsync to finish
        void waitForCompletion(){
            cudaSetDevice(deviceId); CUERR;
            cudaStreamSynchronize(stream); CUERR;
        }

        void getResultFromDevice(){
            cudaSetDevice(deviceId); CUERR;
            cudaMemcpyAsync(resultList.get(), d_result_list.get(),
                            sizeof(FLOAT_T) * std::uint64_t(9) * std::uint64_t(this->n_energies) * std::uint64_t(this->n_cosines),
                            D2H, stream);  CUERR;
            cudaStreamSynchronize(stream);
        }

    private:
        unique_pinned_ptr<FLOAT_T> resultList;

        unique_dev_ptr<FLOAT_T> d_rhos;
        unique_dev_ptr<FLOAT_T> d_radii;
        unique_dev_ptr<int> d_maxlayers;
        unique_dev_ptr<FLOAT_T> d_energy_list;
        unique_dev_ptr<FLOAT_T> d_cosine_list;
        shared_dev_ptr<FLOAT_T> d_result_list;

        cudaStream_t stream;
        int deviceId;

        bool resultsResideOnHost = false;
    };



    /*
    * Wrapper class for multiple GPUs
    *
    *
    */

    template<class FLOAT_T>
    class CudaPropagator : public Propagator<FLOAT_T>{
    public:
        CudaPropagator(int nc, int ne) : CudaPropagator(std::vector<int>{0}, nc, ne, true){}

        CudaPropagator(const std::vector<int>& ids, int nc, int ne, bool failOnInvalidId = true) : Propagator<FLOAT_T>(nc, ne) {

            int nDevices;
            cudaGetDeviceCount(&nDevices);

            if(nDevices == 0) throw std::runtime_error("No GPU found");

            for(const auto& id: ids){
                if(id >= nDevices){
                    if(failOnInvalidId){
                        std::cout << "Available GPUs:" << std::endl;
                        for(int j = 0; j < nDevices; j++){
                            cudaDeviceProp prop;
                            cudaGetDeviceProperties(&prop, j);
                            std::cout << "Id " << j << " : " << prop.name << std::endl;
                        }
                        throw std::runtime_error("The requested GPU Id " + std::to_string(id) + " is not available.");
                    }else{
                        std::cout << "invalid device id found : " << id << std::endl;
                    }
                }else{
                    deviceIds.push_back(id);
                }
            }

            if(deviceIds.size() == 0){
                throw std::runtime_error("No valid device id found.");
            }
            cosineIndices.resize(deviceIds.size());
            localCosineIndices.resize(this->n_cosines);

            for(int icos = 0; icos < this->n_cosines; icos++){

                int deviceIndex = getCosineDeviceIndex(icos);

                cosineIndices[deviceIndex].push_back(icos);
                localCosineIndices[icos] = cosineIndices[deviceIndex].size() - 1;
            }

            for(size_t i = 0; i < deviceIds.size() && i < size_t(this->n_cosines); i++){
                propagatorVector.push_back(std::unique_ptr<CudaPropagatorSingle<FLOAT_T>>(new CudaPropagatorSingle<FLOAT_T>(deviceIds[i], cosineIndices[i].size(), this->n_energies)));
            }
        }

        CudaPropagator(const CudaPropagator& other) = delete;

        CudaPropagator(CudaPropagator&& other) : Propagator<FLOAT_T>(other){
            *this = std::move(other);
        }

        CudaPropagator& operator=(const CudaPropagator& other) = delete;

        CudaPropagator& operator=(CudaPropagator&& other){
            Propagator<FLOAT_T>::operator=(std::move(other));

            deviceIds = std::move(other.deviceIds);
            cosineIndices = std::move(other.cosineIndices);
            localCosineIndices = std::move(other.localCosineIndices);
            cosineBatches = std::move(other.cosineBatches);
            propagatorVector = std::move(other.propagatorVector);

            return *this;
        }

    public:

        void setDensityFromFile(const std::string& filename) override{
            Propagator<FLOAT_T>::setDensityFromFile(filename);

            for(auto& propagator : propagatorVector)
                propagator->setDensityFromFile(filename);
        }

        void setDensity(const std::vector<FLOAT_T>& radii, const std::vector<FLOAT_T>& rhos) override{
            Propagator<FLOAT_T>::setDensity(radii, rhos);

            for(auto& propagator : propagatorVector)
                propagator->setDensity(radii, rhos);
        }

        void setNeutrinoMasses(FLOAT_T dm12sq, FLOAT_T dm23sq) override{
            Propagator<FLOAT_T>::setNeutrinoMasses(dm12sq, dm23sq);

            for(auto& propagator : propagatorVector)
                propagator->setNeutrinoMasses(dm12sq, dm23sq);
        }

        void setMNSMatrix(FLOAT_T theta12, FLOAT_T theta13, FLOAT_T theta23, FLOAT_T dCP) override{
            Propagator<FLOAT_T>::setMNSMatrix(theta12, theta13, theta23, dCP);

            for(auto& propagator : propagatorVector)
                propagator->setMNSMatrix(theta12, theta13, theta23, dCP);
        }

        void setEnergyList(const std::vector<FLOAT_T>& list) override{
            Propagator<FLOAT_T>::setEnergyList(list);

            for(auto& propagator : propagatorVector)
                propagator->setEnergyList(list);
        }

        void setCosineList(const std::vector<FLOAT_T>& list) override{
            Propagator<FLOAT_T>::setCosineList(list);

            for(size_t i = 0; i < propagatorVector.size(); i++){
                std::vector<FLOAT_T> myCos(cosineIndices[i].size());
                std::transform(cosineIndices[i].begin(),
                                cosineIndices[i].end(),
                                myCos.begin(),
                                [&](int icos){ return this->cosineList[icos]; }
                );
                propagatorVector[i]->setCosineList(myCos);
            }
        }

        void setProductionHeight(FLOAT_T heightKM) override{
            Propagator<FLOAT_T>::setProductionHeight(heightKM);

            for(auto& propagator : propagatorVector)
                propagator->setProductionHeight(heightKM);
        }

    public:
        void calculateProbabilities(NeutrinoType type) override{

            for(auto& propagator : propagatorVector)
                    propagator->calculateProbabilitiesAsync(type);

            for(auto& propagator : propagatorVector)
                    propagator->waitForCompletion();
        }

        FLOAT_T getProbability(int index_cosine, int index_energy, ProbType t) override{
                const int deviceIndex = getCosineDeviceIndex(index_cosine);
                const int localCosineIndex = localCosineIndices[index_cosine];

                return propagatorVector[deviceIndex]->getProbability(localCosineIndex, index_energy, t);
        }

    private:

        void setMaxlayers() override{
            Propagator<FLOAT_T>::setMaxlayers();

            for(auto& propagator : propagatorVector)
                propagator->setMaxlayers();
        }

        int getCosineDeviceIndex(int index_cosine){
        #if 0
                // block distribution
                int id = 0;
                for(int i = deviceIds.size(); i-- > 0;){
                        if(index_cosine < (i+1) * n_cosines / deviceIds.size())
                            id = i;
                }
        #else
                // cyclic distribution.
                const int id = index_cosine % deviceIds.size();
        #endif
                return id;
        }

    private:

        std::vector<int> deviceIds;
        std::vector<std::vector<int>> cosineIndices;
        std::vector<int> localCosineIndices;

        std::vector<int> cosineBatches;

        // one CudaPropagatorSingle per GPU
        std::vector<std::unique_ptr<CudaPropagatorSingle<FLOAT_T>>> propagatorVector;
    };







}

#endif


#endif // #ifdef __NVCC__
