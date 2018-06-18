
# CUDAProb3++

CUDAProb3++ calculates the effect of 3-flavor neutrino oscillation for neutrinos propagating through a sphere with piecewise constant density.
Given a matter profile (body), a list of trajectories through this body, and a list of neutrino energies, CUDAProb3++ can propagate the neutrinos for all pairs of (trajectory, neutrino energy) in parallel on multiple CUDA-capable GPUs, or on the host using multi-threading.

Computations can be performed in either 32-bit single-precision (float) or 64-bit double-precision (double)

CUDAProb3++ is a header-only CUDA implementation of Prob3++ http://webhome.phy.duke.edu/~raw22/public/Prob3++/.

# Usage

1.Create propagator

CUDAProb3++ provides three different classes for computing.
All classes have a template parameter which specifies the floating point datatype to use,
i.e. FLOAT_T=float or FLOAT_T=double

```
// cpu propagator with 4 threads
std::unique_ptr<Propagator<FLOAT_T>> propagator( new CpuPropagator<FLOAT_T>(n_cosines, n_energies, 4));
// Single GPU propagator using GPU 0
std::unique_ptr<Propagator<FLOAT_T>> propagator( new CudaPropagatorSingle<FLOAT_T>(0, n_cosines, n_energies));
// Multi GPU propagator which uses GPU 0 and GPU 1
std::unique_ptr<Propagator<FLOAT_T>> propagator( new CudaPropagator<FLOAT_T>(std::vector<int>{0, 1}, n_cosines, n_energies));  
```

2.Specify neutrino parameters and simulation settings

```
propagator->setEnergyList(energyList); // set energy list

propagator->setCosineList(cosineList); //set cosine list

propagator->setMNSMatrix(theta12, theta13, theta23, dcp); // set mixing matrix. angles in radians

propagator->setNeutrinoMasses(dm12sq, dm23sq); // set neutrino mass differences. unit: eV^2

propagator->setDensityFromFile("models/PREM_12layer.dat"); // set density model

propagator->setProductionHeight(22.0); // set neutrino production height in kilometers above earth
```

3.Perform calculation

    propagator->calculateProbabilities(cudaprob3::Neutrino); parameter is either cudaprob3::Neutrino or cudaprob3::Antineutrino

4.Get results

    FLOAT_T prob = propagator->getProbability(i, j, ProbType::m_e); // returns probability P(nu_m -> nu_e) for cosine bin i and energy bin j

A complete example is shown in example/main.cpp

To compile and run the example code, please set the GPU architecture flag in the Makefile according to your architecture.
