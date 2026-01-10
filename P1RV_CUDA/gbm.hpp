#pragma once

#ifndef LSMC_ENABLE_CUDA
#ifdef __CUDACC__
#define LSMC_ENABLE_CUDA 1
#endif
#endif

#ifdef LSMC_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

// =========================================================
// PARAMÈTRES GBM GPU (MATCH EXACT AVEC gbm_cuda.cu)
// =========================================================
struct GbmParams {
    float S0;
    float r;
    float sigma;
    float T;
    int   nSteps;
    int   nPaths;
};

// Types de RNG disponibles pour CUDA (match exact)
enum class RNGType {
    PseudoPhilox = 0,
    QuasiSobol   = 1
};

// =========================================================
// Déclaration du simulateur GPU
// =========================================================
#ifdef LSMC_ENABLE_CUDA
void simulate_gbm_paths_cuda(const GbmParams& params,
                             RNGType rng,
                             float* d_paths,
                             unsigned long long seed = 1234ULL,
                             cudaStream_t stream = 0);
#endif


// =========================================================
// PARTIE CPU (classe GBM)
// =========================================================

#include <vector>
#include <string>
#include "rng.hpp"

class GBM {
private:
    double S0;
    double r;
    double sigma;
    double T;
    int N_steps;

public:
    GBM(double S0, double r, double sigma, double T, int N_steps);

    void simulate_path(RNG& rng, double* path_out);
    std::vector<double> simulate(RNG& rng);

    static void simulatePaths(double* paths,
                              double S0, double r, double sigma,
                              double T, int N_steps, int N_paths);

    static void exportCSV(const std::vector<std::vector<double>>& paths,
                          const std::string& filename);
};
