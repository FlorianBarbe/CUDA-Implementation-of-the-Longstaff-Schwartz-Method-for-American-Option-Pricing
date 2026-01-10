// gbm_cuda.cu
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


#include "gbm.hpp"

#ifndef __CUDACC__
#error "gbm.cu must be compiled with NVCC (CUDA compiler)."
#endif

// =====================================================================================
// Helpers CUDA
// =====================================================================================
#define CUDA_CHECK(ans)                                                        \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR %d (%s) at %s:%d\n", code,
            cudaGetErrorString(code), file, line);
  }
}

#define CURAND_CHECK(ans)                                                      \
  {                                                                            \
    curandAssert((ans), __FILE__, __LINE__);                                   \
  }
inline void curandAssert(curandStatus_t code, const char *file, int line) {
  if (code != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "CURAND ERROR %d at %s:%d\n", code, file, line);
  }
}

// Layout mmoire optimis : TIME-MAJOR [t][path]
// path \in [0, nPaths-1]
// t    \in [0, nSteps]
// index = t * nPaths + path
//
// Cela permet la lecture coalesce quand threads adjacents (paths adjacents)
// lisent le mme t.

// =====================================================================================
// 1) Kernel Philox
// =====================================================================================
__global__ void gbm_paths_philox_kernel(GbmParams params,
                                        unsigned long long seed,
                                        float *__restrict__ d_paths) {
  int pathId = blockIdx.x * blockDim.x + threadIdx.x;
  if (pathId >= params.nPaths)
    return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, pathId, 0ULL, &state);

  float dt = params.T / params.nSteps;
  float drift = (params.r - 0.5f * params.sigma * params.sigma) * dt;
  float vol = params.sigma * sqrtf(dt);

  float S = params.S0;

  // t=0
  // Index: 0 * nPaths + pathId = pathId
  d_paths[pathId] = S;

  for (int t = 1; t <= params.nSteps; ++t) {
    float z = curand_normal(&state);
    S *= expf(drift + vol * z);

    // Index: t * nPaths + pathId
    size_t idx = (size_t)t * (size_t)params.nPaths + (size_t)pathId;
    d_paths[idx] = S;
  }
}

// =====================================================================================
// 2) Kernel Sobol
// =====================================================================================
__global__ void
gbm_paths_from_normals_kernel(GbmParams params,
                              const float *__restrict__ d_normals,
                              float *__restrict__ d_paths) {
  int pathId = blockIdx.x * blockDim.x + threadIdx.x;
  if (pathId >= params.nPaths)
    return;

  float dt = params.T / params.nSteps;
  float drift = (params.r - 0.5f * params.sigma * params.sigma) * dt;
  float vol = params.sigma * sqrtf(dt);

  float S = params.S0;
  // t=0
  d_paths[pathId] = S;

  for (int t = 1; t <= params.nSteps; ++t) {
    // normals: pour l'instant on suppose Layout path-major pour curand
    // (standard) [path][t-1] MAIS si on veut optimiser, il faudrait generer en
    // time-major. Ici on garde l'accs aux normales tel quel (c'est un
    // read-only, moins critique mais...) idxN = pathId * nSteps + (t-1) ->
    // C'est du path-major. Coalesc si t est constant? NON. Si threads lisent
    // t=1: thread 0 lit idx=0, thread 1 lit idx=nSteps. NON COALESC.
    // TODO: Transposer d_normals ou gnrer diffremment.
    // Pour l'instant on laisse tel quel pour d_normals, on optimise d_paths.

    size_t idxN = (size_t)pathId * (size_t)params.nSteps + (size_t)(t - 1);
    float z = d_normals[idxN];

    S *= expf(drift + vol * z);

    size_t idxS = (size_t)t * (size_t)params.nPaths + (size_t)pathId;
    d_paths[idxS] = S;
  }
}

// =====================================================================================
// 3) Implmentation host
// =====================================================================================
void simulate_gbm_paths_cuda(const GbmParams &params, RNGType rng,
                             float *d_paths, unsigned long long seed,
                             cudaStream_t stream) {
  int nPaths = params.nPaths;
  int nSteps = params.nSteps;

  dim3 block(256);
  dim3 grid((nPaths + block.x - 1) / block.x);

  if (rng == RNGType::PseudoPhilox) {
    gbm_paths_philox_kernel<<<grid, block, 0, stream>>>(params, seed, d_paths);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    // NOTE: curandGenerateNormal gnre [path0_t0...tn, path1_t0...tn] par dfaut
    // (sauf si on change l'ordering le generator) C'est du Path-Major. L'accs
    // dans le kernel sera non-coalesc pour les normales. C'est un point
    // d'amlioration futur.

    size_t nDim = static_cast<size_t>(nSteps);
    size_t total = static_cast<size_t>(nPaths) * nSteps;

    float *d_normals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_normals, total * sizeof(float)));

    curandGenerator_t gen;
    CURAND_CHECK(
        curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));

    CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(gen, nDim));
    CURAND_CHECK(curandSetStream(gen, stream));

    // Ordre SOBOL standard
    CURAND_CHECK(curandGenerateNormal(gen, d_normals, total, 0.0f, 1.0f));
    CURAND_CHECK(curandDestroyGenerator(gen));

    gbm_paths_from_normals_kernel<<<grid, block, 0, stream>>>(params, d_normals,
                                                              d_paths);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_normals));
  }
}