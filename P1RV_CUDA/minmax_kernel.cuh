#pragma once

#include <cuda_runtime.h>
#include <device_functions.h>

// Helper Indexation TIME-MAJOR : [t][path]
__device__ __forceinline__ int idx_t_path(int t, int path, int N_paths) {
  // t * N_paths + path
  return t * N_paths + path;
}

__global__ void reduce_minmax_kernel(const float *__restrict__ d_paths,
                                     const float *__restrict__ d_payoff, int t,
                                     int N_steps, int N_paths,
                                     double *d_out_min, double *d_out_max) {
  // Shared mem for block reduction
  __shared__ double s_min[256];
  __shared__ double s_max[256];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  double local_min = 1e30;
  double local_max = -1e30;

  for (int i = tid; i < N_paths; i += stride) {
    int id = idx_t_path(t, i, N_paths);
    if (d_payoff[id] > 0.0f) { // ITM only
      double val = (double)d_paths[id];
      if (val < local_min)
        local_min = val;
      if (val > local_max)
        local_max = val;
    }
  }

  s_min[threadIdx.x] = local_min;
  s_max[threadIdx.x] = local_max;
  __syncthreads();

  // Block reduce
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (s_min[threadIdx.x + s] < s_min[threadIdx.x])
        s_min[threadIdx.x] = s_min[threadIdx.x + s];
      if (s_max[threadIdx.x + s] > s_max[threadIdx.x])
        s_max[threadIdx.x] = s_max[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Write partial result for block to global (using atomic for simplicity on
  // small grid)
  if (threadIdx.x == 0) {
    // AtomicMin/Max for double requires CUDA >= 6.0 and tricks or custom
    // implementation For simplicity: loop on CPU or use simple CAS loop here?
    // Since we have few blocks, let's just use a custom atomic CAS loop or
    // write to an array d_block_min[gridDim.x] and finish on CPU.
    // EASIEST: Just write to d_sums[14] area? No size is small.
    // Let's implement AtomicMin/Max double using CAS.

    // AtomicMin
    unsigned long long *addr_min = (unsigned long long *)d_out_min;
    unsigned long long old = *addr_min, assumed;
    do {
      assumed = old;
      double d_assumed = __longlong_as_double(assumed);
      double d_val = fmin(d_assumed, s_min[0]);
      old = atomicCAS(addr_min, assumed, __double_as_longlong(d_val));
    } while (assumed != old);

    // AtomicMax
    unsigned long long *addr_max = (unsigned long long *)d_out_max;
    old = *addr_max;
    do {
      assumed = old;
      double d_assumed = __longlong_as_double(assumed);
      double d_val = fmax(d_assumed, s_max[0]);
      old = atomicCAS(addr_max, assumed, __double_as_longlong(d_val));
    } while (assumed != old);
  }
}
