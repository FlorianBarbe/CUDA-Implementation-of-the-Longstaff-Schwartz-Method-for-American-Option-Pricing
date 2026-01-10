#include "gbm.hpp"
#include "lsmc.hpp"

#ifdef LSMC_ENABLE_CUDA

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <vector>

// ======================================================
// Helpers CUDA
// ======================================================
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR %d (%s) at %s:%d\n", (int)code,
            cudaGetErrorString(code), file, line);
    abort();
  }
}
#define CUDA_CHECK(x) gpuAssert((x), __FILE__, __LINE__)

// ======================================================
// Helper Indexation TIME-MAJOR : [t][path]
// ======================================================
__device__ __forceinline__ int idx_t_path(int t, int path, int N_paths) {
  // t * N_paths + path
  return t * N_paths + path;
}

// ======================================================
// PAYOFF KERNEL (put amricain)
// ======================================================
__global__ void payoff_kernel(const float *__restrict__ d_paths,
                              float *__restrict__ d_payoff, float K,
                              int N_steps, int N_paths) {
  int path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= N_paths)
    return;

  // On parcourt toute la grille (t=0..N_steps) pour ce path
  // Avec layout [t][path], l'accs mmoire n'est PAS contigu pour un mme thread
  // qui boucle sur t. MAIS, ici on appelle payoff_kernel une seule fois pour
  // tout remplir ? Ou alors on paralllise sur (path) et on boucle sur t ? Mieux
  // : grille 2D ou 1D avec stride ? Vu le layout Time-Major, si on veut
  // coalescing parfait, il faut que thread id (x) corresponde path. Donc thread
  // x lit (t, x), (t+1, x)... A t fix, thread x et x+1 lisent des adresses
  // adjacentes. C'est parfait.

  for (int t = 0; t <= N_steps; ++t) {
    int idx = idx_t_path(t, path, N_paths);
    float S = d_paths[idx];
    d_payoff[idx] = fmaxf(K - S, 0.0f);
  }
}
// Note: On pourrait aussi faire un kernel parallle sur tout (t, path), mais la
// boucle t simple est ok car N_steps petit vs N_paths.

// ======================================================
// INIT CASHFLOWS ( maturit)
// ======================================================
__global__ void init_cashflows_kernel(const float *__restrict__ d_payoff,
                                      float *__restrict__ d_cashflows,
                                      int N_steps, int N_paths) {
  int path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= N_paths)
    return;

  int idxT = idx_t_path(N_steps, path, N_paths);
  d_cashflows[path] = d_payoff[idxT];
}

// ======================================================
// REGRESSION GPU OPTIMISE (SHARED MEMORY REDUCTION)
// ======================================================
__global__ void regression_sums_shared_kernel(
    const float *__restrict__ d_paths, const float *__restrict__ d_payoff,
    const float *__restrict__ d_cashflows, int t, int N_steps, int N_paths,
    float discount, double *__restrict__ d_sums, int basis_type, double min_S,
    double max_S) {

  // Max needed = 14 doubles (4x4 sym A + 4 B)
  __shared__ double s_block[14];
  if (threadIdx.x < 14)
    s_block[threadIdx.x] = 0.0;
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  double r[14] = {0.0};

  for (int i = tid; i < N_paths; i += stride) {
    int id_xt = idx_t_path(t, i, N_paths);

    double immediate = (double)d_payoff[id_xt];
    if (immediate <= 0.0)
      continue; // ITM Only

    double S = (double)d_paths[id_xt];
    double Y = (double)d_cashflows[i] * (double)discount;

    double phi[4];
    int N_basis = 3;

    if (basis_type == 0) { // Monomial
      phi[0] = 1.0;
      phi[1] = S;
      phi[2] = S * S;
    } else if (basis_type == 1) { // Hermite
      phi[0] = 1.0;
      phi[1] = S;
      phi[2] = S * S - 1.0;
    } else if (basis_type == 2) { // Laguerre
      phi[0] = 1.0;
      phi[1] = 1.0 - S;
      phi[2] = 0.5 * (S * S - 4.0 * S + 2.0);
    } else if (basis_type == 3) { // Chebyshev
      // Map [min, max] -> [-1, 1]
      double width = max_S - min_S;
      double x_norm = (width > 1e-6) ? (2.0 * (S - min_S) / width - 1.0) : 0.0;
      phi[0] = 1.0;
      phi[1] = x_norm;
      phi[2] = 2.0 * x_norm * x_norm - 1.0;
    } else if (basis_type == 4) { // Cubic
      phi[0] = 1.0;
      phi[1] = S;
      phi[2] = S * S;
      phi[3] = S * S * S;
      N_basis = 4;
    }

    // Accumulate A (Upper triangular)
    int idx = 0;
    for (int r_idx = 0; r_idx < N_basis; ++r_idx) {
      for (int c_idx = r_idx; c_idx < N_basis; ++c_idx) {
        r[idx++] += phi[r_idx] * phi[c_idx];
      }
    }

    // Accumulate B (Start idx is 10 because max A is 10 for N=4, for N=3 it is
    // 6 but we use fixed layout?) Let's stick to dense packing for generic N
    // Current Generic Packer above puts A sequentially.
    // If N=3, A has 6 elements. If N=4, A has 10.
    // To simplify: We always assume max layout of 10 for A, and B starts at 10.
    // B indices: 10, 11, 12, 13

    for (int k = 0; k < N_basis; ++k) {
      r[10 + k] += phi[k] * Y;
    }
  }

  // Reduction in shared
  for (int k = 0; k < 14; ++k) {
    atomicAdd(&s_block[k], r[k]);
  }

  __syncthreads();

  if (threadIdx.x < 14) {
    atomicAdd(&d_sums[threadIdx.x], s_block[threadIdx.x]);
  }
}

#include "minmax_kernel.cuh"

void computeRegressionSumsGPU(const float *d_paths, const float *d_payoff,
                              const float *d_cashflows, int t, int N_steps,
                              int N_paths, float discount,
                              RegressionSumsGPU &out, RegressionBasis basis) {
  double *d_sums = nullptr;
  // Reduce min/max if Chebyshev
  double *d_minmax = nullptr;
  double h_minmax[2] = {0.0, 0.0};
  double min_S = 0.0, max_S = 1.0;

  int block = 256;
  int grid = (N_paths + block - 1) / block;

  if (basis == RegressionBasis::Chebyshev) {
    CUDA_CHECK(cudaMalloc(&d_minmax, 2 * sizeof(double)));
    // Initialize min to huge, max to -huge
    double init_min = 1e30;
    double init_max = -1e30;
    CUDA_CHECK(cudaMemcpy(d_minmax, &init_min, sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_minmax + 1, &init_max, sizeof(double),
                          cudaMemcpyHostToDevice));

    reduce_minmax_kernel<<<grid, block>>>(d_paths, d_payoff, t, N_steps,
                                          N_paths, d_minmax, d_minmax + 1);

    CUDA_CHECK(cudaMemcpy(h_minmax, d_minmax, 2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    min_S = h_minmax[0];
    max_S = h_minmax[1];
    CUDA_CHECK(cudaFree(d_minmax));
  }

  CUDA_CHECK(cudaMalloc(&d_sums, 14 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_sums, 0, 14 * sizeof(double)));

  // On limite la grille pour viter trop d'overhead si N_paths est immense
  // Mais bon, N_paths=300k, grid=1000 blocks, a va.

  int b_type = 0;
  if (basis == RegressionBasis::Monomial)
    b_type = 0;
  else if (basis == RegressionBasis::Hermite)
    b_type = 1;
  else if (basis == RegressionBasis::Laguerre)
    b_type = 2;
  else if (basis == RegressionBasis::Chebyshev)
    b_type = 3;
  else if (basis == RegressionBasis::Cubic)
    b_type = 4;

  regression_sums_shared_kernel<<<grid, block>>>(d_paths, d_payoff, d_cashflows,
                                                 t, N_steps, N_paths, discount,
                                                 d_sums, b_type, min_S, max_S);
  CUDA_CHECK(cudaPeekAtLastError());

  // Pas besoin de sync ici si on copie juste aprs (le copy synchronise
  // implicitement ou stream 0) Mais pour scurit:
  // CUDA_CHECK(cudaDeviceSynchronize()); // On laisse le memcpy sync.

  double h_sums[14];
  CUDA_CHECK(
      cudaMemcpy(h_sums, d_sums, 14 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_sums));

  // Copy A (10 vals)
  for (int i = 0; i < 10; ++i)
    out.A[i] = h_sums[i];
  // Copy B (4 vals, start at offset 10)
  for (int i = 0; i < 4; ++i)
    out.B[i] = h_sums[10 + i];
}

// ======================================================
// UPDATE CASHFLOWS (dcision exercice / continuation)
// ======================================================
__global__ void update_cashflows_kernel(
    const float *__restrict__ d_paths, const float *__restrict__ d_payoff,
    float *__restrict__ d_cashflows, BetaGPU beta, float discount, int t,
    int N_steps, int N_paths, int basis_type, double min_S, double max_S) {
  int path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= N_paths)
    return;

  int id = idx_t_path(t, path, N_paths);

  float immediate = d_payoff[id];
  float S = d_paths[id];

  if (immediate > 0.0f) {
    double cont = 0.0;

    double phi[4];
    // Monomial default
    phi[0] = 1.0;
    phi[1] = S;
    phi[2] = S * S;
    phi[3] = 0.0;

    if (basis_type == 1) { // Hermite
      phi[2] = S * S - 1.0;
    } else if (basis_type == 2) { // Laguerre
      phi[1] = 1.0 - S;
      phi[2] = 0.5 * (S * S - 4.0 * S + 2.0);
    } else if (basis_type == 3) { // Chebyshev
      double width = max_S - min_S;
      double x_norm = (width > 1e-6) ? (2.0 * (S - min_S) / width - 1.0) : 0.0;
      phi[1] = x_norm;
      phi[2] = 2.0 * x_norm * x_norm - 1.0;
    } else if (basis_type == 4) { // Cubic
      phi[3] = S * S * S;
    }

    // Dot product with beta (size 4)
    for (int i = 0; i < 4; ++i)
      cont += beta.beta[i] * phi[i];

    if ((double)immediate > cont)
      d_cashflows[path] = immediate; // exercice
    else
      d_cashflows[path] *= discount; // continuation
  } else {
    d_cashflows[path] *= discount; // hors de la monnaie
  }
}

// Note: For update, we need min/max again for Chebyshev.
// Ideally we should pass it from computeRegressionSums, but the API separates
// them. We will recompute it. It's suboptimal but consistent.

void updateCashflowsGPU(const float *d_paths, const float *d_payoff,
                        float *d_cashflows, const BetaGPU &beta, float discount,
                        int t, int N_steps, int N_paths,
                        RegressionBasis basis) {
  int block = 256;
  int grid = (N_paths + block - 1) / block;

  double min_S = 0.0, max_S = 1.0;
  if (basis == RegressionBasis::Chebyshev) {
    double *d_minmax = nullptr;
    double h_minmax[2];
    CUDA_CHECK(cudaMalloc(&d_minmax, 2 * sizeof(double)));
    double init_min = 1e30;
    double init_max = -1e30;
    CUDA_CHECK(cudaMemcpy(d_minmax, &init_min, sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_minmax + 1, &init_max, sizeof(double),
                          cudaMemcpyHostToDevice));

    reduce_minmax_kernel<<<grid, block>>>(d_paths, d_payoff, t, N_steps,
                                          N_paths, d_minmax, d_minmax + 1);

    CUDA_CHECK(cudaMemcpy(h_minmax, d_minmax, 2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    min_S = h_minmax[0];
    max_S = h_minmax[1];
    CUDA_CHECK(cudaFree(d_minmax));
  }

  int b_type = 0;
  if (basis == RegressionBasis::Monomial)
    b_type = 0;
  else if (basis == RegressionBasis::Hermite)
    b_type = 1;
  else if (basis == RegressionBasis::Laguerre)
    b_type = 2;
  else if (basis == RegressionBasis::Chebyshev)
    b_type = 3;
  else if (basis == RegressionBasis::Cubic)
    b_type = 4;

  update_cashflows_kernel<<<grid, block>>>(d_paths, d_payoff, d_cashflows, beta,
                                           discount, t, N_steps, N_paths,
                                           b_type, min_S, max_S);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ======================================================
// Rsolution 33 (Cramer)
// ======================================================
// ======================================================
// Helper: Solve NxN System (Gaussian Elimination)
// N <= 4 (Hardcoded limit for registers)
// ======================================================
static BetaGPU solveSystem(int N, const double A_flat[10], const double B[4]) {
  // Reconstruct full matrix A from flat upper triangular
  double A[4][4];

  // Mapping flat idx to (i,j)
  // 00, 01, 02, 03 -> 0, 1, 2, 3
  // 11, 12, 13     -> 4, 5, 6
  // 22, 23         -> 7, 8
  // 33             -> 9

  // 3x3 Case (indices: 0, 1, 2, 4, 5, 7)
  // 4x4 Case (all indices)

  int idx = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      A[i][j] = A_flat[idx++];
      A[j][i] = A[i][j]; // Symmetry
    }
  }

  double rhs[4];
  for (int i = 0; i < N; ++i)
    rhs[i] = B[i];

  // Forward Elimination
  for (int i = 0; i < N; ++i) {
    // Pivot
    double pivot = A[i][i];
    if (fabs(pivot) < 1e-12)
      return {0}; // Singularity check

    for (int j = i + 1; j < N; ++j) {
      double factor = A[j][i] / pivot;
      rhs[j] -= factor * rhs[i];
      for (int k = i; k < N; ++k) {
        A[j][k] -= factor * A[i][k];
      }
    }
  }

  // Backward Substitution
  double x[4] = {0};
  for (int i = N - 1; i >= 0; --i) {
    double sum = 0.0;
    for (int j = i + 1; j < N; ++j) {
      sum += A[i][j] * x[j];
    }
    x[i] = (rhs[i] - sum) / A[i][i];
  }

  BetaGPU out;
  for (int i = 0; i < 4; ++i)
    out.beta[i] = x[i];
  return out;
}

static BetaGPU solveRegression(const RegressionSumsGPU &s,
                               RegressionBasis basis) {
  int N = 3;
  if (basis == RegressionBasis::Cubic)
    N = 4;
  return solveSystem(N, s.A, s.B);
}

// ======================================================
// FINAL REDUCTION GPU
// ======================================================
__global__ void sum_reduce_kernel(const float *__restrict__ input,
                                  double *__restrict__ output, int n) {
  // Simple reduction : pour l'instant un seul bloc pour simplifier, ou
  // atomicAdd global. Optimisation : warp reduce. Ici on fait simple :
  // atomicAdd sur global output[0].

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  double sum = 0.0;
  for (int i = tid; i < n; i += stride) {
    sum += (double)input[i];
  }

  // Rduction Shared Mem
  __shared__ double sdata[256];
  // Init share
  sdata[threadIdx.x] = 0.0;

  // Reduce local sum into shared
  sdata[threadIdx.x] = sum;
  __syncthreads();

  // Tree reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(output, sdata[0]);
  }
}

// ======================================================
// LSMC::priceAmericanPutGPU
// ======================================================
double LSMC::priceAmericanPutGPU(double S0, double K, double r, double sigma,
                                 double T, int N_steps, int N_paths,
                                 RegressionBasis basis) {
  // 1) Paramtres GBM
  GbmParams params;
  params.S0 = (float)S0;
  params.r = (float)r;
  params.sigma = (float)sigma;
  params.T = (float)T;
  params.nSteps = N_steps;
  params.nPaths = N_paths;

  size_t nbPoints = (size_t)(N_steps + 1) * (size_t)N_paths;

  float *d_paths = nullptr;
  float *d_payoff = nullptr;
  float *d_cashflows = nullptr;

  CUDA_CHECK(cudaMalloc(&d_paths, nbPoints * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_payoff, nbPoints * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cashflows, N_paths * sizeof(float)));

  cudaStream_t stream = 0;

  // 2) Simulation des trajectoires (TIME-MAJOR)
  simulate_gbm_paths_cuda(params, RNGType::PseudoPhilox, d_paths, 1234ULL,
                          stream);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 3) Payoff immdiat (put) sur toute la grille
  int block = 256;
  int gridPay = (N_paths + block - 1) / block;
  payoff_kernel<<<gridPay, block>>>(d_paths, d_payoff, (float)K, N_steps,
                                    N_paths);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 4) Cashflows  maturit
  // Init from last step (in time major: idx = N_steps * N_paths + path)
  int gridInit = (N_paths + block - 1) / block;
  init_cashflows_kernel<<<gridInit, block>>>(d_payoff, d_cashflows, N_steps,
                                             N_paths);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 5) Backward induction LSMC
  float dt = (float)T / (float)N_steps;
  float discount = expf(-(float)r * dt);

  for (int t = N_steps - 1; t >= 1; --t) {

    RegressionSumsGPU sums{};
    computeRegressionSumsGPU(d_paths, d_payoff, d_cashflows, t, N_steps,
                             N_paths, discount, sums, basis);

    BetaGPU beta = solveRegression(sums, basis);

    updateCashflowsGPU(d_paths, d_payoff, d_cashflows, beta, discount, t,
                       N_steps, N_paths, basis);
  }

  // 6) Moyenne des cashflows initiaux (Reduction GPU)
  double *d_totalSum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_totalSum, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_totalSum, 0, sizeof(double)));

  // Lancement kernel reduction
  int gridRed = (N_paths + 256 - 1) / 256;
  // Si N_paths est trs grand, peut-tre plusieurs pass.
  // Ici on fait atomicAdd global  la fin de chaque bloc, donc 1 pass suffit.
  sum_reduce_kernel<<<gridRed, 256>>>(d_cashflows, d_totalSum, N_paths);

  double totalSum = 0.0;
  CUDA_CHECK(cudaMemcpy(&totalSum, d_totalSum, sizeof(double),
                        cudaMemcpyDeviceToHost));

  double mean = totalSum / (double)N_paths;

  double step_dt = (double)T / (double)N_steps;
  double price = mean * std::exp(-r * step_dt);

  CUDA_CHECK(cudaFree(d_paths));
  CUDA_CHECK(cudaFree(d_payoff));
  CUDA_CHECK(cudaFree(d_cashflows));
  CUDA_CHECK(cudaFree(d_totalSum));

  return price;
}

#endif // LSMC_ENABLE_CUDA
