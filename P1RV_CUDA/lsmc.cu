/**
 * @file lsmc.cu
 * @brief Implémentation CUDA de l'algorithme Longstaff-Schwartz (LSMC)
 *
 * Ce fichier contient les kernels CUDA pour le pricing d'options américaines
 * par la méthode Monte Carlo avec régression par moindres carrés.
 *
 * Fonctionnalités principales :
 * - Simulation des payoffs sur GPU
 * - Calcul des sommes de régression parallélisé
 * - Résolution de systèmes linéaires (élimination de Gauss)
 * - Mise à jour des cashflows avec décision d'exercice optimal
 *
 * Bases de régression supportées : Monômiale, Hermite, Laguerre, Chebyshev,
 * Cubique
 *
 * @authors Florian Barbe, Narjisse El Manssouri
 * @date Janvier 2026
 * @copyright École Centrale de Nantes - Projet P1RV
 */

#include "lsmc.hpp"

#ifdef LSMC_ENABLE_CUDA

#include "minmax_kernel.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <vector>

// ======================================================
// Utilitaires CUDA - Gestion des erreurs
// ======================================================

/**
 * @brief Vérifie le code de retour CUDA et affiche une erreur en cas d'échec
 * @param code Code d'erreur CUDA à vérifier
 * @param file Nom du fichier source (macro __FILE__)
 * @param line Numéro de ligne (macro __LINE__)
 */
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "ERREUR CUDA %d (%s) dans %s:%d\n", (int)code,
            cudaGetErrorString(code), file, line);
    abort();
  }
}
#define CUDA_CHECK(x) gpuAssert((x), __FILE__, __LINE__)

// ======================================================
// Évaluation générique des bases polynomiales
// ======================================================

/**
 * @brief Évalue les polynômes de base pour la régression LSMC
 *
 * Cette fonction calcule les valeurs phi[0], phi[1], ..., phi[degree]
 * selon la base choisie. Utilisée dans les kernels de régression.
 *
 * @param S Prix du sous-jacent à évaluer
 * @param degree Degré polynomial maximum
 * @param basis_type Type de base (0=Monômiale, 1=Hermite, 2=Laguerre,
 * 3=Chebyshev, 4=Cubique)
 * @param phi Tableau de sortie pour les valeurs des polynômes
 * @param min_S Valeur minimale de S (pour normalisation Chebyshev)
 * @param max_S Valeur maximale de S (pour normalisation Chebyshev)
 */
__device__ void eval_basis_generic(double S, int degree, int basis_type,
                                   double *phi, double min_S, double max_S) {
  phi[0] = 1.0;
  if (degree >= 1) {
    if (basis_type == 3) { // Chebyshev
      double width = max_S - min_S;
      double x_norm = (width > 1e-6) ? (2.0 * (S - min_S) / width - 1.0) : 0.0;
      phi[1] = x_norm;
    } else if (basis_type == 2) { // Laguerre
      phi[1] = 1.0 - S;
    } else { // Monomial, Hermite
      phi[1] = S;
    }
  }

  for (int k = 2; k <= degree; ++k) {
    if (basis_type == 0) { // Monomial
      phi[k] = phi[k - 1] * S;
    } else if (basis_type == 1) { // Hermite
      phi[k] = S * phi[k - 1] - (double)(k - 1) * phi[k - 2];
    } else if (basis_type == 2) { // Laguerre
      phi[k] =
          ((2.0 * (k - 1.0) + 1.0 - S) * phi[k - 1] - (k - 1.0) * phi[k - 2]) /
          (double)k;
    } else if (basis_type == 3) { // Chebyshev
      double x_norm = phi[1];
      phi[k] = 2.0 * x_norm * phi[k - 1] - phi[k - 2];
    } else if (basis_type == 4) { // Cubic -> Monomial
      phi[k] = phi[k - 1] * S;
    }
  }
}

// ======================================================
// Kernels CUDA pour LSMC
// ======================================================

/**
 * @brief Calcule les payoffs pour un Put américain sur toutes les trajectoires
 *
 * Chaque thread traite une trajectoire complète. Le payoff d'un Put est max(K -
 * S, 0).
 *
 * @param d_paths Trajectoires simulées du sous-jacent (layout: [t][path])
 * @param d_payoff Tableau de sortie pour les payoffs
 * @param K Prix d'exercice (Strike)
 * @param N_steps Nombre de pas de temps
 * @param N_paths Nombre de trajectoires Monte Carlo
 */
__global__ void payoff_kernel(const float *__restrict__ d_paths,
                              float *__restrict__ d_payoff, float K,
                              int N_steps, int N_paths) {
  int path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= N_paths)
    return;

  for (int t = 0; t <= N_steps; ++t) {
    int idx = idx_t_path(t, path, N_paths);
    float S = d_paths[idx];
    d_payoff[idx] = fmaxf(K - S, 0.0f);
  }
}

__global__ void init_cashflows_kernel(const float *__restrict__ d_payoff,
                                      float *__restrict__ d_cashflows,
                                      int N_steps, int N_paths) {
  int path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= N_paths)
    return;
  int idxT = idx_t_path(N_steps, path, N_paths);
  d_cashflows[path] = d_payoff[idxT];
}

__global__ void sum_reduce_kernel(const float *__restrict__ d_in, double *d_out,
                                  int N) {
  // Simple block reduction then atomic add to global
  __shared__ double sdata[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double val = (i < N) ? (double)d_in[i] : 0.0;
  sdata[tid] = val;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(d_out, sdata[0]);
  }
}

__global__ void regression_sums_shared_kernel(
    const float *__restrict__ d_paths, const float *__restrict__ d_payoff,
    const float *__restrict__ d_cashflows, int t, int N_steps, int N_paths,
    float discount, double *__restrict__ d_sums, int basis_type, int degree,
    double min_S, double max_S) {

  __shared__ double s_block[80];

  int N_basis = degree + 1;
  int size_A = (N_basis * (N_basis + 1)) / 2;
  int total_size = size_A + N_basis;

  if (threadIdx.x < total_size)
    s_block[threadIdx.x] = 0.0;
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  double r[80];
  for (int k = 0; k < total_size; ++k)
    r[k] = 0.0;

  double phi[11];

  for (int i = tid; i < N_paths; i += stride) {
    int id_xt = idx_t_path(t, i, N_paths);
    double immediate = (double)d_payoff[id_xt];
    if (immediate <= 0.0)
      continue;

    double S = (double)d_paths[id_xt];
    double Y = (double)d_cashflows[i] * (double)discount;

    eval_basis_generic(S, degree, basis_type, phi, min_S, max_S);

    int idx = 0;
    for (int r_idx = 0; r_idx < N_basis; ++r_idx) {
      for (int c_idx = r_idx; c_idx < N_basis; ++c_idx) {
        r[idx++] += phi[r_idx] * phi[c_idx];
      }
    }
    for (int k = 0; k < N_basis; ++k) {
      r[size_A + k] += phi[k] * Y;
    }
  }

  for (int k = 0; k < total_size; ++k)
    atomicAdd(&s_block[k], r[k]);
  __syncthreads();

  if (threadIdx.x < total_size)
    atomicAdd(&d_sums[threadIdx.x], s_block[threadIdx.x]);
}

__global__ void solve_system_kernel(const double *__restrict__ d_sums,
                                    double *__restrict__ d_beta, int N) {
  if (threadIdx.x > 0 || blockIdx.x > 0)
    return;

  double A[11][11], rhs[11];
  int size_A = (N * (N + 1)) / 2;

  int idx = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      if (idx < size_A) {
        A[i][j] = d_sums[idx++];
        A[j][i] = A[i][j];
      }
    }
  }
  for (int i = 0; i < N; ++i)
    rhs[i] = d_sums[size_A + i];

  for (int i = 0; i < N; ++i) {
    double pivot = A[i][i];
    if (fabs(pivot) < 1e-12)
      pivot = 1e-12;
    for (int j = i + 1; j < N; ++j) {
      double factor = A[j][i] / pivot;
      rhs[j] -= factor * rhs[i];
      for (int k = i; k < N; ++k)
        A[j][k] -= factor * A[i][k];
    }
  }

  double x[11];
  for (int i = N - 1; i >= 0; --i) {
    double sum = 0.0;
    for (int j = i + 1; j < N; ++j)
      sum += A[i][j] * x[j];
    x[i] = (rhs[i] - sum) / A[i][i];
  }
  for (int i = 0; i < N; ++i)
    d_beta[i] = x[i];
}

__global__ void update_cashflows_kernel_resident(
    const float *__restrict__ d_paths, const float *__restrict__ d_payoff,
    float *__restrict__ d_cashflows, const double *__restrict__ d_beta,
    float discount, int t, int N_steps, int N_paths, int basis_type, int degree,
    double min_S, double max_S) {
  int path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= N_paths)
    return;

  double local_beta[11];
  for (int i = 0; i <= degree; ++i)
    local_beta[i] = d_beta[i];

  int id = idx_t_path(t, path, N_paths);
  float immediate = d_payoff[id];
  float S = d_paths[id];

  if (immediate > 0.0f) {
    double cont = 0.0;
    double phi[11];
    eval_basis_generic((double)S, degree, basis_type, phi, min_S, max_S);
    for (int i = 0; i <= degree; ++i)
      cont += local_beta[i] * phi[i];

    if ((double)immediate > cont)
      d_cashflows[path] = immediate;
    else
      d_cashflows[path] *= discount;
  } else {
    d_cashflows[path] *= discount;
  }
}

// ======================================================
// WRAPPERS AND LSMC LOGIC
// ======================================================

void computeRegressionSumsGPU_ptr(const float *d_paths, const float *d_payoff,
                                  const float *d_cashflows, int t, int N_steps,
                                  int N_paths, float discount,
                                  double *d_sums_out, RegressionBasis basis,
                                  int poly_degree) {

  double *d_minmax = nullptr;
  double h_minmax[2] = {0.0, 0.0};
  double min_S = 0.0, max_S = 1.0;

  int block = 256;
  int grid = (N_paths + block - 1) / block;

  if (basis == RegressionBasis::Chebyshev) {
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

  CUDA_CHECK(cudaMemset(d_sums_out, 0, 80 * sizeof(double)));

  int b_type = 0;
  if (basis == RegressionBasis::Hermite)
    b_type = 1;
  else if (basis == RegressionBasis::Laguerre)
    b_type = 2;
  else if (basis == RegressionBasis::Chebyshev)
    b_type = 3;
  else if (basis == RegressionBasis::Cubic)
    b_type = 4;

  regression_sums_shared_kernel<<<grid, block>>>(
      d_paths, d_payoff, d_cashflows, t, N_steps, N_paths, discount, d_sums_out,
      b_type, poly_degree, min_S, max_S);
}

double LSMC::priceAmericanPutGPU(double S0, double K, double r, double sigma,
                                 double T, int N_steps, int N_paths,
                                 RegressionBasis basis, int poly_degree,
                                 int block_size) {
  GbmParams params;
  params.S0 = (float)S0;
  params.r = (float)r;
  params.sigma = (float)sigma;
  params.T = (float)T;
  params.nSteps = N_steps;
  params.nPaths = N_paths;

  size_t nbPoints = (size_t)(N_steps + 1) * (size_t)N_paths;
  float *d_paths = nullptr, *d_payoff = nullptr, *d_cashflows = nullptr;
  double *d_sums = nullptr, *d_beta = nullptr;

  CUDA_CHECK(cudaMalloc(&d_paths, nbPoints * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_payoff, nbPoints * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cashflows, N_paths * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sums, 80 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_beta, 11 * sizeof(double)));

  cudaStream_t stream = 0;
  simulate_gbm_paths_cuda(params, RNGType::PseudoPhilox, d_paths, 1234ULL,
                          stream);

  int block = block_size;
  int gridPay = (N_paths + block - 1) / block;
  payoff_kernel<<<gridPay, block>>>(d_paths, d_payoff, (float)K, N_steps,
                                    N_paths);
  init_cashflows_kernel<<<gridPay, block>>>(d_payoff, d_cashflows, N_steps,
                                            N_paths);

  float dt = (float)T / (float)N_steps;
  float discount = expf(-(float)r * dt);

  int b_type = 0;
  if (basis == RegressionBasis::Hermite)
    b_type = 1;
  else if (basis == RegressionBasis::Laguerre)
    b_type = 2;
  else if (basis == RegressionBasis::Chebyshev)
    b_type = 3;

  double min_S_hack = 0.0, max_S_hack = 1.0;

  for (int t = N_steps - 1; t >= 1; --t) {
    computeRegressionSumsGPU_ptr(d_paths, d_payoff, d_cashflows, t, N_steps,
                                 N_paths, discount, d_sums, basis, poly_degree);

    solve_system_kernel<<<1, 1>>>(d_sums, d_beta, poly_degree + 1);

    update_cashflows_kernel_resident<<<gridPay, block>>>(
        d_paths, d_payoff, d_cashflows, d_beta, discount, t, N_steps, N_paths,
        b_type, poly_degree, min_S_hack, max_S_hack);
  }

  double *d_totalSum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_totalSum, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_totalSum, 0, sizeof(double)));

  int gridRed = (N_paths + 256 - 1) / 256;
  sum_reduce_kernel<<<gridRed, 256>>>(d_cashflows, d_totalSum, N_paths);

  double totalSum = 0.0;
  CUDA_CHECK(cudaMemcpy(&totalSum, d_totalSum, sizeof(double),
                        cudaMemcpyDeviceToHost));
  double price = (totalSum / (double)N_paths) *
                 std::exp(-r * ((double)T)); // Discount T or T/N steps?
  // Wait, priceAmericanPut logic usually reduces cashflows at t=1 (discounted).
  // The cashflows vector at end contains values at t=1 discounted from exercise
  // time. We need to discount from t=1 to t=0? No, usually cashflows are PV at
  // t=0 or t=1? Check init: d_cashflows = d_payoff at T. Loop: if continue, *=
  // discount. So at end (after t=1 loop), values are at t=1. We need one more
  // discount step? Or t=0 loop? Loop goes t >= 1. Standard LSMC: loop down to
  // t=1. Value at t=0 is just discounted mean of values at t=1? params.T is
  // total time. Correct logic: Discount from t=0. My loop discounts step by
  // step. At end of t=1 iteration, cashflows are at t=1. So final price needs *
  // exp(-r * dt) to get to t=0. In line 467 above logic was: price = mean *
  // exp(-r * (T/N_steps)). This seems correct (one step discount).
  price = (totalSum / (double)N_paths) * std::exp(-r * dt);

  CUDA_CHECK(cudaFree(d_paths));
  CUDA_CHECK(cudaFree(d_payoff));
  CUDA_CHECK(cudaFree(d_cashflows));
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_totalSum));

  return price;
}

// LEGACY WRAPPERS
void computeRegressionSumsGPU(const float *d_paths, const float *d_payoff,
                              const float *d_cashflows, int t, int N_steps,
                              int N_paths, float discount,
                              RegressionSumsGPU &out, RegressionBasis basis,
                              int poly_degree) {
  double *d_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sums, 80 * sizeof(double)));
  computeRegressionSumsGPU_ptr(d_paths, d_payoff, d_cashflows, t, N_steps,
                               N_paths, discount, d_sums, basis, poly_degree);
  double h_sums[80];
  CUDA_CHECK(
      cudaMemcpy(h_sums, d_sums, 80 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_sums));
  int N = poly_degree + 1;
  int size_A = (N * (N + 1)) / 2;
  for (int i = 0; i < size_A; ++i)
    out.A[i] = h_sums[i];
  for (int i = 0; i < N; ++i)
    out.B[i] = h_sums[size_A + i];
}

void updateCashflowsGPU(const float *d_paths, const float *d_payoff,
                        float *d_cashflows, const BetaGPU &beta, float discount,
                        int t, int N_steps, int N_paths, RegressionBasis basis,
                        int poly_degree) {
  double *d_beta = nullptr;
  CUDA_CHECK(cudaMalloc(&d_beta, 11 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_beta, beta.beta, 11 * sizeof(double),
                        cudaMemcpyHostToDevice));
  int b_type = 0;
  if (basis == RegressionBasis::Hermite)
    b_type = 1;
  else if (basis == RegressionBasis::Laguerre)
    b_type = 2;
  else if (basis == RegressionBasis::Chebyshev)
    b_type = 3;
  int block = 256;
  int grid = (N_paths + block - 1) / block;
  update_cashflows_kernel_resident<<<grid, block>>>(
      d_paths, d_payoff, d_cashflows, d_beta, discount, t, N_steps, N_paths,
      b_type, poly_degree, 0.0, 1.0);
  CUDA_CHECK(cudaFree(d_beta));
}

#endif
