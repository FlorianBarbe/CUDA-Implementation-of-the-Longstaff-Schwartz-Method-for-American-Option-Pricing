#pragma once

#ifdef LSMC_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <vector>

#include "gbm.hpp"
#include "regression.hpp"
#include "rng.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// =======================================
// Enum pour le choix de la base
// =======================================
enum class RegressionBasis { Monomial, Hermite, Laguerre, Chebyshev, Cubic };

// =======================================
// Classe LSMC : API CPU et GPU
// =======================================
class LSMC {
public:
  // Version CPU (que tu avais déjà dans ton .cpp)
  static double
  priceAmericanPut(double S0, double K, double r, double sigma, double T,
                   int N_steps, int N_paths,
                   RegressionBasis basis = RegressionBasis::Monomial,
                   int poly_degree = 3);

#ifdef LSMC_ENABLE_CUDA
  // Version GPU
  static double
  priceAmericanPutGPU(double S0, double K, double r, double sigma, double T,
                      int N_steps, int N_paths,
                      RegressionBasis basis = RegressionBasis::Monomial,
                      int poly_degree = 3);

#endif
};

// =======================================
// Structures pour la régression GPU (Max deg=10 => N=11)
// =======================================
// A (Symétrique) de taille N*N. Triangle sup = N*(N+1)/2.
// Pour N=11 (deg 10), taille = 66.
struct RegressionSumsGPU {
  double A[66];
  double B[11];
};

struct BetaGPU {
  double beta[11];
};

// =======================================
// API GPU (implémentée dans lsmc.cu)
// =======================================
#ifdef LSMC_ENABLE_CUDA

// Sommes de régression AᵀA, Aᵀy
void computeRegressionSumsGPU(const float *d_paths, const float *d_payoff,
                              const float *d_cashflows, int t, int N_steps,
                              int N_paths, float discount,
                              RegressionSumsGPU &out, RegressionBasis basis,
                              int poly_degree);

// Mise à jour des cashflows
void updateCashflowsGPU(const float *d_paths, const float *d_payoff,
                        float *d_cashflows, const BetaGPU &beta, float discount,
                        int t, int N_steps, int N_paths, RegressionBasis basis,
                        int poly_degree);

#endif // LSMC_ENABLE_CUDA
