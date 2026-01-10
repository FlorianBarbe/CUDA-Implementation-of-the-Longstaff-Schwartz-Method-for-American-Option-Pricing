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
                   RegressionBasis basis = RegressionBasis::Monomial);

#ifdef LSMC_ENABLE_CUDA
  // Version GPU
  static double
  priceAmericanPutGPU(double S0, double K, double r, double sigma, double T,
                      int N_steps, int N_paths,
                      RegressionBasis basis = RegressionBasis::Monomial);
#endif
};

// =======================================
// Structures pour la régression GPU (Max 4x4)
// =======================================
// On stocke les coefficients de A (Symétrique) et B pour Ax=B
// Max deg=3 (Cubic) => 4 coefficients => A 4x4 (10 valeurs triangle sup), B 4
// valeurs On utilise un tableau plat pour simplifier
struct RegressionSumsGPU {
  double A[10]; // Triangle supérieur de A en row-major: 00, 01, 02, 03, 11, 12,
                // 13, 22, 23, 33
  double B[4]; // Vecteur B
};

struct BetaGPU {
  double beta[4]; // Coefficients résultants
};

// =======================================
// API GPU (implémentée dans lsmc.cu)
// =======================================
#ifdef LSMC_ENABLE_CUDA

// Sommes de régression AᵀA, Aᵀy
void computeRegressionSumsGPU(const float *d_paths, const float *d_payoff,
                              const float *d_cashflows, int t, int N_steps,
                              int N_paths, float discount,
                              RegressionSumsGPU &out, RegressionBasis basis);

// Mise à jour des cashflows
void updateCashflowsGPU(const float *d_paths, const float *d_payoff,
                        float *d_cashflows, const BetaGPU &beta, float discount,
                        int t, int N_steps, int N_paths, RegressionBasis basis);

#endif // LSMC_ENABLE_CUDA
