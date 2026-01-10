// lsmc.cpp
#include <algorithm>
#include <cmath>
#include <vector>

#include "gbm.hpp"
#include "lsmc.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// Index contigu : i*(N_steps+1) + t
static inline size_t idx(int i, int t, int N_steps) {
  return static_cast<size_t>(i) * (static_cast<size_t>(N_steps) + 1u) +
         static_cast<size_t>(t);
}

// Solveur 3x3 (Gauss avec pivot partiel minimal)
// Max basis functions (Quartic = 5)
#define MAX_BASIS_FUNCTIONS 5

// Solveur NxN (Gauss avec pivot partiel minimal)
static inline void solveNxN(double A[MAX_BASIS_FUNCTIONS][MAX_BASIS_FUNCTIONS],
                            double b[MAX_BASIS_FUNCTIONS],
                            double x[MAX_BASIS_FUNCTIONS], int n) {
  // Copie locale de travail (pour ne pas modifier A et b originaux si besoin de
  // debug, mais ici on peut in-place) On travaille sur des copies locales.
  double Mat[MAX_BASIS_FUNCTIONS][MAX_BASIS_FUNCTIONS];
  double rhs[MAX_BASIS_FUNCTIONS];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      Mat[i][j] = A[i][j];
    rhs[i] = b[i];
  }

  for (int k = 0; k < n; ++k) {
    // Pivot partiel
    int piv = k;
    double best = std::fabs(Mat[k][k]);
    for (int i = k + 1; i < n; ++i) {
      const double v = std::fabs(Mat[i][k]);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (piv != k) {
      for (int j = 0; j < n; ++j)
        std::swap(Mat[k][j], Mat[piv][j]);
      std::swap(rhs[k], rhs[piv]);
    }

    double diag = Mat[k][k];
    if (std::fabs(diag) < 1e-14)
      diag = (diag >= 0.0 ? 1e-14 : -1e-14);

    const double inv = 1.0 / diag;

    for (int j = k; j < n; ++j)
      Mat[k][j] *= inv;
    rhs[k] *= inv;

    for (int i = 0; i < n; ++i) {
      if (i == k)
        continue;
      const double f = Mat[i][k];
      for (int j = k; j < n; ++j)
        Mat[i][j] -= f * Mat[k][j];
      rhs[i] -= f * rhs[k];
    }
  }

  for (int i = 0; i < n; ++i)
    x[i] = rhs[i];
}

// Old 3x3 solver removed (superseded by solveNxN)
// static inline void solve3x3(...)

double LSMC::priceAmericanPut(double S0, double K, double r, double sigma,
                              double T, int N_steps, int N_paths,
                              RegressionBasis basis) {
  const double dt = T / static_cast<double>(N_steps);
  const double discount = std::exp(-r * dt);

  // Layout contigu (path-major)
  std::vector<double> paths(static_cast<size_t>(N_paths) *
                            (static_cast<size_t>(N_steps) + 1u));
  std::vector<double> payoff(paths.size());

  // 1) Simulation GBM (OpenMP à l'intérieur)
  GBM::simulatePaths(paths.data(), S0, r, sigma, T, N_steps, N_paths);

  // 2) Payoff (put)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N_paths; ++i) {
    const size_t base =
        static_cast<size_t>(i) * (static_cast<size_t>(N_steps) + 1u);
    for (int t = 0; t <= N_steps; ++t) {
      const double S = paths[base + static_cast<size_t>(t)];
      payoff[base + static_cast<size_t>(t)] = std::max(K - S, 0.0);
    }
  }

  // 3) Cashflows init à maturité
  std::vector<double> cashflows(static_cast<size_t>(N_paths));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N_paths; ++i)
    cashflows[static_cast<size_t>(i)] = payoff[idx(i, N_steps, N_steps)];

  // 4) Backward induction: t = N_steps-1 ... 1
  for (int t = N_steps - 1; t > 0; --t) {
    // Determine N_basis for current regression
    int N_basis;
    if (basis == RegressionBasis::Cubic) {
      N_basis = 4;
    } else {
      N_basis = 3; // Hermite, Laguerre, Monomial, Chebyshev (default to 3)
    }

    // Initialize matrix A and vector B for regression
    double A_sum[MAX_BASIS_FUNCTIONS][MAX_BASIS_FUNCTIONS] = {{0.0}};
    double B_sum[MAX_BASIS_FUNCTIONS] = {0.0};

    // Check if any ITM paths contributed to regression
    bool all_zero = true;
    for (int k = 0; k < N_basis; ++k) {
      if (std::fabs(B_sum[k]) > 1e-15) {
        all_zero = false;
        break;
      }
    }

    if (all_zero) {
      for (int r = 0; r < N_basis; ++r) {
        for (int c = 0; c < N_basis; ++c) {
          if (std::fabs(A_sum[r][c]) > 1e-15) {
            all_zero = false;
            break;
          }
        }
        if (!all_zero)
          break;
      }
    }

    if (all_zero) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < N_paths; ++i)
        cashflows[static_cast<size_t>(i)] *= discount;
      continue;
    }

    // Solve (A^T A) beta = (A^T y)
    double beta[MAX_BASIS_FUNCTIONS];
    solveNxN(A_sum, B_sum, beta, N_basis);

    // Mise à jour cashflows (exercice vs continuation)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N_paths; ++i) {
      const double immediate = payoff[idx(i, t, N_steps)];

      if (immediate > 0.0) {
        const double S = paths[idx(i, t, N_steps)];
        double continuation = 0.0;

        // Recompute basis functions for this path to evaluate continuation
        // value
        double phi_path[MAX_BASIS_FUNCTIONS] = {0.0};
        if (basis == RegressionBasis::Hermite) {
          phi_path[0] = 1.0;
          phi_path[1] = S;
          phi_path[2] = S * S - 1.0;
        } else if (basis == RegressionBasis::Laguerre) {
          phi_path[0] = 1.0;
          phi_path[1] = 1.0 - S;
          phi_path[2] = 0.5 * (S * S - 4.0 * S + 2.0);
        } else if (basis == RegressionBasis::Chebyshev) {
          phi_path[0] = 1.0;
          phi_path[1] = S;
          phi_path[2] = 2.0 * S * S - 1.0;
        } else if (basis == RegressionBasis::Cubic) {
          phi_path[0] = 1.0;
          phi_path[1] = S;
          phi_path[2] = S * S;
          phi_path[3] = S * S * S;
        } else {
          // Monomial
          phi_path[0] = 1.0;
          phi_path[1] = S;
          phi_path[2] = S * S;
        }

        for (int k = 0; k < N_basis; ++k) {
          continuation += beta[k] * phi_path[k];
        }

        if (immediate > continuation)
          cashflows[static_cast<size_t>(i)] = immediate; // exercise à t
        else
          cashflows[static_cast<size_t>(i)] *=
              discount; // hold: discount vers t
      } else {
        cashflows[static_cast<size_t>(i)] *= discount; // OTM: hold
      }
    }
  }

  // 5) Retour à t0: il reste un pas dt (on a arrêté à t=1)
  double mean = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : mean) schedule(static)
#endif
  for (int i = 0; i < N_paths; ++i)
    mean += cashflows[static_cast<size_t>(i)];

  return (mean / static_cast<double>(N_paths)) * discount;
}
