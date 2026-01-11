/**
 * @file lsmc.cpp
 * @brief Implémentation CPU de l'algorithme Longstaff-Schwartz (LSMC)
 *
 * Ce fichier contient l'implémentation séquentielle et parallèle (OpenMP)
 * de l'algorithme de pricing d'options américaines par Monte Carlo.
 *
 * L'algorithme LSMC fonctionne en plusieurs étapes :
 * 1. Simulation des trajectoires du sous-jacent (GBM)
 * 2. Calcul des payoffs à chaque date
 * 3. Backward induction : régression pour estimer la valeur de continuation
 * 4. Décision d'exercice optimal à chaque pas de temps
 * 5. Moyenne des cashflows actualisés pour obtenir le prix
 *
 * Bases de régression supportées : Monômiale, Hermite, Laguerre, Chebyshev,
 * Cubique
 *
 * @authors Florian Barbe, Narjisse El Manssouri
 * @date Janvier 2026
 * @copyright École Centrale de Nantes - Projet P1RV
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "gbm.hpp"
#include "lsmc.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Calcule l'index dans le tableau contigu pour (trajectoire, temps)
 *
 * Layout path-major : paths[i * (N_steps+1) + t]
 *
 * @param i Index de la trajectoire
 * @param t Index du pas de temps
 * @param N_steps Nombre total de pas de temps
 * @return Index dans le tableau 1D
 */
static inline size_t idx(int i, int t, int N_steps) {
  return static_cast<size_t>(i) * (static_cast<size_t>(N_steps) + 1u) +
         static_cast<size_t>(t);
}

// Solveur 3x3 (Gauss avec pivot partiel minimal)
// Max basis functions (Degree 10 = 11 functions)
#define MAX_BASIS_FUNCTIONS 11

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

// GENERIC BASIS EVAL (Min/Max needed for Chebyshev)
static void eval_basis_cpu(double S, int degree, RegressionBasis basis,
                           double *phi, double min_S, double max_S) {
  phi[0] = 1.0;
  if (degree >= 1) {
    if (basis == RegressionBasis::Chebyshev) {
      double width = max_S - min_S;
      double x_norm = (width > 1e-6) ? (2.0 * (S - min_S) / width - 1.0) : 0.0;
      phi[1] = x_norm;
    } else if (basis == RegressionBasis::Laguerre) {
      phi[1] = 1.0 - S;
    } else { // Monomial, Hermite
      phi[1] = S;
    }
  }

  for (int k = 2; k <= degree; ++k) {
    if (basis == RegressionBasis::Monomial) {
      phi[k] = phi[k - 1] * S;
    } else if (basis == RegressionBasis::Hermite) {
      phi[k] = S * phi[k - 1] - (double)(k - 1) * phi[k - 2];
    } else if (basis == RegressionBasis::Laguerre) {
      // (k)*L_k = (2k-1 - x)*L_{k-1} - (k-1)*L_{k-2}
      phi[k] =
          ((2.0 * (k - 1.0) + 1.0 - S) * phi[k - 1] - (k - 1.0) * phi[k - 2]) /
          (double)k;
    } else if (basis == RegressionBasis::Chebyshev) {
      double x_norm = phi[1];
      phi[k] = 2.0 * x_norm * phi[k - 1] - phi[k - 2];
    } else if (basis == RegressionBasis::Cubic) {
      // Explicit Cubic mapped to Monomial
      phi[k] = phi[k - 1] * S;
    }
  }
}

double LSMC::priceAmericanPut(double S0, double K, double r, double sigma,
                              double T, int N_steps, int N_paths,
                              RegressionBasis basis, int poly_degree) {
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
    int N_basis = poly_degree + 1;

    // Min/Max for Chebyshev
    double min_S = 1e30, max_S = -1e30;
    // Manual reduction for min/max (MSVC OpenMP 2.0 does not support min/max
    // reduction)
    double local_min = 1e30;
    double local_max = -1e30;

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    std::vector<double> t_mins(max_threads, 1e30);
    std::vector<double> t_maxs(max_threads, -1e30);

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      double th_min = 1e30;
      double th_max = -1e30;

#pragma omp for nowait
      for (int i = 0; i < N_paths; ++i) {
        double val = paths[idx(i, t, N_steps)];
        if (payoff[idx(i, t, N_steps)] > 0.0) {
          if (val < th_min)
            th_min = val;
          if (val > th_max)
            th_max = val;
        }
      }
      t_mins[tid] = th_min;
      t_maxs[tid] = th_max;
    }

    for (double v : t_mins)
      if (v < min_S)
        min_S = v;
    for (double v : t_maxs)
      if (v > max_S)
        max_S = v;

#else
    for (int i = 0; i < N_paths; ++i) {
      double val = paths[idx(i, t, N_steps)];
      if (payoff[idx(i, t, N_steps)] > 0.0) {
        if (val < min_S)
          min_S = val;
        if (val > max_S)
          max_S = val;
      }
    }
#endif

    if (min_S > max_S) {
      min_S = 0.0;
      max_S = 1.0;
    }

    // Initialize matrix A and vector B for regression
    // Defined outside ifdef to be visible for solveNxN
    double A_sum[MAX_BASIS_FUNCTIONS][MAX_BASIS_FUNCTIONS] = {{0.0}};
    double B_sum[MAX_BASIS_FUNCTIONS] = {0.0};

#ifdef _OPENMP
    {
      int max_threads = omp_get_max_threads();
      std::vector<std::vector<double>> thread_A(
          max_threads,
          std::vector<double>(MAX_BASIS_FUNCTIONS * MAX_BASIS_FUNCTIONS, 0.0));
      std::vector<std::vector<double>> thread_B(
          max_threads, std::vector<double>(MAX_BASIS_FUNCTIONS, 0.0));

#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        double phi[MAX_BASIS_FUNCTIONS];

#pragma omp for schedule(static) nowait
        for (int i = 0; i < N_paths; ++i) {
          double immediate = payoff[idx(i, t, N_steps)];
          if (immediate <= 0.0)
            continue;

          double S = paths[idx(i, t, N_steps)];
          double Y = cashflows[static_cast<size_t>(i)] * discount;

          eval_basis_cpu(S, poly_degree, basis, phi, min_S, max_S);

          for (int r = 0; r < N_basis; ++r) {
            for (int c = 0; c < N_basis; ++c) {
              thread_A[tid][r * MAX_BASIS_FUNCTIONS + c] += phi[r] * phi[c];
            }
            thread_B[tid][r] += phi[r] * Y;
          }
        }
      }

      // Reduction
      for (int th = 0; th < max_threads; ++th) {
        for (int r = 0; r < N_basis; ++r) {
          for (int c = 0; c < N_basis; ++c) {
            A_sum[r][c] += thread_A[th][r * MAX_BASIS_FUNCTIONS + c];
          }
          B_sum[r] += thread_B[th][r];
        }
      }
    }
#else
    {
      double phi[MAX_BASIS_FUNCTIONS];
      for (int i = 0; i < N_paths; ++i) {
        double immediate = payoff[idx(i, t, N_steps)];
        if (immediate <= 0.0)
          continue;

        double S = paths[idx(i, t, N_steps)];
        double Y = cashflows[static_cast<size_t>(i)] * discount;

        eval_basis_cpu(S, poly_degree, basis, phi, min_S, max_S);

        for (int r = 0; r < N_basis; ++r) {
          for (int c = 0; c < N_basis; ++c) {
            A_sum[r][c] += phi[r] * phi[c];
          }
          B_sum[r] += phi[r] * Y;
        }
      }
    }
#endif

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
        double phi_path[MAX_BASIS_FUNCTIONS];

        eval_basis_cpu(S, poly_degree, basis, phi_path, min_S, max_S);

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
