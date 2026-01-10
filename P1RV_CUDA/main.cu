#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <vector>

#include "fdm.hpp"
#include "lsmc.hpp"

int main() {
  // ================================
  // Paramètres du produit
  // ================================
  const double S0 = 100.0;
  const double K = 100.0;
  const double r = 0.05;
  const double sigma = 0.2;
  const double T = 1.0;

  // ================================
  // Paramètres Benchmark
  // ================================
  const int N_steps = 50;

  const std::string csv_file = "Benchmarks/benchmark_degree_precision.csv";
  std::ofstream csv(csv_file, std::ios::trunc);
  if (!csv.is_open()) {
    std::cerr << "[ERROR] Impossible d'ouvrir le CSV\n";
    return 1;
  }
  csv << std::fixed << std::setprecision(6);

  // =====================================================
  // 1. FDM REFERENCE (RK4)
  // =====================================================
  std::cout << "==========================================\n";
  std::cout << "[INFO] Establishing FDM Baseline (RK4)...\n";
  std::cout << "==========================================\n";

  FiniteDifference fdm(S0, K, r, sigma, T, 200, 1000); // M=200, N=1000
  double p_rk4 = fdm.price(FdmMethod::RungeKutta4);
  std::cout << "[FDM RK4] Price: " << p_rk4 << "\n\n";

  // ================================
  // Warm-up GPU
  // ================================
#ifdef LSMC_ENABLE_CUDA
  std::cout << "[INFO] Warm-up GPU...\n";
  LSMC::priceAmericanPutGPU(S0, K, r, sigma, T, 50, 10'000,
                            RegressionBasis::Monomial, 3);
#endif

  // Save original max threads
  int orig_max_threads = 1;
#ifdef _OPENMP
  orig_max_threads = omp_get_max_threads();
#endif

  std::vector<int> path_counts = {10000, 100000};

  // Boucle benchmark
  std::vector<RegressionBasis> bases = {
      RegressionBasis::Monomial, RegressionBasis::Hermite,
      RegressionBasis::Laguerre, RegressionBasis::Chebyshev};

  csv << "Base,Degree,Mode,Paths,Price,Time_ms,Diff_RK4\n";

  for (auto basis : bases) {
    std::string basis_name;
    if (basis == RegressionBasis::Monomial)
      basis_name = "Monomial";
    else if (basis == RegressionBasis::Hermite)
      basis_name = "Hermite";
    else if (basis == RegressionBasis::Laguerre)
      basis_name = "Laguerre";
    else if (basis == RegressionBasis::Chebyshev)
      basis_name = "Chebyshev";

    std::cout << "Starting Benchmark for Basis: " << basis_name << "\n";

    for (int n_paths : path_counts) {
      std::cout << "  -> N_paths: " << n_paths << "\n";

      for (int degree = 1; degree <= 10; ++degree) {

        // -------------------------------------------------
        // GPU CUDA
        // -------------------------------------------------
        double price_gpu = 0.0;
        double time_gpu = 0.0;
#ifdef LSMC_ENABLE_CUDA
        auto t0_g = std::chrono::high_resolution_clock::now();
        price_gpu = LSMC::priceAmericanPutGPU(S0, K, r, sigma, T, N_steps,
                                              n_paths, basis, degree);
        auto t1_g = std::chrono::high_resolution_clock::now();
        time_gpu =
            std::chrono::duration<double, std::milli>(t1_g - t0_g).count();
#endif
        // -------------------------------------------------
        // CPU SEQUENTIEL
        // -------------------------------------------------
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        auto t0_c = std::chrono::high_resolution_clock::now();
        double price_cpu = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps,
                                                  n_paths, basis, degree);
        auto t1_c = std::chrono::high_resolution_clock::now();
        double time_cpu =
            std::chrono::duration<double, std::milli>(t1_c - t0_c).count();

        // -------------------------------------------------
        // CPU OPENMP
        // -------------------------------------------------
#ifdef _OPENMP
        omp_set_num_threads(orig_max_threads); // Restore original threads
        auto t0_o = std::chrono::high_resolution_clock::now();
        double price_omp = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps,
                                                  n_paths, basis, degree);
        auto t1_o = std::chrono::high_resolution_clock::now();
        double time_omp =
            std::chrono::duration<double, std::milli>(t1_o - t0_o).count();
#else
        double price_omp = 0.0;
        double time_omp = 0.0;
#endif

        double diff_cpu = std::abs(price_cpu - p_rk4);
        double diff_omp = std::abs(price_omp - p_rk4);
        double diff_gpu = std::abs(price_gpu - p_rk4);

        // CSV Format: Base,Degree,Mode,Paths,Price,Time_ms,Diff_RK4
        csv << basis_name << "," << degree << ",GPU," << n_paths << ","
            << price_gpu << "," << time_gpu << "," << diff_gpu << "\n";
        csv << basis_name << "," << degree << ",CPU," << n_paths << ","
            << price_cpu << "," << time_cpu << "," << diff_cpu << "\n";
#ifdef _OPENMP
        csv << basis_name << "," << degree << ",OMP," << n_paths << ","
            << price_omp << "," << time_omp << "," << diff_omp << "\n";
#endif
      }
    }
  }

  csv.close();
  std::cout << "\n[INFO] CSV écrit : " << csv_file << "\n";
  return 0;
}
