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

// Structure to hold stats
struct BenchmarkStats {
  double mean_price;
  double std_price;
  double mean_time_ms;
  double throughput_steps_paths_per_sec;
};

// Helper for statistics
BenchmarkStats compute_stats(const std::vector<double> &prices,
                             const std::vector<double> &times_ms, int N_steps,
                             int N_paths) {
  double sum_price = std::accumulate(prices.begin(), prices.end(), 0.0);
  double mean_price = sum_price / prices.size();

  double sq_sum = 0.0;
  for (double p : prices) {
    sq_sum += (p - mean_price) * (p - mean_price);
  }
  double std_price = std::sqrt(sq_sum / prices.size());

  double sum_time = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
  double mean_time = sum_time / times_ms.size(); // ms

  // Throughput: Total Steps / Time in Seconds
  double total_ops = (double)N_paths * (double)N_steps;
  double time_sec = mean_time / 1000.0;
  double throughput = (time_sec > 0) ? (total_ops / time_sec) : 0.0;

  return {mean_price, std_price, mean_time, throughput};
}

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
  // Paramètres benchmark "BOOSTED"
  // ================================
  const std::vector<int> N_steps_list = {50};
  const std::vector<int> N_paths_list = {100'000, 1'000'000};
  constexpr int N_REPEAT = 1;
  const std::string csv_file = "Benchmarks/benchmark_boosted.csv";

  // ================================
  // CSV
  // ================================
  std::ofstream csv(csv_file, std::ios::trunc);
  if (!csv.is_open()) {
    std::cerr << "[ERROR] Impossible d'ouvrir le CSV\n";
    return 1;
  }

  csv << "mode,N_steps,N_paths,price_mean,price_std,time_ms,throughput_ops_sec,"
         "diff_fdm\n";
  csv << std::fixed << std::setprecision(6);

  // =====================================================
  // 1. FDM SOLVERS (Baseline)
  // =====================================================
  std::cout << "==========================================\n";
  std::cout << "[INFO] Establishing FDM Baseline (Implicit Euler)...\n";
  std::cout << "==========================================\n";

  FiniteDifference fdm_impl(S0, K, r, sigma, T, 400, 400);
  double p_implicit = fdm_impl.price(FdmMethod::ImplicitEuler);
  std::cout << "[FDM Baseline] Price: " << p_implicit << "\n\n";

  // ================================
  // Warm-up GPU
  // ================================
#ifdef LSMC_ENABLE_CUDA
  std::cout << "[INFO] Warm-up GPU...\n";
  LSMC::priceAmericanPutGPU(S0, K, r, sigma, T, 50, 10'000);
#endif

  // ================================
  // Boucle benchmark
  // ================================
  // Iterate over Basis Types
  std::vector<RegressionBasis> bases = {
      RegressionBasis::Monomial, RegressionBasis::Hermite,
      RegressionBasis::Laguerre, RegressionBasis::Chebyshev,
      RegressionBasis::Cubic};

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
    else if (basis == RegressionBasis::Cubic)
      basis_name = "Cubic";
    std::cout << "Starting Benchmark for Basis: " << basis_name << "\n";

    for (int N_steps : N_steps_list) {
      for (int N_paths : N_paths_list) {
        std::cout << "------------------------------------------\n";
        std::cout << "Basis=" << basis_name << ", N_steps = " << N_steps
                  << ", N_paths = " << N_paths << "\n";

        BenchmarkStats stats_cpu = {0, 0, 0, 0};
        BenchmarkStats stats_omp = {0, 0, 0, 0};

        // -------------------------------------------------
        // CPU SEQUENTIEL (Force 1 thread)
        // -------------------------------------------------
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        { // Scope for vectors
          std::vector<double> prices_cpu;
          std::vector<double> times_cpu;
          prices_cpu.reserve(N_REPEAT);
          times_cpu.reserve(N_REPEAT);

          for (int k = 0; k < N_REPEAT; ++k) {
            auto t0 = std::chrono::high_resolution_clock::now();
            double p = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps,
                                              N_paths, basis);
            auto t1 = std::chrono::high_resolution_clock::now();
            prices_cpu.push_back(p);
            times_cpu.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
          }
          stats_cpu = compute_stats(prices_cpu, times_cpu, N_steps, N_paths);
        }

        csv << "CPU_Seq_" << basis_name << "," << N_steps << "," << N_paths
            << "," << stats_cpu.mean_price << "," << stats_cpu.std_price << ","
            << stats_cpu.mean_time_ms << ","
            << stats_cpu.throughput_steps_paths_per_sec << ","
            << (stats_cpu.mean_price - p_implicit) << "\n";

        std::cout << "[CPU Seq " << basis_name
                  << "]     Price = " << stats_cpu.mean_price
                  << " | Time = " << stats_cpu.mean_time_ms << " ms\n";

        // -------------------------------------------------
        // CPU OPENMP (Force Max threads)
        // -------------------------------------------------
#ifdef _OPENMP
        omp_set_num_threads(omp_get_max_threads());

        {
          std::vector<double> prices_omp;
          std::vector<double> times_omp;
          prices_omp.reserve(N_REPEAT);
          times_omp.reserve(N_REPEAT);

          for (int k = 0; k < N_REPEAT; ++k) {
            auto t0 = std::chrono::high_resolution_clock::now();
            double p = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps,
                                              N_paths, basis);
            auto t1 = std::chrono::high_resolution_clock::now();
            prices_omp.push_back(p);
            times_omp.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
          }
          stats_omp = compute_stats(prices_omp, times_omp, N_steps, N_paths);
        }

        csv << "CPU_OMP_" << basis_name << "," << N_steps << "," << N_paths
            << "," << stats_omp.mean_price << "," << stats_omp.std_price << ","
            << stats_omp.mean_time_ms << ","
            << stats_omp.throughput_steps_paths_per_sec << ","
            << (stats_omp.mean_price - p_implicit) << "\n";

        std::cout << "[CPU OMP " << basis_name
                  << "]     Price = " << stats_omp.mean_price
                  << " | Time = " << stats_omp.mean_time_ms << " ms"
                  << " | Speedup vs Seq = x"
                  << (stats_cpu.mean_time_ms / stats_omp.mean_time_ms) << "\n";
#endif

#ifdef LSMC_ENABLE_CUDA
        // -------------------------------------------------
        // GPU CUDA
        // -------------------------------------------------
        std::vector<double> prices_gpu;
        std::vector<double> times_gpu;
        prices_gpu.reserve(N_REPEAT);
        times_gpu.reserve(N_REPEAT);

        for (int k = 0; k < N_REPEAT; ++k) {
          auto t0 = std::chrono::high_resolution_clock::now();
          double p = LSMC::priceAmericanPutGPU(S0, K, r, sigma, T, N_steps,
                                               N_paths, basis);
          auto t1 = std::chrono::high_resolution_clock::now();
          prices_gpu.push_back(p);
          times_gpu.push_back(
              std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        BenchmarkStats stats_gpu =
            compute_stats(prices_gpu, times_gpu, N_steps, N_paths);

        csv << "GPU_" << basis_name << "," << N_steps << "," << N_paths << ","
            << stats_gpu.mean_price << "," << stats_gpu.std_price << ","
            << stats_gpu.mean_time_ms << ","
            << stats_gpu.throughput_steps_paths_per_sec << ","
            << (stats_gpu.mean_price - p_implicit) << "\n";

        std::cout << "[GPU " << basis_name
                  << "]         Price = " << stats_gpu.mean_price
                  << " | Time = " << stats_gpu.mean_time_ms << " ms"
                  << " | Speedup vs Seq = x"
                  << (stats_cpu.mean_time_ms / stats_gpu.mean_time_ms);

#ifdef _OPENMP
        std::cout << " | Speedup vs OMP = x"
                  << (stats_omp.mean_time_ms / stats_gpu.mean_time_ms);
#endif
        std::cout << "\n";
#endif
      }
    }
  }

  csv.close();
  std::cout << "\n[INFO] CSV écrit : " << csv_file << "\n";
  return 0;
}
