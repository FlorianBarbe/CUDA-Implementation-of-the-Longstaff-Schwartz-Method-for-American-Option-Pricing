// gbm.cpp
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <string>

#include "gbm.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

GBM::GBM(double S0_, double r_, double sigma_, double T_, int N_steps_)
    : S0(S0_), r(r_), sigma(sigma_), T(T_), N_steps(N_steps_) {
}

// Simulation complète (utile pour tests/unitaires)
std::vector<double> GBM::simulate(RNG& rng)
{
    std::vector<double> path(static_cast<size_t>(N_steps) + 1);
    simulate_path(rng, path.data());
    return path;
}

// Simulation d'une trajectoire dans un buffer préalloué (perf)
void GBM::simulate_path(RNG& rng, double* path_out)
{
    const double dt = T / static_cast<double>(N_steps);
    const double drift = (r - 0.5 * sigma * sigma) * dt;
    const double vol_dt = sigma * std::sqrt(dt);

    double S = S0;
    path_out[0] = S;

    for (int t = 1; t <= N_steps; ++t)
    {
        const double Z = rng.normal();
        S *= std::exp(drift + vol_dt * Z);
        path_out[t] = S;
    }
}

// Simulation de plusieurs trajectoires : layout contigu (path-major)
void GBM::simulatePaths(double* paths,
    double S0, double r, double sigma, double T,
    int N_steps, int N_paths)
{
    const double dt = T / static_cast<double>(N_steps);
    const double drift = (r - 0.5 * sigma * sigma) * dt;
    const double vol_dt = sigma * std::sqrt(dt);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N_paths; ++i)
    {
        // Seed déterministe par trajectoire (reproductible)
        RNG rng(i + 1234);

        double* path_ptr = paths + static_cast<size_t>(i) * (static_cast<size_t>(N_steps) + 1);

        double S = S0;
        path_ptr[0] = S;

        for (int t = 1; t <= N_steps; ++t)
        {
            const double Z = rng.normal();
            S *= std::exp(drift + vol_dt * Z);
            path_ptr[t] = S;
        }
    }
}

// Export CSV (écriture atomique via .tmp puis rename)
void GBM::exportCSV(const std::vector<std::vector<double>>& paths,
    const std::string& filename)
{
    if (paths.empty()) {
        std::cerr << "[ERREUR] exportCSV: paths est vide\n";
        return;
    }

    const std::string tmpname = filename + ".tmp";

    std::ofstream f(tmpname, std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "[ERREUR] Impossible d'ouvrir le fichier temporaire : " << tmpname << "\n";
        return;
    }

    const size_t cols = paths[0].size();
    f << std::fixed << std::setprecision(6);

    for (size_t p = 0; p < paths.size(); ++p)
    {
        if (paths[p].size() != cols) {
            std::cerr << "[AVERTISSEMENT] Ligne " << p << " ignorée (taille incorrecte)\n";
            continue;
        }

        for (size_t j = 0; j < cols; ++j) {
            f << paths[p][j];
            if (j + 1 < cols) f << ",";
        }
        f << "\n";
    }

    f.flush();
    f.close();

    try {
        std::filesystem::rename(tmpname, filename);
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[INFO] Rename impossible, tentative de suppression : " << e.what() << "\n";
        std::error_code ec;
        std::filesystem::remove(filename, ec);
        std::filesystem::rename(tmpname, filename);
    }
}
