#pragma once

#include <vector>
#include <string>

enum class FdmMethod {
    ExplicitEuler,
    ImplicitEuler,
    RungeKutta4
};

class FiniteDifference {
public:
    // Paramètres
    double S0;
    double K;
    double r;
    double sigma;
    double T;
    
    // Paramètres Grille
    int M;      // Nombre de pas d'espace (S)
    int N;      // Nombre de pas de temps (t)
    double Smax_mult; // Multiplicateur pour Smax (ex: 3*K ou 4*K)

    FiniteDifference(double S0, double K, double r, double sigma, double T, 
                     int M = 100, int N = 100, double Smax_mult = 4.0);

    // Résolution
    double price(FdmMethod method);

private:
    double solveExplicit();
    double solveImplicit();
    double solveRK4();

    // Helpers
    std::vector<double> coeff_lower;
    std::vector<double> coeff_diag;
    std::vector<double> coeff_upper;
};
