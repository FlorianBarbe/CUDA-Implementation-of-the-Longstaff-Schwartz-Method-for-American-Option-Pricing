#include "fdm.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>


FiniteDifference::FiniteDifference(double S0_, double K_, double r_,
                                   double sigma_, double T_, int M_, int N_,
                                   double Smax_mult_)
    : S0(S0_), K(K_), r(r_), sigma(sigma_), T(T_), M(M_), N(N_),
      Smax_mult(Smax_mult_) {}

double FiniteDifference::price(FdmMethod method) {
  switch (method) {
  case FdmMethod::ExplicitEuler:
    return solveExplicit();
  case FdmMethod::ImplicitEuler:
    return solveImplicit();
  case FdmMethod::RungeKutta4:
    return solveRK4();
  default:
    return 0.0;
  }
}

// ==========================================================
// Utilitaires
// ==========================================================
// Interpolation linéaire pour trouver la valeur S0 dans la grille
static double interpolate_price(double S_target, double dS,
                                const std::vector<double> &V) {
  // S_i = i * dS
  // i_target = S_target / dS
  double i_exact = S_target / dS;
  int i_low = static_cast<int>(std::floor(i_exact));
  int i_high = i_low + 1;

  if (i_low < 0)
    return V[0];
  if (i_high >= (int)V.size())
    return V.back();

  double w = i_exact - i_low;
  return (1.0 - w) * V[i_low] + w * V[i_high];
}

// ==========================================================
// 1. EULER EXPLICITE
// ==========================================================
double FiniteDifference::solveExplicit() {
  double Smax = Smax_mult * K;
  double dS = Smax / M;
  double dt = T / N;

  // Vecteur valeurs V[i] correspond à S[i] = i*dS
  // i va de 0 à M
  std::vector<double> V(M + 1);
  std::vector<double> V_new(M + 1);

  // Condition initiale (Maturité, t=T => tau=0)
  for (int i = 0; i <= M; ++i) {
    double S = i * dS;
    V[i] = std::max(K - S, 0.0);
  }

  // Boucle temps (tau = 0 -> T)
  for (int n = 0; n < N; ++n) {
    // Conditions aux limites
    V_new[0] = K * std::exp(-r * (n + 1) * dt); // S=0
    V_new[M] = 0.0;                             // S=Smax (Put -> 0)

    // Intérieur
    for (int i = 1; i < M; ++i) {
      double S = i * dS;

      // Discrétisation Black-Scholes (Backward en temps réel / Forward en tau)
      // dV/dtau = r*S*dV/dS + 0.5*sigma^2*S^2*d2V/dS2 - r*V

      double delta = (V[i + 1] - V[i - 1]) / (2.0 * dS);
      double gamma = (V[i + 1] - 2.0 * V[i] + V[i - 1]) / (dS * dS);

      double theta =
          r * S * delta + 0.5 * sigma * sigma * S * S * gamma - r * V[i];

      double val = V[i] + dt * theta;

      // Put Américain : max(val, intrinsic)
      V_new[i] = std::max(val, std::max(K - S, 0.0));
    }
    V = V_new;
  }

  return interpolate_price(S0, dS, V);
}

// ==========================================================
// 2. EULER IMPLICITE
// ==========================================================
// Résolution système Ax=b tridiagonal
static void thomas_algorithm(int size,
                             const std::vector<double> &a, // lower
                             const std::vector<double> &b, // diag
                             const std::vector<double> &c, // upper
                             std::vector<double> &d,       // rhs -> solution
                             std::vector<double> &c_prime) // buffer
{
  // size = nombre d'inconnues (indices 1 à M-1)
  // Ici on adapte pour indices 0 à size-1 correspondant à i=1..M-1

  // Forward
  c_prime[0] = c[0] / b[0];
  d[0] = d[0] / b[0];

  for (int i = 1; i < size; i++) {
    double temp = b[i] - a[i] * c_prime[i - 1];
    c_prime[i] = c[i] / temp;
    d[i] = (d[i] - a[i] * d[i - 1]) / temp;
  }

  // Back substitution
  for (int i = size - 2; i >= 0; i--) {
    d[i] = d[i] - c_prime[i] * d[i + 1];
  }
}

double FiniteDifference::solveImplicit() {
  double Smax = Smax_mult * K;
  double dS = Smax / M;
  double dt = T / N;

  std::vector<double> V(M + 1);

  // Init Payoff
  for (int i = 0; i <= M; ++i) {
    double S = i * dS;
    V[i] = std::max(K - S, 0.0);
  }

  // Coefficients constants du système linéaire
  // On résout (I - dt*L) V^{n+1} = V^n
  // Equation i: alpha*V_{i-1} + beta*V_i + gamma*V_{i+1} = V^n_i
  // L V_i = r*S*delta + 0.5*sigma^2*S^2*gamma - r*V

  // Coefficients tridiagonaux (pour i=1..M-1)
  int dim = M - 1;
  std::vector<double> lower(dim), diag(dim), upper(dim);
  std::vector<double> rhs(dim);
  std::vector<double> c_prime(dim); // buffer thomas

  for (int i = 1; i < M; ++i) {
    double S = i * dS;
    double i2 = (double)i * i;

    // Coeffs de l'opérateur discret L (sans dt)
    // a_i V_{i-1} + b_i V_i + c_i V_{i+1}
    double l_i = 0.5 * sigma * sigma * i2 - 0.5 * r * i; // V_{i-1}
    double d_i = -sigma * sigma * i2 - r;                // V_i
    double u_i = 0.5 * sigma * sigma * i2 + 0.5 * r * i; // V_{i+1}

    // Système implicite: V^{n+1} - dt * L V^{n+1} = V^n
    // (1 - dt*d_i) V_i - dt*l_i V_{i-1} - dt*u_i V_{i+1} = V^n_i
    // Attention aux signes pour Thomas : A_i u_{i-1} + B_i u_i + C_i u_{i+1} =
    // D_i

    int idx = i - 1;
    lower[idx] = -dt * l_i;
    diag[idx] = 1.0 - dt * d_i;
    upper[idx] = -dt * u_i;
  }

  // Boucle temps
  for (int n = 0; n < N; ++n) {
    // Préparer RHS
    // Boundary conditions courantes pour le pas n+1
    double V_boundary_0 = K * std::exp(-r * (n + 1) * dt);
    double V_boundary_M = 0.0;

    for (int i = 1; i < M; ++i) {
      rhs[i - 1] = V[i];
    }

    // Ajustement BC dans RHS
    // i=1 : terme lower dépend de V[0]
    rhs[0] -= lower[0] * V_boundary_0;
    // i=M-1 : terme upper dépend de V[M]
    rhs[dim - 1] -= upper[dim - 1] * V_boundary_M;

    // On modifie les vecteurs a,b,c car Thomas les modifie parfois (ici
    // implémente copie locale des vecteurs) Pour optimisation, on réutilise.
    // Ici on copie pour clarté.
    std::vector<double> a = lower;
    std::vector<double> b = diag;
    std::vector<double> c = upper;

    // Résolution
    std::vector<double> sol = rhs;
    thomas_algorithm(dim, a, b, c, sol, c_prime);

    // MAJ V et application condition Américaine
    V[0] = V_boundary_0;
    V[M] = V_boundary_M;
    for (int i = 1; i < M; ++i) {
      double val = sol[i - 1];
      double S = i * dS;
      // Operator Splitting pour Américain (approx)
      V[i] = std::max(val, std::max(K - S, 0.0));
    }
  }

  return interpolate_price(S0, dS, V);
}

// ==========================================================
// 3. RUNGE-KUTTA 4
// ==========================================================
// Evaluation de L(V)
static std::vector<double> evaluate_operator(const std::vector<double> &V,
                                             double r, double sigma, double dS,
                                             int M) {
  // L(V) sur l'intérieur i=1..M-1
  // Les bords sont gérés via Dirichlet fixes pour L (approx)
  std::vector<double> LV(M + 1, 0.0);

  for (int i = 1; i < M; ++i) {
    double S = i * dS;

    double delta = (V[i + 1] - V[i - 1]) / (2.0 * dS);
    double gamma = (V[i + 1] - 2.0 * V[i] + V[i - 1]) / (dS * dS);

    LV[i] = r * S * delta + 0.5 * sigma * sigma * S * S * gamma - r * V[i];
  }
  return LV;
}

double FiniteDifference::solveRK4() {
  double Smax = Smax_mult * K;
  double dS = Smax / M;
  double dt = T / N;

  std::vector<double> V(M + 1);
  for (int i = 0; i <= M; ++i)
    V[i] = std::max(K - i * dS, 0.0);

  for (int n = 0; n < N; ++n) {
    // BC temporelle (approx: on impose les valeurs aux bords à chaque
    // sous-étape ou à la fin) Ici on fixe les bords à la valeur analytique pour
    // l'étape n (ou n+1)
    double bc0 = K * std::exp(-r * n * dt);
    double bcM = 0.0;

    // k1 = dt * L(V)
    auto L1 = evaluate_operator(V, r, sigma, dS, M);
    std::vector<double> V1 = V;
    for (int i = 1; i < M; ++i)
      V1[i] += 0.5 * dt * L1[i];
    V1[0] = bc0;
    V1[M] = bcM; // BC simple

    // k2 = dt * L(V + 0.5*k1)
    auto L2 = evaluate_operator(V1, r, sigma, dS, M);
    std::vector<double> V2 = V;
    for (int i = 1; i < M; ++i)
      V2[i] += 0.5 * dt * L2[i];
    V2[0] = bc0;
    V2[M] = bcM;

    // k3 = dt * L(V + 0.5*k2)
    auto L3 = evaluate_operator(V2, r, sigma, dS, M);
    std::vector<double> V3 = V;
    for (int i = 1; i < M; ++i)
      V3[i] += dt * L3[i];
    V3[0] = bc0;
    V3[M] = bcM;

    // k4 = dt * L(V + k3)
    auto L4 = evaluate_operator(V3, r, sigma, dS, M);

    // V_new
    double bc0_next = K * std::exp(-r * (n + 1) * dt);

    V[0] = bc0_next;
    V[M] = 0.0;
    for (int i = 1; i < M; ++i) {
      double val =
          V[i] + (dt / 6.0) * (L1[i] + 2.0 * L2[i] + 2.0 * L3[i] + L4[i]);
      double S = i * dS;
      V[i] = std::max(val, std::max(K - S, 0.0));
    }
  }

  return interpolate_price(S0, dS, V);
}
