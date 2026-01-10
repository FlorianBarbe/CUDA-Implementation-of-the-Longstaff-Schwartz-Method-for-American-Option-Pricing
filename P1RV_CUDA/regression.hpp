#pragma once
#include <vector>

class Regression {
public:
    // Calcule les coefficients d'une r�gression polynomiale (degr� 2)
    static std::vector<double> ols(const std::vector<double>& X, const std::vector<double>& Y);

    // �value la r�gression aux points donn�s
    static std::vector<double> predict(const std::vector<double>& X, const std::vector<double>& coeffs);
};
