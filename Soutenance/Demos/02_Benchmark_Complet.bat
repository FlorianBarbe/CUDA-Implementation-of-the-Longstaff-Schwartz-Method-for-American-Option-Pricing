@echo off
title P1RV - Benchmark Complet
color 0E
echo ===================================================
echo           BENCHMARK COMPLET (Auto)
echo ===================================================
echo.
echo Ce script lance le mode de benchmark automatique.
echo Il va tester :
echo - Bases : Monome, Hermite, Laguerre, Chebyshev
echo - Degres : 1 a 10
echo - Modes : CPU, OpenMP, GPU
echo.
echo Les resultats seront enregistres dans Benchmarks/benchmark_degree_precision.csv
echo.
echo Appuyez sur une touche pour demarrer...
pause
P1RV_CUDA.exe
echo.
echo Fin du benchmark. Verifiez le fichier CSV.
pause
