@echo off
title P1RV - CPU vs GPU Showdown
color 0B
echo ===================================================
echo         DUEL : CPU (Sequentiel) vs GPU (CUDA)
echo ===================================================
echo.
echo Nous allons pricer une Option Americaine Put.
echo Parametres : S0=100, K=100, T=1.0, r=5%%, sigma=20%%
echo Pas de temps : 50
echo Trajectoires : 1,000,000 (UN MILLION)
echo.
echo 1. Lancement sur CPU (Sequentiel)...
echo (Cela peut prendre quelques secondes...)
echo.
P1RV_CUDA.exe 1000000 50 cpu 1
echo.
echo ---------------------------------------------------
echo.
echo 2. Lancement sur GPU (CUDA)...
echo (Attention les yeux...)
echo.
P1RV_CUDA.exe 1000000 50 gpu 256
echo.
echo ---------------------------------------------------
echo.
echo Le GPU devrait etre environ 15x plus rapide !
echo.
pause
