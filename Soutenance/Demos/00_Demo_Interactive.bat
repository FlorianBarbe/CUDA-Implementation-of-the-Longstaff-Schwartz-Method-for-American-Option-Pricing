@echo off
title Demo P1RV - Mode Interactif
color 0A
echo ===================================================
echo      DEMONSTRATION P1RV - IA AVANT DEEP LEARNING
echo ===================================================
echo.
echo Ce script lance le simulateur P1RV_CUDA.exe avec vos parametres.
echo.

:ask_paths
set /p paths="Nombre de trajectoires (ex: 100000) : "
if "%paths%"=="" goto ask_paths

:ask_steps
set /p steps="Nombre de pas de temps (ex: 50) : "
if "%steps%"=="" set steps=50

:ask_mode
echo Modes disponibles : cpu, omp, gpu
set /p mode="Mode (cpu/omp/gpu) : "
if "%mode%"=="" set mode=gpu

:ask_params
set param=1
if "%mode%"=="gpu" (
    set /p param="Taille de bloc GPU (defaut 256) : "
    if "%param%"=="" set param=256
) else (
    set /p param="Nombre de threads CPU (defaut 4) : "
    if "%param%"=="" set param=4
)

echo.
echo Lancement de la simulation...
echo Commande : P1RV_CUDA.exe %paths% %steps% %mode% %param%
echo.
echo ---------------------------------------------------
P1RV_CUDA.exe %paths% %steps% %mode% %param%
echo ---------------------------------------------------
echo.
pause
goto ask_paths
