## Compilation et exécution

Ce projet implémente la méthode de Monte Carlo de Longstaff–Schwartz pour le pricing d’options américaines,
avec une accélération GPU via CUDA.

### Prérequis

- CMake ≥ 3.20
- Compilateur compatible C++17
- GPU NVIDIA compatible CUDA
- CUDA Toolkit ≥ 11.x
- Python 3 (optionnel, pour les graphiques de benchmark)

### Compilation (recommandé : build hors source)

```bash
git clone https://github.com/<username>/CUDA-Implementation-of-the-Longstaff-Schwartz-Method-for-American-Option-Pricing.git

cd CUDA-Implementation-of-the-Longstaff-Schwartz-Method-for-American-Option-Pricing

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

Après compilation, l’exécutable est généré dans le dossier build/.

./P1RV_CUDA


(Sous Windows : P1RV_CUDA.exe.)

Compilation sans CUDA (CPU uniquement)

Si CUDA n’est pas disponible, il est possible de compiler le projet sans accélération GPU :

cmake .. -DENABLE_CUDA=OFF
cmake --build .

Benchmarks

Les résultats de performance sont exportés sous forme de fichiers CSV (benchmark.csv, benchmark_lsmc.csv).

Pour générer les graphiques de benchmark :

python plot_benchmark.py

Remarques

L’exécution CUDA n’est pas testée en intégration continue (CI), les runners GitHub ne disposant pas de GPU.

La compilation et l’exécution CUDA doivent être réalisées sur une machine locale équipée d’un GPU NVIDIA.


## Build and Run

This project implements the Longstaff–Schwartz Monte Carlo method for pricing American options,
with a CUDA-accelerated version.

### Requirements

- CMake ≥ 3.20
- C++17 compatible compiler
- NVIDIA GPU with CUDA support
- CUDA Toolkit ≥ 11.x
- Python 3 (optional, for plotting benchmarks)

### Build (recommended: out-of-source)

```bash
git clone https://github.com/<username>/CUDA-Implementation-of-the-Longstaff-Schwartz-Method-for-American-Option-Pricing.git
cd CUDA-Implementation-of-the-Longstaff-Schwartz-Method-for-American-Option-Pricing

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
