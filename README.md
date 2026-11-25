# Valorisation d'options américaines par Longstaff–Schwartz (CPU, OpenMP, CUDA)

Implémentation en C++17, OpenMP et CUDA de la méthode de Monte-Carlo par moindres carrés de **Longstaff–Schwartz (LSMC)** pour la valorisation d'une option Put américaine, comparée à des solveurs déterministes par **différences finies (FDM)** sur l'équation de Black–Scholes–Merton.

## Problème abordé

Évaluer le prix d'une option Put américaine (exercice anticipé possible à des dates discrètes) sous dynamique géométrique brownienne $(S_0, K, r, \sigma, T)$. L'algorithme LSMC procède en deux phases :

1. **Simulation Monte-Carlo (forward)** : génération de $M$ trajectoires sur $N$ pas de temps (`gbm.cpp` / `gbm.cu`).
2. **Remontée arrière (*backward induction*)** : à chaque date d'exercice, sélection des trajectoires dans la monnaie (*in-the-money*), régression aux moindres carrés pour approcher la valeur de continuation, comparaison au payoff d'exercice immédiat et actualisation des flux (`lsmc.cpp` / `lsmc.cu`).

## Implémentations et bases de régression présentes

- **Séquentielle CPU et parallèle CPU (OpenMP)** : `P1RV_CUDA/lsmc.cpp`, `P1RV_CUDA/gbm.cpp` (contrôle du nombre de threads via `omp_set_num_threads` dans `P1RV_CUDA/main.cu` ; 1 thread pour le mode séquentiel).
- **Parallèle GPU (CUDA)** : `P1RV_CUDA/lsmc.cu`, `P1RV_CUDA/gbm.cu`, `P1RV_CUDA/minmax_kernel.cuh` (génération via `curand`, réductions GPU des sommes normales $A^\top A$ et $A^\top y$ en `double` jusqu'au degré 10 ; macro `LSMC_ENABLE_CUDA` définie dans `P1RV_CUDA/gbm.hpp`).
- **Référence déterministe 1D (FDM)** : `P1RV_CUDA/fdm.cpp`, `P1RV_CUDA/fdm.hpp` (schémas Explicite, Implicite et Runge–Kutta d'ordre 4).
- **Bases de régression définies dans `P1RV_CUDA/lsmc.hpp`** (`enum class RegressionBasis`) : `Monomial`, `Hermite`, `Laguerre`, `Chebyshev` et `Cubic`.
- **Scripts d'analyse et rapport** : graphiques et tableaux dans `Benchmarks/` (Python), rapport complet dans `Rapport/Rapport.pdf`.

## Contributions du projet

Projet académique réalisé en binôme à l'École Centrale de Nantes (projet P1RV, janvier 2026) par **Florian Barbe** et **Narjisse El Manssouri**. Le partage individuel des tâches n'étant pas documenté par fichier dans le dépôt, les fonctionnalités décrites ici correspondent aux réalisations collectives du projet (solveurs CPU/OpenMP/CUDA, solveurs FDM de référence, scripts de benchmark et rapport).

## Performances (résultats archivés)

Résultats traçables au tableau de comparaison de `Rapport/Rapport.tex` et au fichier `Benchmarks/benchmark_lsmc.csv` :

- **Matériel** : Intel Core i7-13620H + NVIDIA GeForce RTX 4060 (architecture Ada Lovelace, 3072 cœurs CUDA, 8 Go GDDR6).
- **Cas test** : Put américain ($S_0 = 100$, $K = 100$, $r = 0{,}05$, $\sigma = 0{,}2$, $T = 1$), $N = 50$ pas de temps.
- **Référence FDM (1000 pas)** : FDM Implicite `6,067` (`0,67 ms`), FDM Explicite `6,079` (`60,18 ms`), FDM RK4 `6,079` (`231,88 ms`).
- **Périmètre chronométré** : temps mural mesuré autour de l'appel au solveur (`std::chrono::high_resolution_clock` autour de `LSMC::priceAmericanPut` / `LSMC::priceAmericanPutGPU` dans `P1RV_CUDA/main.cu`, hors écriture CSV).

| Mode | Trajectoires ($M$) | Pas ($N$) | Prix estimé | Temps (ms) | Ratio vs CPU séq. |
| :--- | ---: | ---: | ---: | ---: | ---: |
| CPU Séquentiel | 100 000 | 50 | 6,057 | 559,91 | 1,0× |
| CPU OpenMP | 100 000 | 50 | 6,057 | 540,23 | ~1,04× |
| GPU CUDA | 100 000 | 50 | 6,070 | 41,96 | ~13,3× |
| CPU Séquentiel | 1 000 000 | 50 | 6,059 | 6 914,19 | 1,0× |
| CPU OpenMP | 1 000 000 | 50 | 6,059 | 6 562,99 | ~1,05× |
| GPU CUDA | 1 000 000 | 50 | 6,047 | 455,63 | ~15,2× |

*Note : d'autres fichiers CSV présents dans `Benchmarks/` (`benchmark_enriched.csv`, `benchmark_boosted_new.csv`, `table_output.tex`) correspondent à des campagnes et configurations distinctes (variations de pas de temps, bases ou degrés).*

## Dépendances et compilation

### Dépendances
- **CMake** >= 3.18
- Compilateur **C++17** avec support **OpenMP**
- **NVIDIA CUDA Toolkit** (`nvcc`, `cudart`, `curand`) compatible avec le GPU cible

### Compilation dans `build-varprox`

Dans `CMakeLists.txt`, `CMAKE_CUDA_ARCHITECTURES` est initialisé par défaut à `89` (Ada Lovelace, ex. RTX 4060) uniquement s'il n'est pas déjà défini. Cette valeur est à adapter à la *compute capability* du GPU cible (par exemple `75`, `86`, `89`) et aux architectures prises en charge par le toolkit CUDA installé :

```bash
cmake -S . -B build-varprox -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-varprox --config Release --parallel
```

*(Sous Linux, le binaire produit est `build-varprox/P1RV_CUDA` ; sous Windows/MSVC multi-config, `build-varprox/Release/P1RV_CUDA.exe`.)*

## Exemples d'exécution (arguments vérifiés dans `P1RV_CUDA/main.cu`)

Deux modes CLI sont disponibles dans `P1RV_CUDA/main.cu` :

1. **Mode drapeaux (`--`)** — utilise `RegressionBasis::Monomial` de degré 3, affiche `[DONE] Price=... Time=...ms` et exporte `resultats_lsmc.csv` (et `paths.csv` si `--dump_paths 1` en mode GPU, limité à 2 000 trajectoires) :

```bash
# GPU CUDA (100 000 trajectoires, 50 pas, blocs de 256 threads)
./build-varprox/P1RV_CUDA --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 \
  --N_steps 50 --N_paths 100000 --mode gpu --block_size 256

# CPU séquentiel (1 thread) ou OpenMP (--threads 4)
./build-varprox/P1RV_CUDA --N_steps 50 --N_paths 100000 --mode cpu --threads 1
```

2. **Mode positionnel** — `./build-varprox/P1RV_CUDA <N_paths> <N_steps> <mode> [threads|block_size]` (utilise `RegressionBasis::Monomial` de degré 2 et affiche uniquement la durée en ms) :

```bash
./build-varprox/P1RV_CUDA 100000 50 cpu 1
./build-varprox/P1RV_CUDA 100000 50 gpu 256
```

## Limites connues

- **Goulot séquentiel de la remontée arrière** : la simulation Monte-Carlo est massivement parallèle, mais la *backward induction* impose une synchronisation globale à chaque pas de temps pour résoudre le système normal de régression (loi d'Amdahl).
- **Pertinence en dimension 1** : pour un seul sous-jacent, le solveur FDM implicite (`0,67 ms`) reste nettement plus rapide que LSMC ; l'intérêt du couplage LSMC/GPU concerne surtout l'extension à plusieurs actifs en haute dimension.
- **Précision et mémoire GPU** : les trajectoires GPU sont stockées en simple précision (`float`), tandis que les sommes normales sont accumulées en double précision (`double`) ; le stockage complet $M \times (N+1)$ borne la taille des simulations à la mémoire GPU disponible et les polynômes de degré élevé restent sensibles au conditionnement.
- **État actuel de `P1RV_CUDA/main.cu`** : les options CLI ne permettent pas encore de choisir la base ou le degré depuis la ligne de commande (fixés à `Monomial`, degré 3 en mode `--` et degré 2 en mode positionnel), l'argument `--seed` est lu mais non transmis au solveur, et la boucle du mode benchmark par défaut (sans argument) est actuellement tronquée dans `main.cu`.
- **Artefacts historiques** : le dépôt conserve un dossier `build/` issu de la compilation Windows/Visual Studio utilisée lors de la soutenance ; utiliser un dossier séparé (`build-varprox`) pour toute nouvelle compilation.
