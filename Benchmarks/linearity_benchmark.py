import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

EXE_PATH = "build/Release/P1RV_CUDA.exe"
OUTPUT_DIR = "Rapport/images"

def run_benchmark(n_paths, n_steps, n_threads=1):
    try:
        # Run executable with arguments (optional third arg: n_threads)
        args = [EXE_PATH, str(n_paths), str(n_steps)]
        if n_threads > 1:
            args.append(str(n_threads))
            
        result = subprocess.run(
            args,
            capture_output=True, text=True, check=True
        )
        # Parse last line which should be the time in ms
        lines = result.stdout.strip().split('\n')
        time_ms = float(lines[-1])
        return time_ms
    except Exception as e:
        print(f"Error running {n_paths} {n_steps} (threads={n_threads}): {e}")
        return None

def plot_linearity_omp():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_threads = 4
    n_steps_fixed = 50
    n_paths_fixed = 100000
    
    # 1. Varying N_paths (OpenMP)
    print(f"Benchmarking N_paths linearity (OpenMP {n_threads} threads)...")
    paths_list = np.linspace(10000, 500000, 20, dtype=int)
    times_paths = []
    
    for p in paths_list:
        print(f"  Running paths={p}...")
        t = run_benchmark(p, n_steps_fixed, n_threads)
        times_paths.append(t)
        
    plt.figure(figsize=(7, 5))
    plt.plot(paths_list, times_paths, 'o-', color='purple', label=f'OpenMP ({n_threads} threads)')
    
    coef = np.polyfit(paths_list, times_paths, 1)
    poly = np.poly1d(coef)
    plt.plot(paths_list, poly(paths_list), '--k', label=f'Fit: y={coef[0]:.2e}x + {coef[1]:.2f}')
    
    plt.title(f"Temps OpenMP ({n_threads} threads) vs N_paths")
    plt.xlabel("Nombre de trajectoires")
    plt.ylabel("Temps (ms)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/linearity_paths_omp.png", dpi=300)
    plt.close()

    # 2. Varying N_steps (OpenMP)
    print(f"Benchmarking N_steps linearity (OpenMP {n_threads} threads)...")
    steps_list = np.linspace(10, 200, 20, dtype=int)
    times_steps = []
    
    for s in steps_list:
        print(f"  Running steps={s}...")
        t = run_benchmark(n_paths_fixed, s, n_threads)
        times_steps.append(t)
        
    plt.figure(figsize=(7, 5))
    plt.plot(steps_list, times_steps, 's-', color='orange', label=f'OpenMP ({n_threads} threads)')
    
    coef = np.polyfit(steps_list, times_steps, 1)
    poly = np.poly1d(coef)
    plt.plot(steps_list, poly(steps_list), '--k', label=f'Fit: y={coef[0]:.2f}x + {coef[1]:.2f}')
    
    plt.title(f"Temps OpenMP ({n_threads} threads) vs N_steps")
    plt.xlabel("Nombre de pas")
    plt.ylabel("Temps (ms)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/linearity_steps_omp.png", dpi=300)
    plt.close()

def plot_linearity():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Varying N_paths (fixed N_steps=50)
    print("Benchmarking N_paths linearity...")
    n_steps_fixed = 50
    paths_list = np.linspace(10000, 500000, 20, dtype=int)
    times_paths = []
    
    for p in paths_list:
        print(f"  Running paths={p}...")
        t = run_benchmark(p, n_steps_fixed)
        times_paths.append(t)
        
    plt.figure(figsize=(7, 5))
    plt.plot(paths_list, times_paths, 'o-', color='blue', label=f'N_steps={n_steps_fixed}')
    
    # Linear fit
    coef = np.polyfit(paths_list, times_paths, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(paths_list, poly1d_fn(paths_list), '--k', label=f'Fit: y={coef[0]:.2e}x + {coef[1]:.2f}')
    
    plt.title("Temps d'exécution CPU vs Nombre de trajectoires (Séquentiel)")
    plt.xlabel("Nombre de trajectoires ($N_{paths}$)")
    plt.ylabel("Temps (ms)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/linearity_paths.png", dpi=300)
    plt.close()
    
    # 2. Varying N_steps (fixed N_paths=100_000)
    print("Benchmarking N_steps linearity...")
    n_paths_fixed = 100000
    steps_list = np.linspace(10, 200, 20, dtype=int)
    times_steps = []
    
    for s in steps_list:
        print(f"  Running steps={s}...")
        t = run_benchmark(n_paths_fixed, s)
        times_steps.append(t)
        
    plt.figure(figsize=(7, 5))
    plt.plot(steps_list, times_steps, 's-', color='red', label=f'N_paths={n_paths_fixed}')
    
    # Linear fit
    coef = np.polyfit(steps_list, times_steps, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(steps_list, poly1d_fn(steps_list), '--k', label=f'Fit: y={coef[0]:.2f}x + {coef[1]:.2f}')
    
    plt.title("Temps d'exécution CPU vs Nombre de pas de temps (Séquentiel)")
    plt.xlabel("Nombre de pas ($N_{steps}$)")
    plt.ylabel("Temps (ms)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/linearity_steps.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # plot_linearity() # CPU already done
    plot_linearity_omp()
