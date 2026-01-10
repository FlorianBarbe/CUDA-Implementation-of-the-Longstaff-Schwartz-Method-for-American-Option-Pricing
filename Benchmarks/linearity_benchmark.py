import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

EXE_PATH = "build/Release/P1RV_CUDA.exe"
OUTPUT_DIR = "Rapport/images"

def run_benchmark(n_paths, n_steps, mode='cpu', n_threads=1):
    try:
        # Args: [EXE, paths, steps, mode, threads (optional)]
        args = [EXE_PATH, str(n_paths), str(n_steps), mode]
        if mode == 'omp' or mode == 'cpu':
             args.append(str(n_threads))
            
        result = subprocess.run(
            args,
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        # Filter empty lines
        lines = [l for l in lines if l.strip()]
        if not lines:
             return None
        time_ms = float(lines[-1])
        return time_ms
    except Exception as e:
        print(f"Error running {n_paths} {n_steps} {mode}: {e}")
        return None

def plot_linearity_gpu():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_steps_fixed = 50
    # Larger range for GPU
    n_paths_fixed = 100000
    
    # 1. Varying N_paths (GPU)
    print(f"Benchmarking N_paths linearity (GPU)...")
    paths_list = np.linspace(10000, 1000000, 20, dtype=int)
    times_paths = []
    
    for p in paths_list:
        print(f"  Running paths={p}...")
        t = run_benchmark(p, n_steps_fixed, mode='gpu')
        times_paths.append(t)
    
    # Filter None
    valid_idxs = [i for i, v in enumerate(times_paths) if v is not None]
    if valid_idxs:
        paths_list = paths_list[valid_idxs]
        times_paths = [times_paths[i] for i in valid_idxs]
        
        plt.figure(figsize=(7, 5))
        plt.plot(paths_list, times_paths, 'o-', color='green', label='GPU (CUDA)')
        
        coef = np.polyfit(paths_list, times_paths, 1)
        poly = np.poly1d(coef)
        plt.plot(paths_list, poly(paths_list), '--k', label=f'Fit: y={coef[0]:.2e}x + {coef[1]:.2f}')
        
        plt.title(f"Temps GPU vs Nombre de trajectoires")
        plt.xlabel("Nombre de trajectoires")
        plt.ylabel("Temps (ms)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(f"{OUTPUT_DIR}/linearity_paths_gpu.png", dpi=300)
        plt.close()

    # 2. Varying N_steps (GPU)
    print(f"Benchmarking N_steps linearity (GPU)...")
    steps_list = np.linspace(10, 200, 20, dtype=int)
    times_steps = []
    
    for s in steps_list:
        print(f"  Running steps={s}...")
        t = run_benchmark(n_paths_fixed, s, mode='gpu')
        times_steps.append(t)
    
    # Filter None
    valid_idxs = [i for i, v in enumerate(times_steps) if v is not None]
    if valid_idxs:
        steps_list = steps_list[valid_idxs]
        times_steps = [times_steps[i] for i in valid_idxs]

        plt.figure(figsize=(7, 5))
        plt.plot(steps_list, times_steps, 's-', color='teal', label='GPU (CUDA)')
        
        coef = np.polyfit(steps_list, times_steps, 1)
        poly = np.poly1d(coef)
        plt.plot(steps_list, poly(steps_list), '--k', label=f'Fit: y={coef[0]:.2f}x + {coef[1]:.2f}')
        
        plt.title(f"Temps GPU vs Nombre de pas de temps")
        plt.xlabel("Nombre de pas")
        plt.ylabel("Temps (ms)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(f"{OUTPUT_DIR}/linearity_steps_gpu.png", dpi=300)
        plt.close()

def plot_linearity_omp():
    # Omitted for brevity since user only asked for GPU now, but keeping functionality
    pass # Assume already run or not needed immediately

if __name__ == "__main__":
    plot_linearity_gpu()
