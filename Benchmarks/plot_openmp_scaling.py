
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Configuration
EXECUTABLE_PATH = os.path.join("build", "Release", "P1RV_CUDA.exe")
N_PATHS = 1000
N_STEPS = 50
MODE = "omp"
MAX_THREADS = 16
OUTPUT_IMAGE = os.path.join("..", "Rapport", "images", "openmp_scaling_1000.png")

def run_benchmark_average(threads, exe_path, n_trials=10):
    times = []
    for i in range(n_trials):
        cmd = [exe_path, str(N_PATHS), str(N_STEPS), MODE, str(threads)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            t = float(lines[-1])
            times.append(t)
        except Exception as e:
            print(f"Error (run {i+1}): {e}")
    
    if len(times) < 3:
        return np.mean(times) if times else None
    
    # Remove min and max (trimmed mean) to reduce noise
    times.remove(min(times))
    times.remove(max(times))
    
    avg_time = np.mean(times)
    return avg_time

def main():
    exe_path = EXECUTABLE_PATH
    if not os.path.exists(exe_path):
        print(f"Executable not found at {exe_path}")
        alt_path = os.path.join("..", "build", "Release", "P1RV_CUDA.exe")
        if os.path.exists(alt_path):
             exe_path = alt_path
             print(f"Found executable at {exe_path}")
        else:
            print(f"Error: Could not find P1RV_CUDA.exe in {exe_path} or {alt_path}")
            return

    print(f"Running OpenMP scaling benchmark (Avg of 10 runs, trimmed) for N={N_PATHS}, Steps={N_STEPS}...")
    
    threads_list = range(1, MAX_THREADS + 1)
    times = []
    
    for t in threads_list:
        avg_time = run_benchmark_average(t, exe_path, n_trials=10)
        if avg_time is not None:
            print(f"Threads: {t}, Time: {avg_time:.4f} ms")
            times.append(avg_time)
        else:
            times.append(float('nan'))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(threads_list, times, marker='o', linestyle='-', color='b', label='Avg Execution Time (Trimmed)')
    plt.title(f'OpenMP Scaling ($N_{{paths}}={N_PATHS}, N_{{steps}}={N_STEPS}$)')
    plt.xlabel('Number of Threads (Cores)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xticks(threads_list)
    plt.legend()
    
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Plot saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
