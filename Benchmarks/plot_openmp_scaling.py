
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

def run_benchmark(threads, exe_path):
    cmd = [exe_path, str(N_PATHS), str(N_STEPS), MODE, str(threads)]
    try:
        # Run command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse output (last line should be time in ms)
        lines = result.stdout.strip().split('\n')
        # The main.cu prints "[DEBUG]..." then maybe other stuff, but eventually "dt" on a single line
        # Based on main.cu code:
        # std::cout << dt << std::endl;
        # We look for the last non-empty line
        time_ms = float(lines[-1])
        return time_ms
    except Exception as e:
        print(f"Error running threads={threads}: {e}")
        return None

def main():
    exe_path = EXECUTABLE_PATH
    if not os.path.exists(exe_path):
        print(f"Executable not found at {exe_path}")
        # Try finding it relative to script if run from Benchmarks dir
        alt_path = os.path.join("..", "build", "Release", "P1RV_CUDA.exe")
        if os.path.exists(alt_path):
             exe_path = alt_path
             print(f"Found executable at {exe_path}")
        else:
            print(f"Error: Could not find P1RV_CUDA.exe in {exe_path} or {alt_path}")
            return

    print(f"Running OpenMP scaling benchmark for N={N_PATHS}, Steps={N_STEPS}...")
    
    threads_list = range(1, MAX_THREADS + 1)
    times = []
    
    for t in threads_list:
        time_ms = run_benchmark(t, exe_path)
        if time_ms is not None:
            print(f"Threads: {t}, Time: {time_ms:.4f} ms")
            times.append(time_ms)
        else:
            times.append(float('nan'))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(threads_list, times, marker='o', linestyle='-', color='b', label='OpenMP Execution Time')
    plt.title(f'OpenMP Scaling ($N_{{paths}}={N_PATHS}, N_{{steps}}={N_STEPS}$)')
    plt.xlabel('Number of Threads (Cores)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xticks(threads_list)
    plt.legend()
    
    # Check if directory exists
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Plot saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
