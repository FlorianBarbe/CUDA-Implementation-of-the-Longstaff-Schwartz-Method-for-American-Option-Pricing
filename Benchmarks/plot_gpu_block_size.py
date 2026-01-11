
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
EXECUTABLE_PATH = os.path.join("build", "Release", "P1RV_CUDA.exe")
N_PATHS = 1000000  # 1M paths for meaningful GPU benchmark
N_STEPS = 50
MODE = "gpu"
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
N_TRIALS = 10
OUTPUT_IMAGE = os.path.join("..", "Rapport", "images", "gpu_block_size_scaling.png")

def run_benchmark_average(block_size, exe_path, n_trials=5):
    times = []
    for i in range(n_trials):
        cmd = [exe_path, str(N_PATHS), str(N_STEPS), MODE, str(block_size)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            t = float(lines[-1])
            times.append(t)
        except Exception as e:
            print(f"Error (run {i+1}): {e}")
    
    if len(times) < 3:
        return np.mean(times) if times else None
    
    # Trimmed mean
    times.remove(min(times))
    times.remove(max(times))
    
    return np.mean(times)

def main():
    exe_path = EXECUTABLE_PATH
    if not os.path.exists(exe_path):
        alt_path = os.path.join("..", "build", "Release", "P1RV_CUDA.exe")
        if os.path.exists(alt_path):
             exe_path = alt_path
             print(f"Found executable at {exe_path}")
        else:
            print(f"Error: Could not find P1RV_CUDA.exe")
            return

    print(f"Running GPU Block Size Benchmark (Avg of {N_TRIALS} runs, trimmed) for N={N_PATHS}, Steps={N_STEPS}...")
    
    times = []
    
    for bs in BLOCK_SIZES:
        avg_time = run_benchmark_average(bs, exe_path, n_trials=N_TRIALS)
        if avg_time is not None:
            print(f"Block Size: {bs}, Time: {avg_time:.4f} ms")
            times.append(avg_time)
        else:
            times.append(float('nan'))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar([str(bs) for bs in BLOCK_SIZES], times, color='green', edgecolor='black')
    plt.title(f'GPU Block Size Impact ($N_{{paths}}={N_PATHS}, N_{{steps}}={N_STEPS}$)')
    plt.xlabel('Block Size (Threads per Block)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, axis='y', alpha=0.5)
    
    # Annotate bars with values
    for i, (bs, t) in enumerate(zip(BLOCK_SIZES, times)):
        if not np.isnan(t):
            plt.annotate(f'{t:.1f}', xy=(i, t), ha='center', va='bottom', fontsize=9)
    
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Plot saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
