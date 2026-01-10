import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load CSV
try:
    df = pd.read_csv('Benchmarks/benchmark_degree_precision.csv')
    print("Columns:", df.columns)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Check if Mode exists
if 'Mode' in df.columns:
    modes = ['CPU', 'OMP', 'GPU']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
else:
    # Fallback for GPU only (Old format)
    modes = ['GPU']
    df['Mode'] = 'GPU' # Add Mode column for consistency
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    axes = [axes] # List for consistent indexing

# Filter Degree >= 2
df = df[df['Degree'] >= 2]

# Calculate Reference RK4 Price
df['Ref_RK4'] = df['Price'] - df['Diff_RK4']
df['RelativeError'] = df['Diff_RK4'].abs() / df['Ref_RK4']

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# Get unique Paths
paths_list = sorted(df['Paths'].unique())
modes = ['CPU', 'OMP', 'GPU']
titles_mode = {'CPU': 'CPU Seq', 'OMP': 'CPU OpenMP', 'GPU': 'GPU CUDA'}

# Create subplots: Rows = Paths, Cols = Modes
n_rows = len(paths_list)
n_cols = len(modes)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey='row')

for r, n_paths in enumerate(paths_list):
    for c, mode in enumerate(modes):
        ax = axes[r, c] if n_rows > 1 else axes[c]
        
        data = df[(df['Mode'] == mode) & (df['Paths'] == n_paths)]
        
        if data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center')
            continue

        sns.lineplot(data=data, x='Degree', y='RelativeError', hue='Base', marker='o', ax=ax, linewidth=2)
        
        ax.set_title(f"{titles_mode[mode]} (N={n_paths})", fontweight='bold')
        ax.set_xlabel("Degré")
        if c == 0:
            ax.set_ylabel("Erreur Relative |LSMC - RK4| / RK4")
        else:
            ax.set_ylabel("")
            
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--")
        ax.set_xticks(range(2, 11))
        
        # Legend only on last column or remove duplicates
        if c < n_cols - 1:
             ax.get_legend().remove()
        else:
             ax.legend(title='Base', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('Benchmarks/precision_convergence_full.png', dpi=300, bbox_inches='tight')
print("Graphique généré : Benchmarks/precision_convergence_full.png")
