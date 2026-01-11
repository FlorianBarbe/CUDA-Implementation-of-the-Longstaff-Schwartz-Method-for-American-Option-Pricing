import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparison(old_file, new_file, output_image):
    try:
        df_old = pd.read_csv(old_file)
        df_new = pd.read_csv(new_file)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # Normalize Old Data
    df_old.columns = [c.lower() for c in df_old.columns]
    if 'mode' in df_old.columns:
        df_old['Base'] = df_old['mode'].apply(lambda x: x.replace('GPU_', ''))
    df_old = df_old.rename(columns={'n_paths': 'Paths', 'time_ms': 'Time'})
    df_old['Version'] = 'GPU Sync (Old)'
    
    # Filter Old for Monomial only (standard)
    df_old = df_old[df_old['Base'] == 'Monomial']

    # Normalize New Data
    df_new = df_new[df_new['Degree'] == 3].copy()
    df_new = df_new[df_new['Mode'] == 'GPU'].copy()
    df_new = df_new.rename(columns={'Time_ms': 'Time'})
    df_new['Version'] = 'GPU No-Sync (New)'
    
    # Filter New for Monomial
    df_new = df_new[df_new['Base'] == 'Monomial']

    # Combine
    cols = ['Paths', 'Time', 'Version']
    df_combined = pd.concat([df_old[cols], df_new[cols]])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=df_combined, x='Paths', y='Time', hue='Version', marker='o')
    
    plt.title('Performance Comparison: GPU Sync vs No-Sync (Monomial Basis)', fontsize=14)
    plt.xlabel('Number of Paths ($N_{paths}$)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    x_ticks = sorted(df_combined['Paths'].unique())
    plt.xticks(x_ticks, rotation=45)
    plt.legend(title='Implementation')
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(old_file), '../Rapport/images', output_image)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    plot_comparison('Benchmarks/benchmark_boosted_old.csv', 
                    'Benchmarks/benchmark_boosted_new.csv', 
                    'gpu_sync_vs_nosync.png')
