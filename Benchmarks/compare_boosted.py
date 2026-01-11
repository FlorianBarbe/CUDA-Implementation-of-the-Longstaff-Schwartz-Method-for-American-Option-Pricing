import pandas as pd
import sys

def compare(old_file, new_file):
    try:
        df_old = pd.read_csv(old_file)
        df_new = pd.read_csv(new_file)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # Normalize Old Schema
    # Old: mode,N_steps,N_paths,price,time_ms,speedup
    # New: Base,Degree,Mode,Paths,Price,Time_ms,Diff_RK4

    # 1. Rename columns in Old
    df_old.columns = [c.lower() for c in df_old.columns] # ensure lowercase check
    # Map old 'mode' values to 'Base'
    if 'mode' in df_old.columns:
        df_old['Base'] = df_old['mode'].apply(lambda x: x.replace('GPU_', ''))
        # Map 'Cubic' to 'Monomial' for comparison with specific degree, or keep as is?
        # New code doesn't have 'Cubic' base exposed, it maps to Monomial deg 3 internally?
        # Or maybe Main.cu still loops over "Cubic"? No, Main.cu loops Mono, Hermite, Lag, Cheby.
        # But EvalBasis handles Cubic as Monomial.
        # If Old has 'Cubic', we compare it to New 'Monomial' at degree 3?
        # Let's assume Old Degree was 3.
        df_old['Degree'] = 3
    
    df_old = df_old.rename(columns={'n_paths': 'Paths', 'time_ms': 'Time_Old'})

    # 2. Filter New Data
    # Only keep Degree 3 for fair comparison (assuming Old was Deg 3)
    df_new = df_new[df_new['Degree'] == 3].copy()
    df_new = df_new[df_new['Mode'] == 'GPU'].copy()
    df_new = df_new.rename(columns={'Time_ms': 'Time_New'})

    # 3. Merge
    # Join on Base and Paths.
    # Handle 'Cubic' in Old vs 'Monomial' in New ??
    # If Old has 'Cubic', and New has 'Monomial', they won't join.
    # Let's see what Bases are in Old.
    
    print("Old Bases:", df_old['Base'].unique())
    print("New Bases:", df_new['Base'].unique())

    # Map Old 'Cubic' to 'Monomial' if present, because New uses 'Monomial' for x^k
    df_old.loc[df_old['Base'] == 'Cubic', 'Base'] = 'Monomial'

    merged = pd.merge(df_new, df_old[['Base', 'Paths', 'Time_Old']], 
                      on=['Base', 'Paths'], how='inner')

    merged['Speedup'] = merged['Time_Old'] / merged['Time_New']
    
    print("\n=== GPU SPEEDUP COMPARISON (New 'NoSync' vs Old 'Sync') [Degree 3] ===\n")
    
    # Sort
    merged = merged.sort_values(by=['Paths', 'Base'])

    # Display for N=100k
    large_paths = merged[merged['Paths'] == 100000]
    
    if not large_paths.empty:
        print(f"{'Base':<12} | {'Old (ms)':<10} | {'New (ms)':<10} | {'Speedup':<8}")
        print("-" * 50)
        for index, row in large_paths.iterrows():
            print(f"{row['Base']:<12} | {row['Time_Old']:<10.2f} | {row['Time_New']:<10.2f} | {row['Speedup']:<8.2f}x")
    else:
        print("No intersecting data found for Paths=100000.")
        # Print whatever we have
        print(merged[['Base', 'Paths', 'Time_Old', 'Time_New', 'Speedup']])

    # Overall Stats
    if not merged.empty:
        avg_speedup = merged['Speedup'].mean()
        max_speedup = merged['Speedup'].max()
        print("\n" + "-" * 40)
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Max Speedup:     {max_speedup:.2f}x")
    else:
        print("\nNo common data points found.")

if __name__ == "__main__":
    compare('Benchmarks/benchmark_boosted_old.csv', 'Benchmarks/benchmark_boosted_new.csv')
