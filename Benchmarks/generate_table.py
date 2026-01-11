import pandas as pd

def generate_latex():
    try:
        df_new = pd.read_csv('Benchmarks/benchmark_boosted_new.csv')
        df_old = pd.read_csv('Benchmarks/benchmark_boosted_old.csv')
    except Exception as e:
        print(f"Error: {e}")
        return

    # Normalize Old
    df_old.columns = [c.strip().lower() for c in df_old.columns]
    
    if 'mode' in df_old.columns:
        df_old['Base'] = df_old['mode'].apply(lambda x: x.split('_')[-1])
        df_old['Architecture'] = df_old['mode'].apply(lambda x: "_".join(x.split('_')[:-1]))
    
    # Correct Mapping based on debug
    rename_map = {'n_paths': 'Paths', 'time_ms': 'Time', 'price_mean': 'Price'}
    df_old = df_old.rename(columns=rename_map)
    
    # Normalize New
    df_new.columns = [c.strip() for c in df_new.columns]
    df_new['Architecture'] = 'GPU'
    df_new = df_new.rename(columns={'Time_ms': 'Time'})
    
    # Filter Degree 3
    df_new = df_new[df_new['Degree'] == 3]

    cols = ['Base', 'Architecture', 'Paths', 'Price', 'Time']
    arch_map = {
        'CPU_Seq': 'CPU Séquentiel',
        'CPU_OMP': 'CPU OpenMP',
        'GPU': 'GPU'
    }
    
    bases = ['Monomial', 'Hermite', 'Laguerre', 'Chebyshev', 'Cubic']
    target_paths = [100000, 1000000]
    
    latex_output = []
    latex_output.append("\\begin{table}[H]")
    latex_output.append("\\centering")
    latex_output.append("\\small")
    latex_output.append("\\begin{tabular}{|l|l|c|r|r|r|}")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Base} & \\textbf{Architecture} & \\textbf{Trajectoires} & \\textbf{Prix (€)} & \\textbf{Temps (ms)} & \\textbf{Throughput (ops/s)} \\\\")
    latex_output.append("\\hline")

    for n in target_paths:
        found_n = False
        current_data = []

        for b in bases:
            # CPU/OMP logic
            for arch_key in ['CPU_Seq', 'CPU_OMP']:
                # Handle 'Cubic' in Old CSV if it exists (it maps to Monomial usually or distinct)
                # If 'Cubic' exists in df_old['Base'], it works.
                row = df_old[(df_old['Base'] == b) & (df_old['Paths'] == n) & (df_old['Architecture'] == arch_key)]
                
                if row.empty and b == 'Cubic':
                     # If old bench didn't have Cubic base execution, skip
                     pass
                elif not row.empty:
                    found_n = True
                    r = row.iloc[0]
                    current_data.append({
                        'Base': b,
                        'Arch': arch_map.get(arch_key, arch_key.replace('_',' ')),
                        'Paths': n,
                        'Price': r['Price'],
                        'Time': r['Time']
                    })

            # GPU logic from New CSV
            # New CSV Base names: Monomial, Hermite, Laguerre, Chebyshev.
            # Does it have 'Cubic'?
            # Usually users run with --basis Cubic.
            # My 'main.cu' mapped Cubic -> Monomial logic internally but key might be 'Cubic' if passed 'Cubic'.
            # If I ran logic with `basis=3` (Cheby) in previous runs.
            # Let's check if 'Cubic' is in df_new['Base'].
            # Alternatively, compare 'Cubic' to 'Monomial' GPU?
            row_gpu = df_new[(df_new['Base'] == b) & (df_new['Paths'] == n)]
            
            # Special handling for Cubic if user wants it (Table has it).
            if row_gpu.empty and b == 'Cubic':
                # Maybe map to Monomial GPU for comparison if logic is same?
                # User image shows Cubic GPU.
                pass

            if not row_gpu.empty:
                 found_n = True
                 r = row_gpu.iloc[0]
                 current_data.append({
                        'Base': b,
                        'Arch': 'GPU',
                        'Paths': n,
                        'Price': r['Price'],
                        'Time': r['Time']
                 })

        if found_n:
            latex_output.append(f"\\multicolumn{{6}}{{c}}{{\\textbf{{N = {n:,}}}}} \\\\".replace(',', ' '))
            latex_output.append("\\hline")
            
            for item in current_data:
                throughput = (item['Paths'] / item['Time']) * 1000.0
                th_str = f"{throughput/1e6:.1f} M"
                
                base_str = item['Base']
                if base_str == 'Monomial': base_str = 'Monomiale'
                if base_str == 'Cubic': base_str = 'Cubique'
                
                p_str = f"{item['Paths']//1000}k" if item['Paths'] < 1000000 else f"{item['Paths']//1000000}M"
                
                latex_output.append(f"{base_str} & {item['Arch']} & {p_str} & {item['Price']:.3f} & {item['Time']:.2f} & {th_str} \\\\")
            
            latex_output.append("\\hline")
    
    latex_output.append("\\end{tabular}")
    latex_output.append("\\caption{Comparaison de performance : Nouveau Solveur GPU Resident vs CPU}")
    latex_output.append("\\label{tab:perf_resident}")
    latex_output.append("\\end{table}")
    
    with open('Benchmarks/table_output.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_output))
    print("Table written to Benchmarks/table_output.tex")

if __name__ == "__main__":
    generate_latex()
