import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Chargement des données
# =========================
csv_path = Path("benchmark_lsmc.csv")
df = pd.read_csv(csv_path)

# =========================
# Fonction de tracé
# =========================
def plot_mode(df, mode):
    df_mode = df[df["mode"] == mode]

    plt.figure(figsize=(8, 6))

    for n_steps in sorted(df_mode["N_steps"].unique()):
        sub = df_mode[df_mode["N_steps"] == n_steps]
        plt.scatter(
            sub["N_paths"],
            sub["time_ms"],
            marker="x",
            s=70,
            label=f"N_steps = {n_steps}"
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.xlabel("N_paths")
    plt.ylabel("Execution time (ms)")
    plt.title(f"LSMC {mode} – Execution Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================
# Tracés
# =========================
plot_mode(df, "CPU")
plot_mode(df, "GPU")
