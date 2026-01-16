import os
import time
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


IS_WINDOWS = (os.name == "nt")


class LSMCUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LSMC – Runner + Visualisation")
        self.root.geometry("1550x900")

        self.proc = None
        self.out_q = queue.Queue()
        self.after_id = None
        # Le script est dans Benchmarks/, on veut la racine du projet (../)
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # UNE SEULE boucle after
        self.after_id = self.root.after(80, self._poll_output)

    # ---------------- UI ----------------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.grid(row=0, column=0, sticky="nsew")

        self.left = ttk.Frame(paned, padding=10)
        self.right = ttk.Frame(paned, padding=10)
        paned.add(self.left, weight=0)
        paned.add(self.right, weight=1)

        self.left.columnconfigure(0, weight=1)
        self.right.columnconfigure(0, weight=1)
        self.right.rowconfigure(0, weight=1)
        self.right.rowconfigure(1, weight=0)

        # ---- Runner box
        runner = ttk.LabelFrame(self.left, text="Simulation C++", padding=10)
        runner.grid(row=0, column=0, sticky="ew")
        runner.columnconfigure(1, weight=1)

        ttk.Label(runner, text="Exécutable").grid(row=0, column=0, sticky="w")
        self.exe_var = tk.StringVar(value=r"build\Release\P1RV_CUDA.exe")
        exe_entry = ttk.Entry(runner, textvariable=self.exe_var)
        exe_entry.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(runner, text="Parcourir", command=self._browse_exe).grid(row=0, column=2, padx=(0, 2))

        # Paramètres modèle
        self.param_vars = {}
        params = [
            ("S0", "100.0"),
            ("K", "100.0"),
            ("r", "0.05"),
            ("sigma", "0.20"),
            ("T", "1.0"),
        ]
        for i, (k, default) in enumerate(params, start=1):
            ttk.Label(runner, text=k).grid(row=i, column=0, sticky="w")
            v = tk.StringVar(value=default)
            self.param_vars[k] = v
            ttk.Entry(runner, textvariable=v, width=12).grid(row=i, column=1, sticky="w", padx=6)

        # Taille de run (single-run)
        ttk.Separator(runner).grid(row=6, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Label(runner, text="N_steps").grid(row=7, column=0, sticky="w")
        self.nsteps_var = tk.StringVar(value="500")
        ttk.Entry(runner, textvariable=self.nsteps_var, width=12).grid(row=7, column=1, sticky="w", padx=6)

        ttk.Label(runner, text="N_paths").grid(row=8, column=0, sticky="w")
        self.npaths_var = tk.StringVar(value="2000")
        ttk.Entry(runner, textvariable=self.npaths_var, width=12).grid(row=8, column=1, sticky="w", padx=6)

        # Options
        self.dump_paths_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(runner, text="Sauver paths.csv", variable=self.dump_paths_var).grid(row=9, column=0, columnspan=2, sticky="w")

        self.autoload_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(runner, text="Auto visualiser à la fin", variable=self.autoload_var).grid(row=10, column=0, columnspan=2, sticky="w")

        # Boutons
        ttk.Separator(runner).grid(row=11, column=0, columnspan=3, sticky="ew", pady=10)

        btn_row = ttk.Frame(runner)
        btn_row.grid(row=12, column=0, columnspan=3, sticky="ew")
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        btn_row.columnconfigure(2, weight=1)

        ttk.Button(btn_row, text="Lancer", command=self.run_cpp).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(btn_row, text="STOP", command=self.stop_cpp).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(btn_row, text="Charger & Visualiser", command=self.load_and_visualize).grid(row=0, column=2, sticky="ew")

        # Status
        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(self.left, textvariable=self.status_var).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # Console
        console_box = ttk.LabelFrame(self.left, text="Console C++", padding=8)
        console_box.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self.left.rowconfigure(2, weight=1)

        self.console = tk.Text(console_box, height=18, wrap="word", bg="#0f1115", fg="#d6f5d6", insertbackground="white")
        ysb = ttk.Scrollbar(console_box, orient="vertical", command=self.console.yview)
        self.console.configure(yscrollcommand=ysb.set)
        self.console.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        console_box.columnconfigure(0, weight=1)
        console_box.rowconfigure(0, weight=1)
        self._console_write("Prêt.\n")

        # ---- Plot
        plot_box = ttk.LabelFrame(self.right, text="Graphiques", padding=8)
        plot_box.grid(row=0, column=0, sticky="nsew")
        plot_box.columnconfigure(0, weight=1)
        plot_box.rowconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _browse_exe(self):
        f = filedialog.askopenfilename(title="Choisir l'exécutable", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if f:
            self.exe_var.set(f)

    # -------------- Console helpers --------------
    def _console_write(self, s: str):
        self.console.insert("end", s)
        self.console.see("end")

    def _console_clear(self):
        self.console.delete("1.0", "end")

    def _poll_output(self):
        try:
            while True:
                line = self.out_q.get_nowait()
                self._console_write(line)
        except queue.Empty:
            pass

        self.after_id = self.root.after(80, self._poll_output)

    # -------------- Process I/O --------------
    def _reader_thread(self, p: subprocess.Popen):
        try:
            for line in p.stdout:
                self.out_q.put(line)
        except Exception:
            pass
        rc = p.poll()
        self.out_q.put(f"\n=== Fin du process (code {rc}) ===\n")
        self.proc = None
        self.status_var.set("Terminé" if rc == 0 else f"Terminé (code {rc})")

        if self.autoload_var.get() and rc == 0:
            # laisser un tout petit délai pour que l'OS flush les fichiers
            time.sleep(0.1)
            self.root.after(0, self.load_and_visualize)

    # -------------- RUN / STOP --------------
    def run_cpp(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Déjà en cours", "Un process tourne déjà.")
            return

        exe = self.exe_var.get().strip()
        
        # Résolution du chemin absolu pour le check
        full_exe = exe
        if not os.path.isabs(exe):
            full_exe = os.path.join(self.project_dir, exe)

        if not os.path.exists(full_exe):
            messagebox.showerror("Erreur", f"Exécutable introuvable:\n{full_exe}\n(Project Dir: {self.project_dir})")
            return

        cwd = self.project_dir  # <<< ICI

        cmd = [exe]
        for k, v in self.param_vars.items():
            cmd += [f"--{k}", v.get().strip()]

        cmd += ["--bench", "0"]
        cmd += ["--N_steps", self.nsteps_var.get().strip()]
        cmd += ["--N_paths", self.npaths_var.get().strip()]
        cmd += ["--dump_paths", "1" if self.dump_paths_var.get() else "0"]
        
        # Random seed pour varier les paths
        seed_val = random.randint(0, 999999)
        cmd += ["--seed", str(seed_val)]

        self._console_clear()
        self._console_write("=== Lancement C++ ===\n")

        # Nettoyage des anciens fichiers pour éviter de lire des vieux résultats
        for fcsv in ["resultats_lsmc.csv", "paths.csv"]:
            p = os.path.join(cwd, fcsv)
            if os.path.exists(p):
                try:
                    os.remove(p)
                    self._console_write(f"[UI] Supprimé avant run: {fcsv}\n")
                except Exception as e:
                    self._console_write(f"[UI] Erreur suppression {fcsv}: {e}\n")

        self._console_write("CWD: " + cwd + "\n")
        self._console_write("CMD: " + " ".join(cmd) + "\n\n")

        try:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0

            self.proc = subprocess.Popen(
                cmd,
                cwd=cwd,  # <<< ICI
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags
            )
        except Exception as e:
            self.proc = None
            messagebox.showerror("Erreur", str(e))
            return

        self.status_var.set("En cours...")
        threading.Thread(target=self._reader_thread, args=(self.proc,), daemon=True).start()


    def load_and_visualize(self):
        cwd = self.project_dir  # <<< ICI

        res_path = os.path.join(cwd, "resultats_lsmc.csv")
        paths_path = os.path.join(cwd, "paths.csv")

        if not os.path.exists(res_path):
            messagebox.showerror("Erreur", f"resultats_lsmc.csv manquant dans:\n{cwd}")
            return
        if self.dump_paths_var.get() and not os.path.exists(paths_path):
            messagebox.showerror("Erreur", f"paths.csv manquant dans:\n{cwd}")
            return

        self.ax.clear()

        if os.path.exists(paths_path):
            # Debug info
            mtime = os.path.getmtime(paths_path)
            self._console_write(f"\n[UI] Lecture paths.csv (mtime={mtime})\n")

            dfp = pd.read_csv(paths_path)
            time_cols = [c for c in dfp.columns if c.startswith("t")]
            if len(time_cols) > 0:
                for _, row in dfp[time_cols].head(60).iterrows():
                    self.ax.plot(row.to_numpy(dtype=float), alpha=0.25)

        self.ax.set_title("Chemins Monte Carlo")
        self.ax.set_xlabel("Pas de temps")
        self.ax.set_ylabel("Prix")

        df = pd.read_csv(res_path)
        df = self._filter_results_on_params(df)

        price, n_kept = self._price_band_20pct(df)
        if price is None:
            txt = "Prix théorique indisponible"
        else:
            txt = f"Prix (bande ±20%) = {price:.4f}\nRuns retenus: {n_kept}"

        self.ax.text(
            0.02, 0.98, txt,
            transform=self.ax.transAxes,
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        self.fig.tight_layout()
        self.canvas.draw()

    def stop_cpp(self):
        p = self.proc
        if p is None:
            self._console_write("\n=== STOP: rien à arrêter ===\n")
            return
        if p.poll() is not None:
            self._console_write("\n=== STOP: déjà terminé ===\n")
            self.proc = None
            return

        self._console_write("\n=== STOP demandé ===\n")
        self.status_var.set("Arrêt demandé...")

        # STOP propre d'abord, puis kill si ça traîne
        try:
            if IS_WINDOWS:
                # envoie CTRL+BREAK au groupe (doit être géré côté C++)
                try:
                    p.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
                except Exception:
                    p.terminate()
            else:
                p.terminate()

            # attendre un peu
            for _ in range(40):
                if p.poll() is not None:
                    break
                time.sleep(0.05)

            if p.poll() is None:
                self._console_write("=== STOP: kill (forcé) ===\n")
                p.kill()
        finally:
            # Ne mettez pas self.proc=None ici: le thread reader le fera
            pass

    def on_close(self):
        try:
            self.stop_cpp()
        except Exception:
            pass

        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass

        self.root.destroy()

    # -------------- Visualisation --------------
    def _filter_results_on_params(self, df):
        # tolérance floats
        tol = 1e-9
        cur = {k: float(v.get()) for k, v in self.param_vars.items()}

        for c in ["S0", "K", "r", "sigma", "T", "Prix", "Threads", "N_paths", "N_steps"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["S0", "K", "r", "sigma", "T", "Prix"])

        self._console_write(f"[DEBUG] Filtering {len(df)} rows...\n")
        self._console_write(f"[DEBUG] Target params: {cur}\n")

        # Debug filter loop
        kept_indices = []
        for idx, row in df.iterrows():
            failures = []
            if np.abs(row["S0"] - cur["S0"]) > tol: failures.append(f"S0({row['S0']} vs {cur['S0']})")
            if np.abs(row["K"] - cur["K"]) > tol: failures.append(f"K({row['K']} vs {cur['K']})")
            if np.abs(row["r"] - cur["r"]) > tol: failures.append(f"r({row['r']} vs {cur['r']})")
            if np.abs(row["sigma"] - cur["sigma"]) > tol: failures.append(f"sigma({row['sigma']} vs {cur['sigma']})")
            if np.abs(row["T"] - cur["T"]) > tol: failures.append(f"T({row['T']} vs {cur['T']})")
            
            if not failures:
                kept_indices.append(idx)
            else:
                self._console_write(f"[DEBUG] Row {idx} rejected: {', '.join(failures)}\n")

        filtered_df = df.loc[kept_indices].copy()
        self._console_write(f"[DEBUG] Rows kept: {len(filtered_df)}\n")
        return filtered_df

    def _price_band_20pct(self, df):
        if "Prix" not in df.columns:
            return None, 0

        prices = pd.to_numeric(df["Prix"], errors="coerce").dropna().to_numpy(dtype=float)
        if prices.size == 0:
            return None, 0

        ref = float(np.median(prices))
        denom = max(abs(ref), 1e-12)
        kept = prices[np.abs(prices - ref) / denom <= 0.20]
        if kept.size == 0:
            return ref, 0
        return float(np.mean(kept)), int(kept.size)

def main():
    root = tk.Tk()
    LSMCUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()