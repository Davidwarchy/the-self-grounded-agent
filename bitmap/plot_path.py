#!/usr/bin/env python
"""
plot_path_simple.py
Draw only the (x, y) robot trajectory – no map image, no exploration overlay.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from create_parquet import load_data   # <-- your helper that caches the parquet

# ----------------------------------------------------------------------
def plot_xy(data_dir: str, start_step: int, end_step: int):
    # 1. Load / create the merged parquet (uses up to 1000 log files by default)
    parquet_path = load_data(data_dir, n=5_000)
    print(f"[INFO] Loading data from {parquet_path}...")

    df = pd.read_parquet(parquet_path)
    total_steps = len(df)
    print(f"[INFO] Total steps in log: {total_steps}")

    # 2. Validate / clamp the requested range
    if start_step < 0:
        start_step = 0
    if end_step >= total_steps:
        print(f"[WARNING] end_step {end_step} > max step {total_steps-1} → clamping")
        end_step = total_steps - 1
    if start_step > end_step:
        print("[ERROR] start_step > end_step after clamping – nothing to plot.")
        return

    # 3. Slice the dataframe
    df_slice = df.iloc[start_step : end_step + 1]
    x = df_slice["x"].values
    y = df_slice["y"].values
    steps = df_slice["step"].values

    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    scatter = ax.scatter(x, y, c=steps, cmap="viridis", s=12, edgecolor="none", alpha=0.8)
    ax.plot(x, y, color="steelblue", linewidth=1.2, alpha=0.7, label="Path")

    # start / end markers
    ax.plot(x[0], y[0], "go", markersize=9, label="Start")
    ax.plot(x[-1], y[-1], "ro", markersize=9, label="End")

    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")
    ax.set_title(f"Robot trajectory – steps {start_step} → {end_step}\n"
                 f"{len(df_slice)} points")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Simulation step")

    # 5. Save
    out_png = os.path.join(data_dir, f"trajectory_{start_step}_to_{end_step}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved → {out_png}")
    plt.show()


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot only the (x, y) robot path from simulation logs."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="output/2025-11-13-110326_random_walk",
        help="Folder that contains the log_*.csv files (and metadata.json).",
    )
    parser.add_argument(
        "--start_step", type=int, required=True, help="First step (inclusive)."
    )
    parser.add_argument(
        "--end_step", type=int, required=True, help="Last step (inclusive)."
    )
    args = parser.parse_args()

    plot_xy(args.data_dir, args.start_step, args.end_step)


if __name__ == "__main__":
    main()