#!/usr/bin/env python
"""
plot_path_with_map.py
Draw the (x, y) robot trajectory with map image overlay.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
from create_parquet import load_data

# ----------------------------------------------------------------------
def plot_xy_with_map(data_dir: str, start_step: int, end_step: int):
    # 1. Load / create the merged parquet
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

    # 4. Load map image from metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        map_image_name = metadata["environment_parameters"]["map_image"]
        # Try to find the map image path
        map_image_path = None
        possible_paths = [
            os.path.join(data_dir, map_image_name),
            os.path.join("environments", "images", map_image_name),
            os.path.join("..", "environments", "images", map_image_name),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                map_image_path = path
                break
    else:
        print("[WARNING] No metadata.json found, trying default map paths")
        map_image_path = None

    # 5. Plot with map background
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
    # Add map image as background if found
    if map_image_path and os.path.exists(map_image_path):
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            # Flip the image to have (0,0) at bottom-left
            map_img = np.flipud(map_img)
            ax.imshow(map_img, cmap='gray', extent=[0, map_img.shape[1], 0, map_img.shape[0]], alpha=0.7)
            print(f"[INFO] Map overlay added: {map_image_path}")

    # Plot trajectory
    scatter = ax.scatter(x, y, c=steps, cmap="viridis", s=8, edgecolor="none", alpha=0.8)
    ax.plot(x, y, color="red", linewidth=1.5, alpha=0.9, label="Robot Path")

    # start / end markers
    ax.plot(x[0], y[0], "go", markersize=10, label="Start", markeredgecolor='black')
    ax.plot(x[-1], y[-1], "ro", markersize=10, label="End", markeredgecolor='black')

    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")
    ax.set_title(f"Robot Trajectory with Map Overlay\nSteps {start_step} → {end_step} ({len(df_slice)} points)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis to have (0,0) at bottom-left
    ax.set_ylim(ax.get_ylim()[::-1])

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Simulation Step")

    # 6. Save
    out_png = os.path.join(data_dir, f"trajectory_with_map_{start_step}_to_{end_step}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved → {out_png}")
    plt.show()


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot robot path with map image overlay."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="output/2025-11-13-180740_manual_control",
        help="Folder that contains the log_*.csv files and metadata.json",
    )
    parser.add_argument(
        "--start_step", type=int, required=True, help="First step (inclusive)."
    )
    parser.add_argument(
        "--end_step", type=int, required=True, help="Last step (inclusive)."
    )
    args = parser.parse_args()

    plot_xy_with_map(args.data_dir, args.start_step, args.end_step)


if __name__ == "__main__":
    main()
    # python plot_path.py --data_dir "output/2025-11-14-111702_random_walk" --start_step 0 --end_step 1000