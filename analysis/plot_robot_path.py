import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # pip install tqdm

def load_xy(output_dir, n):
    parquet_path = os.path.join(output_dir, f"merged_{n}.parquet")

    # ---------- FAST PATH: parquet exists ----------
    if os.path.exists(parquet_path):
        print(f"[INFO] Using cached parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        return df["x"].to_numpy(), df["y"].to_numpy()

    # ---------- SLOW PATH: read CSVs & build parquet ----------
    print("[INFO] Merged parquet not found. Reading CSV logs...")

    log_files = glob.glob(os.path.join(output_dir, "log_*.csv"))
    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    log_files = log_files[:n]

    dfs = []
    for f in tqdm(log_files, desc="Reading CSVs", unit="file"):
        dfs.append(pd.read_csv(f, usecols=["x","y"]))

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Saved merged parquet: {parquet_path}")

    return df["x"].to_numpy(), df["y"].to_numpy()


if __name__ == "__main__":
    output_dir = "output/2025-10-18-175300"
    n = 5000

    x, y = load_xy(output_dir, n)

    plt.scatter(x, y, s=10, alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Robot Path")
    plt.show()
