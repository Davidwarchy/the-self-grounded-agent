import os
import glob
import pandas as pd
from tqdm import tqdm

def load_data(data_dir, n):
    """Load or create merged parquet file"""
    parquet_path = os.path.join(data_dir, f"merged_{n}.parquet")

    if os.path.exists(parquet_path):
        print(f"[INFO] Using cached parquet: {parquet_path}")
        return parquet_path

    print("[INFO] Merged parquet not found. Reading CSV logs...")
    log_files = glob.glob(os.path.join(data_dir, "log_*.csv"))
    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    log_files = log_files[:n]

    dfs = []
    for f in tqdm(log_files, desc="Reading CSVs", unit="file"):
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Saved merged parquet: {parquet_path}")

    return parquet_path

# Example usage:
if __name__ == "__main__":
    data_dir = "output/2025-11-13-110326_random_walk"  # Current directory
    n = 1000  # Number of log files to process
    
    parquet_path = load_data(data_dir, n)
    print(f"Created parquet file: {parquet_path}")