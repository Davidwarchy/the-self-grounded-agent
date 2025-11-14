# train/dataset.py
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_or_create_parquet(data_dir, n_files, cache_dir=None):
    if cache_dir is None:
        cache_dir = data_dir
    parquet_path = os.path.join(cache_dir, f"merged_{n_files}.parquet")
    if os.path.exists(parquet_path):
        print(f"[INFO] Using cached parquet: {parquet_path}")
        return parquet_path

    print("[INFO] Creating merged parquet...")
    log_files = sorted(
        glob.glob(os.path.join(data_dir, "log_*.csv")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )[:n_files]

    dfs = [pd.read_csv(f) for f in tqdm(log_files, desc="Loading CSVs")]
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Saved merged parquet: {parquet_path}")
    return parquet_path

class LidarDataset(Dataset):
    def __init__(self, parquet_path, num_rays=100, start_idx=0, end_idx=None):
        self.df = pd.read_parquet(parquet_path)
        self.num_rays = num_rays
        ray_cols = [f'ray_{i}' for i in range(num_rays)]
        self.lidar = (self.df[ray_cols].values.astype(np.float32) / 200.0)
        self.x = self.df['x'].values
        self.y = self.df['y'].values
        self.theta = self.df['orientation'].values

        if end_idx is None:
            end_idx = len(self.df) - 1
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.length = max(0, end_idx - start_idx - 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = self.start_idx + idx
        anchor = torch.from_numpy(self.lidar[i])
        positive = torch.from_numpy(self.lidar[i + 1])
        neg_idx = np.random.randint(len(self.df))
        negative = torch.from_numpy(self.lidar[neg_idx])
        return anchor, positive, negative, self.x[i], self.y[i], self.theta[i]

    @property
    def valid_slice(self):
        s = slice(self.start_idx, self.start_idx + len(self))
        return {
            'x': self.x[s], 'y': self.y[s], 'theta': self.theta[s],
            'lidar': self.lidar[s]
        }