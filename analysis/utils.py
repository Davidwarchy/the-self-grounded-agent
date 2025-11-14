# analysis/utils.py
# This script contains utility functions for data loading and processing

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

def load_parquet(parquet_path):
    """Load parquet file and return DataFrame."""
    print(f"[INFO] Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Total columns: {len(df.columns)}")
    return df

def load_embeddings(embeddings_path):
    """Load embeddings from .npy file."""
    embeddings = np.load(embeddings_path)
    print(f"[INFO] Loaded embeddings: {embeddings.shape}")
    return embeddings

def prepare_data(df, embeddings):
    """Prepare positions, orientations, and lidar data, aligning lengths if necessary."""
    x = df["x"].values
    y = df["y"].values
    theta = df["orientation"].values
    lidar_data = df[[f"ray_{i}" for i in range(100)]].values  # Assuming 100 rays

    if len(embeddings) == len(df) - 1:
        x, y, theta, lidar_data = x[:-1], y[:-1], theta[:-1], lidar_data[:-1]

    return x, y, theta, lidar_data

def compute_pca_rgb(embeddings):
    """Reduce embeddings to 3D via PCA and normalize to RGB [0,1]."""
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    rgb = (embeddings_3d - embeddings_3d.min(axis=0)) / (embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0))
    return rgb

def compute_features(lidar_data, theta):
    """Compute derived features: dist_to_wall, openness, turn_intensity."""
    dist_to_wall = np.min(lidar_data, axis=1)
    openness = np.max(lidar_data, axis=1)
    turn_intensity = np.abs(np.diff(theta))
    turn_intensity = np.append(turn_intensity, 0)  # Pad to match length
    return dist_to_wall, openness, turn_intensity

def align_arrays(*arrays):
    """Align multiple arrays to the minimum length."""
    min_length = min(len(arr) for arr in arrays)
    return [arr[:min_length] for arr in arrays]

def compute_correlations(feature_values, rgb, corr_type='pearson'):
    """Compute correlations between feature and RGB channels."""
    correlations = []
    for dim in range(rgb.shape[1]):
        if corr_type == 'pearson':
            corr, _ = pearsonr(feature_values, rgb[:, dim])
        elif corr_type == 'spearman':
            corr, _ = spearmanr(feature_values, rgb[:, dim])
        else:
            raise ValueError("corr_type must be 'pearson' or 'spearman'")
        correlations.append(corr)
    return np.array(correlations)