"""
cluster_inspect.py

Clusters embedding vectors and plots representative LiDAR rays
per cluster in a subplot grid.

Run standalone: python cluster_inspect.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ============================================================
def nearest_indices_to_centroid(embeddings, labels, centers, cid, n, min_sep=100):
    """
    Return indices of n embeddings closest to centroid of cluster cid,
    but ensure chosen indices are >= min_sep apart in index space.
    Also prints them in a clean format.
    """
    idxs = np.where(labels == cid)[0]
    cluster_embs = embeddings[idxs]
    center = centers[cid][None, :]

    d = np.linalg.norm(cluster_embs - center, axis=1)
    candidates = idxs[np.argsort(d)]

    chosen = []
    for idx in candidates:
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)
            if len(chosen) == n:
                break

    # ----- print nicely -----
    print(f"cluster {cid}: " + ", ".join(map(str, chosen)))

    return np.array(chosen)



# ============================================================
def plot_clusters_grid(results, n_per):
    """
    results is list of (cluster_id, [rays])
    Each subplot = all representative rays for one cluster.
    """
    k = len(results)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3*rows))
    axes = axes.flatten()

    for i, (cid, ray_list) in enumerate(results):
        ax = axes[i]
        for r in ray_list[:n_per]:
            ax.plot(r, alpha=0.4)
        ax.set_title(f"C{cid}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # kill extra axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("LiDAR rays per embedding cluster", fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
if __name__ == "__main__":

    # --- CONFIG ---------------------------------------------
    OUT_DIR = r"output/2025-10-19-184833"
    PARQUET = "merged_5000.parquet"
    K = 6
    N_PER = 100
    # ----------------------------------------------------------

    print("[INFO] loading...")
    emb = np.load(os.path.join(OUT_DIR, "embeddings.npy"))
    df = pd.read_parquet(PARQUET)

    # build lidar matrix from ray_0...ray_99
    ray_cols = [c for c in df.columns if c.startswith("ray_")]
    ray_cols = sorted(ray_cols, key=lambda x: int(x.split("_")[1]))
    lidar_data = df[ray_cols].values[:len(emb)]

    print(f"[INFO] embeddings = {emb.shape}, lidar = {lidar_data.shape}")


    print(f"[INFO] k-means K={K}")
    km = KMeans(n_clusters=K, n_init="auto")
    labels = km.fit_predict(emb)
    centers = km.cluster_centers_

    # collect results
    cluster_results = []
    for cid in range(K):
        near = nearest_indices_to_centroid(emb, labels, centers, cid, N_PER)
        rays = [lidar_data[i] for i in near]
        cluster_results.append((cid, rays))

    print("[INFO] plotting...")
    plot_clusters_grid(cluster_results, N_PER)
