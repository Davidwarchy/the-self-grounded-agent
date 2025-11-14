# train/analysis/clustering.py
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def sample_clusters_and_inspect(embeddings, lidar_data, k=20, n_samples=5):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    results = []
    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        dists = np.linalg.norm(embeddings[idxs] - centers[cid], axis=1)
        nearest = idxs[np.argsort(dists)[:n_samples]]
        rays = lidar_data[nearest]
        results.append((cid, nearest, rays))
    return results

def plot_clusters_grid(results, n_samples=5, save_path=None):
    k = len(results)
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    cmap = get_cmap('tab20')

    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3*rows))
    axes = axes.flatten() if k > 1 else [axes]

    for i, (cid, _, rays) in enumerate(results):
        ax = axes[i]
        color = cmap(cid % 20)
        for r in rays:
            ax.plot(r, color=color, alpha=0.5)
        ax.set_title(f"Cluster {cid}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(results):]:
        ax.axis('off')

    plt.suptitle("LiDAR Samples per Embedding Cluster")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Cluster grid saved: {save_path}")
    plt.close(fig)