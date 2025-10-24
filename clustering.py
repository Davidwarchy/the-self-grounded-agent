"""
clustering.py

Utility functions for:
- clustering embeddings with k-means
- sampling representative LiDAR rays from each cluster
- visualizing clusters in a grid of ray plots

Intended to be imported from train.py as:
    from clustering import sample_clusters_and_inspect, plot_clusters_grid
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from matplotlib.cm import get_cmap


def sample_clusters_and_inspect(embeddings, lidar_data, k=20, n_samples_per=5):
    """
    Cluster embeddings with k-means, then for each cluster:
        - find nearest n_samples_per embeddings to the centroid
        - return their LiDAR ray readings for inspection

    Returns: list of tuples (cluster_id, indices, [ray_vectors])
    """
    kmeans = KMeans(n_clusters=k, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    results = []
    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        cluster_embs = embeddings[idxs]
        center = centers[cid][None, :]

        # nearest to cluster center
        dists = np.linalg.norm(cluster_embs - center, axis=1)
        nearest = idxs[np.argsort(dists)[:n_samples_per]]
        rays = [lidar_data[i] for i in nearest]
        results.append((cid, nearest, rays))

    return results


def plot_clusters_grid(cluster_results, n_samples_per=5):
    """
    Draw a grid of subplots.
    Each subplot = a cluster, showing LiDAR ray profiles of representative samples.
    """
    k = len(cluster_results)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    cmap = get_cmap('tab20')

    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3*rows))
    axes = axes.flatten()

    for i, (cid, idxs, rays) in enumerate(cluster_results):
        ax = axes[i]
        col = cmap(cid % 20)
        for r in rays[:n_samples_per]:
            ax.plot(r, color=col, alpha=0.4)
        ax.set_title(f"Cluster {cid}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("Example LiDAR rays per embedding cluster")
    plt.tight_layout()
    plt.show()
