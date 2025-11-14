#### .\train\spatial.py
# train/spatial.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2

def plot_embedding_distribution(embeddings, x, y, save_path, map_size=40, map_image_path=None):
    pca = PCA(n_components=3, random_state=42)
    rgb = pca.fit_transform(embeddings)

    rgb_min = rgb.min(axis=0)
    rgb_ptp = np.ptp(rgb, axis=0)
    rgb = (rgb - rgb_min) / (rgb_ptp + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    if map_image_path:
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            h, w = map_img.shape
            ax.imshow(map_img, cmap='gray', extent=[0, w, h, 0], alpha=0.7)
    ax.scatter(x, y, c=rgb, s=10, alpha=0.8)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Embeddings in Space (PCA RGB)")
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_cluster_spatial_distribution(embeddings, x, y, save_path, n_clusters=50, map_image_path=None):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(embeddings)
    counts = np.bincount(labels)
    valid = counts > 3
    if not valid.any():
        return
    cid = np.random.choice(np.where(valid)[0])
    mask = labels == cid

    fig, ax = plt.subplots(figsize=(10, 8))
    if map_image_path:
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            h, w = map_img.shape
            ax.imshow(map_img, cmap='gray', extent=[0, w, h, 0], alpha=0.7)
    ax.scatter(x, y, c='lightgray', s=10, alpha=0.5)
    ax.scatter(x[mask], y[mask], c='red', s=30)
    ax.set_title(f"Cluster {cid} in Space")
    ax.axis("equal")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved cluster plot: {save_path}")