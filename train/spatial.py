#### .\train\spatial.py
# train/spatial.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2

def plot_embedding_distribution(
        embeddings, x, y, save_path,
        map_image_path=None, rgb_precomputed=None):
    """
    Plot all embeddings in space using PCA → RGB coloring.
    
    If `rgb_precomputed` is provided (shape: N x 3, values in [0,1]), 
    the PCA step is skipped and the precomputed colors are used directly.
    This ensures color consistency with orientation-filtered plots.
    """
    if rgb_precomputed is None:
        pca = PCA(n_components=3, random_state=42)
        rgb_3d = pca.fit_transform(embeddings)
        rgb_min = rgb_3d.min(axis=0)
        rgb_ptp = np.ptp(rgb_3d, axis=0) + 1e-8
        rgb = (rgb_3d - rgb_min) / rgb_ptp
    else:
        rgb = rgb_precomputed  # Already normalized to [0,1]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Optional: overlay map image
    if map_image_path:
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            h, w = map_img.shape
            ax.imshow(map_img, cmap='gray', extent=[0, w, h, 0], alpha=0.7)

    ax.scatter(x, y, c=rgb, s=10, alpha=0.8)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Embeddings in Space (PCA RGB)")
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved embedding distribution: {save_path}")


def plot_oriented_embedding_distribution(
        embeddings, x, y, theta, save_path,
        target_orientation=0, tolerance=10,
        map_image_path=None,
        pca_global=None, rgb_min=None, rgb_ptp=None):
    """
    Plot embeddings filtered by orientation using PCA → RGB coloring.
    
    If `pca_global`, `rgb_min`, and `rgb_ptp` are provided, 
    the same PCA transformation and normalization as the global plot are reused.
    This ensures **color consistency across all orientation plots in the same epoch**.
    """
    # --- Filter by orientation ---
    theta_mod = np.mod(theta, 360)
    diff = np.abs((theta_mod - target_orientation + 180) % 360 - 180)
    mask = diff <= tolerance
    
    if mask.sum() == 0:
        print(f"[WARN] No samples in orientation range {target_orientation}° ± {tolerance}°")
        return
    
    x_f, y_f, emb_f = x[mask], y[mask], embeddings[mask]

    # --- Compute RGB using shared PCA (if provided) ---
    if pca_global is not None and rgb_min is not None and rgb_ptp is not None:
        rgb_3d = pca_global.transform(emb_f)  # Reuse fitted PCA
        rgb = (rgb_3d - rgb_min) / rgb_ptp
    else:
        pca = PCA(n_components=3, random_state=42)
        rgb_3d = pca.fit_transform(emb_f)
        rgb = (rgb_3d - rgb_3d.min(axis=0)) / (np.ptp(rgb_3d, axis=0) + 1e-8)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add map background if available
    if map_image_path:
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            h, w = map_img.shape
            ax.imshow(map_img, cmap='gray', extent=[0, w, h, 0], alpha=0.7)

    ax.scatter(x_f, y_f, c=rgb, s=15, alpha=0.8)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(
        f"Embeddings (Orientation {target_orientation}° ± {tolerance}°)\n"
        f"{len(x_f)} samples"
    )
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved oriented embedding distribution: {save_path}")

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