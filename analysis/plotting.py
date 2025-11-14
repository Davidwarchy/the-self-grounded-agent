# analysis/plotting.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
import cv2
import json
from analysis.utils import compute_correlations

def find_map_image_path(data_dir):
    """Find map image path from metadata.json or common locations"""
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            map_image_name = metadata.get("environment_parameters", {}).get("map_image")
            if map_image_name:
                # Try common locations
                possible_paths = [
                    os.path.join(data_dir, map_image_name),
                    os.path.join("environments", "images", map_image_name),
                    os.path.join("..", "environments", "images", map_image_name),
                    os.path.join(".", "environments", "images", map_image_name),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        return path
        except Exception as e:
            print(f"[WARNING] Could not read metadata: {e}")
    
    print("[WARNING] Map image not found via metadata, trying default locations")
    # Try to find any common map images
    default_paths = [
        os.path.join("environments", "images", "6.png"),
        os.path.join("..", "environments", "images", "6.png"),
        os.path.join(".", "environments", "images", "6.png"),
    ]
    for path in default_paths:
        if os.path.exists(path):
            return path
    
    return None

def plot_embeddings_on_map(x, y, rgb, data_dir=None, map_image_path=None, max_points=None, save_path=None):
    """Plot embeddings overlaid on actual map image."""
    if max_points and len(x) > max_points:
        print(f"[INFO] Subsampling to {max_points} points for plotting... len={len(x)}")
        idxs = np.random.choice(len(x), min(max_points, len(x)), replace=False)
        x, y, rgb = x[idxs], y[idxs], rgb[idxs]

    # Find map image path
    if map_image_path is None and data_dir is not None:
        map_image_path = find_map_image_path(data_dir)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Add map image as background if found
    if map_image_path and os.path.exists(map_image_path):
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            # Flip the image to have (0,0) at bottom-left
            map_img = np.flipud(map_img)
            ax.imshow(map_img, cmap='gray', 
                     extent=[0, map_img.shape[1], 0, map_img.shape[0]], 
                     alpha=0.7)
            print(f"[INFO] Map overlay added: {map_image_path}")
            
            # Set axis limits to match image
            ax.set_xlim(0, map_img.shape[1])
            ax.set_ylim(0, map_img.shape[0])
    
    # Plot trajectory points
    scatter = ax.scatter(x, y, c=rgb, s=30, alpha=0.8)
    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")
    ax.set_title("Embedding Distribution on Map (PCA RGB)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    # Set axis to have (0,0) at bottom-left
    ax.set_ylim(ax.get_ylim()[::-1])
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_oriented_embeddings(x, y, theta, rgb, data_dir=None, map_image_path=None, 
                           target_orientation=0, tolerance=10, save_path=None):
    """Plot embeddings filtered by orientation with map overlay."""
    # Filter by orientation
    diff = np.abs((theta - target_orientation + 180) % 360 - 180)
    mask = diff <= tolerance
    if mask.sum() == 0:
        print("[WARN] No samples in orientation range.")
        return

    x_f, y_f, colors_f = x[mask], y[mask], rgb[mask]

    # Find map image path
    if map_image_path is None and data_dir is not None:
        map_image_path = find_map_image_path(data_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add map image as background if found
    if map_image_path and os.path.exists(map_image_path):
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            map_img = np.flipud(map_img)
            ax.imshow(map_img, cmap='gray', 
                     extent=[0, map_img.shape[1], 0, map_img.shape[0]], 
                     alpha=0.7)
            ax.set_xlim(0, map_img.shape[1])
            ax.set_ylim(0, map_img.shape[0])
    
    ax.scatter(x_f, y_f, c=colors_f, s=15, alpha=0.8)
    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")
    ax.set_title(f"Embeddings (Orientation {target_orientation}° ± {tolerance}°)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ax.get_ylim()[::-1])
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_random_cluster(x, y, embeddings, data_dir=None, map_image_path=None, 
                       n_clusters=50, save_path=None):
    """Plot spatial distribution of a random cluster with map overlay."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    unique, counts = np.unique(labels, return_counts=True)
    valid = unique[counts > 3]
    if len(valid) == 0:
        print("[WARN] No valid clusters.")
        return

    chosen = np.random.choice(valid)
    mask = labels == chosen

    # Find map image path
    if map_image_path is None and data_dir is not None:
        map_image_path = find_map_image_path(data_dir)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add map image as background if found
    if map_image_path and os.path.exists(map_image_path):
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            map_img = np.flipud(map_img)
            ax.imshow(map_img, cmap='gray', 
                     extent=[0, map_img.shape[1], 0, map_img.shape[0]], 
                     alpha=0.7)
            ax.set_xlim(0, map_img.shape[1])
            ax.set_ylim(0, map_img.shape[0])
    
    # Plot all points in light gray, cluster points in red
    ax.scatter(x, y, c='lightgray', s=10, alpha=0.5, label="All positions")
    ax.scatter(x[mask], y[mask], c='red', s=30, alpha=0.9, label=f"Cluster {chosen}")
    
    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")
    ax.set_title(f"Spatial Distribution of Cluster {chosen}")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ax.get_ylim()[::-1])
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_correlations(features, rgb, save_path=None):
    """Plot grouped bar chart of correlations for multiple features."""
    all_corrs = {}
    for name, vals in features.items():
        all_corrs[name] = compute_correlations(vals, rgb)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(3)
    width = 0.35
    colors = ['#9b59b6', '#ff8c42']  # Adjust for number of features

    for i, (name, corrs) in enumerate(all_corrs.items()):
        offset = width * (i - (len(features)-1)/2)
        bars = ax.bar(x_pos + offset, corrs, width, label=name, color=colors[i], alpha=0.8)
        for bar, corr in zip(bars, corrs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03 if height > 0 else height - 0.03,
                    f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel("RGB Channel")
    ax.set_ylabel("Pearson Correlation")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Red', 'Green', 'Blue'])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-1, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)