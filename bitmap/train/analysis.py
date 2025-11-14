# analysis.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Map drawing functions
def get_map_objects(size=10):
    """Get wall segments and circular obstacles for the map."""
    height = size
    width = size * 2
    radius = height / 10
    objects = {
        'walls': [
            [(0, 0), (width, 0)],
            [(0, height), (width, height)],
            [(0, 0), (0, height)],
            [(width, 0), (width, height)],
            [(.6*height, 0), (height, height / 2)],
            [(1.1*height, 0), (height, height / 2)],
            [(1.8*height, height), (width, .8*height)],
            [(.8*height, height), (.8*height, .9*height)],
            [(1.2*height, height), (1.2*height, .9*height)],
            [(.8*height, .9*height), (1.2*height, .9*height)],
            [(.1*height, 0), (.1*height, .05*height)],
            [(.1*height, .05*height), (.05*height, .1*height)],
            [(0, .1*height), (.05*height, .1*height)],
        ],
        'circles': [
            {'center': (radius * 2, height - radius * 2), 'radius': radius*1.3},
            {'center': (width - radius * 2, radius * 2), 'radius': radius}
        ]
    }
    return objects

def plot_map_overlay(ax, map_size=40):
    """Plot map walls and obstacles on the given axis."""
    objects = get_map_objects(size=map_size)
    
    # Plot walls
    for wall in objects['walls']:
        xs = [wall[0][0], wall[1][0]]
        ys = [wall[0][1], wall[1][1]]
        ax.plot(xs, ys, 'k-', linewidth=2, alpha=0.6)
    
    # Plot circles
    for circle in objects['circles']:
        circle_plot = plt.Circle(circle['center'], circle['radius'], 
                                 color='black', fill=False, linewidth=2, alpha=0.6)
        ax.add_patch(circle_plot)

def plot_cluster_spatial_distribution(embeddings, x, y, save_path, n_clusters=50, map_size=40, random_state=42):
    """Cluster embeddings, pick a random cluster, plot it in space."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts > 3]
    if len(valid_clusters) == 0:
        print("[WARN] No valid clusters found.")
        return
    
    chosen_cluster = np.random.choice(valid_clusters)
    mask = labels == chosen_cluster
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_map_overlay(ax, map_size=map_size)
    ax.scatter(x, y, c='lightgray', s=10, alpha=0.5, label="All positions")
    ax.scatter(x[mask], y[mask], c='red', s=30, alpha=0.9, label=f"Cluster {chosen_cluster}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Spatial Distribution of Random Cluster {chosen_cluster}")
    ax.axis("equal")
    ax.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved cluster spatial distribution to {save_path}")

def plot_cluster_spatial_distribution_arrows(embeddings, x, y, theta, save_path, n_clusters=50, arrow_scale=2, map_size=40, random_state=42):
    """Cluster embeddings, pick a random cluster, plot it in space with arrows."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts > 3]
    if len(valid_clusters) == 0:
        print("[WARN] No valid clusters found.")
        return
    
    chosen_cluster = np.random.choice(valid_clusters)
    mask = labels == chosen_cluster
    
    theta_rad = np.deg2rad(theta)
    dx = np.cos(theta_rad)
    dy = np.sin(theta_rad)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_map_overlay(ax, map_size=map_size)
    ax.scatter(x, y, c='lightgray', s=10, alpha=0.5, label="All positions")
    ax.scatter(x[mask], y[mask], c='red', s=30, alpha=0.9, label=f"Cluster {chosen_cluster}")
    ax.quiver(x[mask], y[mask], dx[mask], dy[mask],
              angles='xy', scale_units='xy', scale=1/arrow_scale,
              color='blue', width=0.003, alpha=0.8)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Spatial Distribution of Random Cluster {chosen_cluster} with Orientations")
    ax.axis("equal")
    ax.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved cluster spatial distribution with arrows to {save_path}")

def plot_embedding_distribution(embeddings, x, y, save_path, map_size=40, random_state=42):
    """Plot embeddings in space using PCA RGB coloring."""
    pca = PCA(n_components=3, random_state=random_state)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Normalize to [0, 1] for RGB
    rgb = (embeddings_3d - embeddings_3d.min(axis=0)) / (embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_map_overlay(ax, map_size=map_size)
    ax.scatter(x, y, c=rgb, s=10, alpha=0.8)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Embedding Distribution in Space (PCA RGB)")
    ax.axis("equal")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved embedding distribution to {save_path}")

def plot_embedding_distribution_oriented(embeddings, x, y, theta, save_path, target_orientation=0, tolerance=10, map_size=40, random_state=42):
    """Plot embeddings in space, filtered by orientation, using PCA RGB coloring."""
    theta_mod = np.mod(theta, 360)
    diff = np.abs((theta_mod - target_orientation + 180) % 360 - 180)
    mask = diff <= tolerance
    
    if mask.sum() == 0:
        print("[WARN] No samples in orientation range.")
        return
    
    pca = PCA(n_components=3, random_state=random_state)
    embeddings_3d = pca.fit_transform(embeddings)
    embeddings_rgb = (embeddings_3d - embeddings_3d.min(axis=0)) / (embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0))
    
    x_f, y_f, colors_f = x[mask], y[mask], embeddings_rgb[mask]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_map_overlay(ax, map_size=map_size)
    ax.scatter(x_f, y_f, c=colors_f, s=15, alpha=0.8)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Embedding Distribution (Orientation {target_orientation}° ± {tolerance}°)")
    ax.axis("equal")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved oriented embedding distribution to {save_path}")