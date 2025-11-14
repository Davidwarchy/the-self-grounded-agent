# analysis/plotting.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import cv2
import json
from analysis.utils import compute_correlations
import math
from scipy.stats import pearsonr, spearmanr

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
    
    return None

def plot_cluster_lidars(embeddings, lidar_data, x=None, y=None, theta=None, 
                       n_clusters=20, n_samples_per_cluster=10, min_temporal_sep=100,
                       save_path=None, show_map=False, data_dir=None, 
                       max_points=5000, random_state=42):
    """
    Cluster embeddings and plot LiDAR readings for each cluster.
    
    Args:
        embeddings: Embedding vectors (n_samples, embedding_dim)
        lidar_data: LiDAR readings (n_samples, n_rays)
        x, y, theta: Position and orientation data for map overlay
        n_clusters: Number of clusters for KMeans
        n_samples_per_cluster: Number of LiDAR samples to show per cluster
        min_temporal_sep: Minimum temporal separation between samples (frames)
        save_path: Path to save PDF
        show_map: Whether to show spatial distribution on map
        data_dir: Data directory for finding map image
        max_points: Maximum points to use for clustering
        random_state: Random seed for reproducibility
    """
    
    # Subsample if needed
    if max_points and len(embeddings) > max_points:
        print(f"[INFO] Subsampling to {max_points} points...")
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        lidar_data = lidar_data[indices]
        if x is not None: x = x[indices]
        if y is not None: y = y[indices]
        if theta is not None: theta = theta[indices]
    
    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    
    # Function to get temporally separated samples
    def get_temporally_separated_samples(cluster_indices, n_samples, min_sep):
        """Get samples with minimum temporal separation"""
        if len(cluster_indices) < 2:
            return cluster_indices[:n_samples]
        
        # Sort by index (assuming indices correspond to temporal order)
        sorted_indices = np.sort(cluster_indices)
        selected = [sorted_indices[0]]
        
        for idx in sorted_indices[1:]:
            if all(abs(idx - sel) >= min_sep for sel in selected):
                selected.append(idx)
                if len(selected) >= n_samples:
                    break
        
        return np.array(selected)
    
    # Prepare subplots
    if show_map and x is not None and y is not None:
        fig, axes = plt.subplots(2, n_clusters, figsize=(3*n_clusters, 6))
        if n_clusters == 1:
            axes = axes.reshape(2, 1)
    else:
        fig, axes = plt.subplots(1, n_clusters, figsize=(3*n_clusters, 3))
        if n_clusters == 1:
            axes = [axes]
    
    # Process each cluster
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        # Get temporally separated samples
        sample_indices = get_temporally_separated_samples(
            cluster_indices, n_samples_per_cluster, min_temporal_sep
        )
        
        if show_map and x is not None and y is not None:
            # Spatial distribution subplot
            ax_map = axes[0, cluster_id] if n_clusters > 1 else axes[0]
            ax_map.scatter(x, y, c='lightgray', s=5, alpha=0.3)
            ax_map.scatter(x[cluster_indices], y[cluster_indices], 
                          c='red', s=20, alpha=0.7)
            ax_map.scatter(x[sample_indices], y[sample_indices], 
                          c='blue', s=50, alpha=1.0, marker='x')
            ax_map.set_title(f'Cluster {cluster_id}\n{len(cluster_indices)} pts')
            ax_map.set_aspect('equal')
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            
            # LiDAR subplot
            ax_lidar = axes[1, cluster_id] if n_clusters > 1 else axes[1]
        else:
            ax_lidar = axes[cluster_id] if n_clusters > 1 else axes
        
        # Plot LiDAR readings
        for idx in sample_indices:
            ax_lidar.plot(lidar_data[idx], alpha=0.6, linewidth=1)
        
        if not show_map:
            ax_lidar.set_title(f'Cluster {cluster_id}\n{len(cluster_indices)} pts')
        ax_lidar.set_xlabel('Ray Index')
        if cluster_id == 0:
            ax_lidar.set_ylabel('Distance')
        ax_lidar.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved cluster LiDARs to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_cluster_on_map(embeddings, x, y, theta=None, n_clusters=50, 
                       arrow_scale=2, save_path=None, data_dir=None,
                       max_points=5000, random_state=42):
    """
    Cluster embeddings and plot spatial distribution of a random cluster on map.
    
    Args:
        embeddings: Embedding vectors
        x, y: Position data
        theta: Orientation data for arrows
        n_clusters: Number of clusters
        arrow_scale: Scale factor for orientation arrows
        save_path: Path to save PDF
        data_dir: Data directory for map image
        max_points: Maximum points for clustering
        random_state: Random seed
    """
    
    # Subsample if needed
    if max_points and len(embeddings) > max_points:
        print(f"[INFO] Subsampling to {max_points} points...")
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        x = x[indices]
        y = y[indices]
        if theta is not None: theta = theta[indices]
    
    # Cluster and pick random cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts > 3]
    
    if len(valid_clusters) == 0:
        print("[WARN] No valid clusters found.")
        return
    
    chosen_cluster = np.random.choice(valid_clusters)
    mask = labels == chosen_cluster
    
    # Find map image
    map_image_path = find_map_image_path(data_dir) if data_dir else None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add map background if available
    if map_image_path and os.path.exists(map_image_path):
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            map_img = np.flipud(map_img)
            ax.imshow(map_img, cmap='gray', 
                     extent=[0, map_img.shape[1], 0, map_img.shape[0]], 
                     alpha=0.7)
            ax.set_xlim(0, map_img.shape[1])
            ax.set_ylim(0, map_img.shape[0])
    
    # Plot all points and cluster points
    ax.scatter(x, y, c='lightgray', s=10, alpha=0.5, label="All positions")
    ax.scatter(x[mask], y[mask], c='red', s=30, alpha=0.9, 
               label=f"Cluster {chosen_cluster} ({mask.sum()} pts)")
    
    # Add orientation arrows if available
    if theta is not None:
        theta_rad = np.deg2rad(theta[mask])
        dx = np.cos(theta_rad)
        dy = np.sin(theta_rad)
        
        ax.quiver(x[mask], y[mask], dx, dy,
                  angles='xy', scale_units='xy', scale=1/arrow_scale,
                  color='blue', width=0.004, alpha=0.8, label='Orientation')
    
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Spatial Distribution of Cluster {chosen_cluster}")
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ensure (0,0) at bottom-left
    ax.set_ylim(ax.get_ylim()[::-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved cluster map to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_embeddings_rgb(embeddings, x, y, save_path=None, data_dir=None,
                       max_points=5000, random_state=42):
    """
    Plot embeddings in space using PCA-RGB coloring.
    
    Args:
        embeddings: Embedding vectors
        x, y: Position data
        save_path: Path to save PDF
        data_dir: Data directory for map image
        max_points: Maximum points to plot
        random_state: Random seed
    """
    
    # Subsample if needed
    if max_points and len(embeddings) > max_points:
        print(f"[INFO] Subsampling to {max_points} points...")
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        x = x[indices]
        y = y[indices]
    
    # Reduce to 3D RGB
    pca = PCA(n_components=3, random_state=random_state)
    embeddings_3d = pca.fit_transform(embeddings)
    rgb = (embeddings_3d - embeddings_3d.min(axis=0)) / (
        embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0))
    
    # Find map image
    map_image_path = find_map_image_path(data_dir) if data_dir else None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Add map background if available
    if map_image_path and os.path.exists(map_image_path):
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            map_img = np.flipud(map_img)
            ax.imshow(map_img, cmap='gray', 
                     extent=[0, map_img.shape[1], 0, map_img.shape[0]], 
                     alpha=0.7)
            ax.set_xlim(0, map_img.shape[1])
            ax.set_ylim(0, map_img.shape[0])
    
    # Plot embeddings with RGB coloring
    scatter = ax.scatter(x, y, c=rgb, s=30, alpha=0.8)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Embedding Distribution in Space (PCA RGB)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Ensure (0,0) at bottom-left
    ax.set_ylim(ax.get_ylim()[::-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved RGB embeddings to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_oriented_embeddings_rgb(embeddings, x, y, theta, target_orientation=0, 
                                tolerance=10, save_path=None, data_dir=None,
                                max_points=5000, random_state=42):
    """
    Plot embeddings filtered by orientation using PCA-RGB coloring.
    
    Args:
        embeddings: Embedding vectors
        x, y, theta: Position and orientation data
        target_orientation: Target orientation in degrees
        tolerance: Orientation tolerance in degrees
        save_path: Path to save PDF
        data_dir: Data directory for map image
        max_points: Maximum points to plot
        random_state: Random seed
    """
    
    # Filter by orientation
    theta_mod = np.mod(theta, 360)
    diff = np.abs((theta_mod - target_orientation + 180) % 360 - 180)
    mask = diff <= tolerance
    
    if mask.sum() == 0:
        print(f"[WARN] No samples in orientation range {target_orientation}±{tolerance}°")
        return
    
    # Apply filtering and subsample if needed
    x_f, y_f, embeddings_f = x[mask], y[mask], embeddings[mask]
    
    if max_points and len(embeddings_f) > max_points:
        print(f"[INFO] Subsampling to {max_points} points...")
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings_f), max_points, replace=False)
        x_f, y_f, embeddings_f = x_f[indices], y_f[indices], embeddings_f[indices]
    
    # Reduce to 3D RGB
    pca = PCA(n_components=3, random_state=random_state)
    embeddings_3d = pca.fit_transform(embeddings_f)
    rgb = (embeddings_3d - embeddings_3d.min(axis=0)) / (
        embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0))
    
    # Find map image
    map_image_path = find_map_image_path(data_dir) if data_dir else None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add map background if available
    if map_image_path and os.path.exists(map_image_path):
        map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is not None:
            map_img = np.flipud(map_img)
            ax.imshow(map_img, cmap='gray', 
                     extent=[0, map_img.shape[1], 0, map_img.shape[0]], 
                     alpha=0.7)
            ax.set_xlim(0, map_img.shape[1])
            ax.set_ylim(0, map_img.shape[0])
    
    # Plot filtered embeddings
    ax.scatter(x_f, y_f, c=rgb, s=15, alpha=0.8)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Embeddings (Orientation {target_orientation}° ± {tolerance}°)\n"
                f"{len(x_f)} samples")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Ensure (0,0) at bottom-left
    ax.set_ylim(ax.get_ylim()[::-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved oriented embeddings to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_correlations(features, embeddings=None, rgb=None, use_pca=True,
                     save_path=None, corr_type='pearson'):
    """
    Plot correlations between features and embeddings (full or PCA-reduced).
    
    Args:
        features: Dict of {feature_name: feature_values}
        embeddings: Full embeddings (if use_pca=False)
        rgb: PCA-reduced embeddings (if use_pca=True)
        use_pca: Whether to use PCA-reduced embeddings
        save_path: Path to save PDF
        corr_type: 'pearson' or 'spearman'
    """
    
    if use_pca:
        if rgb is None and embeddings is not None:
            # Compute PCA RGB if not provided
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(embeddings)
            rgb = (embeddings_3d - embeddings_3d.min(axis=0)) / (
                embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0))
        
        if rgb is None:
            raise ValueError("Either provide rgb or embeddings with use_pca=True")
        
        # For PCA mode, we correlate with RGB channels
        target_data = rgb
        n_dims = 3
        dim_labels = ['Red', 'Green', 'Blue']
        title_suffix = "PCA-RGB Embeddings"
        
    else:
        # For full embeddings mode
        if embeddings is None:
            raise ValueError("Must provide embeddings when use_pca=False")
        
        target_data = embeddings
        n_dims = min(embeddings.shape[1], 10)  # Limit to first 10 dimensions for readability
        dim_labels = [f'Dim {i}' for i in range(n_dims)]
        title_suffix = "Full Embeddings"
    
    # Compute correlations
    all_corrs = {}
    for name, vals in features.items():
        # Align lengths
        min_len = min(len(vals), len(target_data))
        vals_aligned = vals[:min_len]
        target_aligned = target_data[:min_len]
        
        if use_pca:
            # Correlate with RGB channels
            correlations = []
            for dim in range(n_dims):
                corr, _ = pearsonr(vals_aligned, target_aligned[:, dim])
                correlations.append(corr)
        else:
            # Correlate with embedding dimensions
            correlations = []
            for dim in range(n_dims):
                corr, _ = pearsonr(vals_aligned, target_aligned[:, dim])
                correlations.append(corr)
        
        all_corrs[name] = np.array(correlations)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(10, n_dims * 1.5), 6))
    
    x_pos = np.arange(n_dims)
    width = 0.8 / len(features)
    colors = plt.cm.Set3(np.linspace(0, 1, len(features)))
    
    for i, (name, corrs) in enumerate(all_corrs.items()):
        offset = width * (i - (len(features)-1)/2)
        bars = ax.bar(x_pos + offset, corrs, width, label=name, 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, corr in zip(bars, corrs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + 0.03 if height > 0 else height - 0.03,
                   f'{corr:.3f}', ha='center', 
                   va='bottom' if height > 0 else 'top',
                   fontsize=8, fontweight='bold')
    
    ax.set_xlabel("Dimension" if not use_pca else "RGB Channel")
    ax.set_ylabel(f"{corr_type.title()} Correlation")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dim_labels)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-1, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    ax.set_title(f"Feature Correlations with {title_suffix}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved correlations to {save_path}")
    else:
        plt.show()
    plt.close(fig)

# Convenience function for full analysis pipeline
def comprehensive_analysis(embeddings, x, y, theta, lidar_data, data_dir=None,
                          save_dir="analysis_output", max_points=5000, 
                          n_clusters=20, random_state=42):
    """
    Run comprehensive analysis and generate all plots.
    
    Args:
        embeddings, x, y, theta, lidar_data: Input data
        data_dir: Data directory for map images
        save_dir: Directory to save outputs
        max_points: Maximum points for each analysis
        n_clusters: Number of clusters
        random_state: Random seed
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Cluster LiDARs with temporal separation
    plot_cluster_lidars(
        embeddings, lidar_data, x, y, theta,
        n_clusters=n_clusters, min_temporal_sep=100,
        save_path=os.path.join(save_dir, "1_cluster_lidars.pdf"),
        show_map=True, data_dir=data_dir, max_points=max_points,
        random_state=random_state
    )
    
    # 2. Cluster spatial distribution with arrows
    plot_cluster_on_map(
        embeddings, x, y, theta,
        n_clusters=n_clusters, arrow_scale=2,
        save_path=os.path.join(save_dir, "2_cluster_spatial.pdf"),
        data_dir=data_dir, max_points=max_points,
        random_state=random_state
    )
    
    # 3. Full embeddings RGB
    plot_embeddings_rgb(
        embeddings, x, y,
        save_path=os.path.join(save_dir, "3_embeddings_rgb.pdf"),
        data_dir=data_dir, max_points=max_points,
        random_state=random_state
    )
    
    # 4. Oriented embeddings RGB (example: facing 180°)
    plot_oriented_embeddings_rgb(
        embeddings, x, y, theta, target_orientation=180, tolerance=10,
        save_path=os.path.join(save_dir, "4_oriented_180.pdf"),
        data_dir=data_dir, max_points=max_points,
        random_state=random_state
    )
    
    # 5. Correlations with human-intuitive features
    from analysis.utils import compute_features, align_arrays
    
    # Compute features
    dist_to_wall, openness, turn_intensity = compute_features(lidar_data, theta)
    
    # Align all arrays
    features_aligned = align_arrays(
        dist_to_wall, openness, turn_intensity, embeddings
    )
    dist_to_wall, openness, turn_intensity, embeddings_aligned = features_aligned
    
    features_dict = {
        "Distance to Wall": dist_to_wall,
        "Openness": openness,
        "Turn Intensity": turn_intensity
    }
    
    plot_correlations(
        features_dict, embeddings=embeddings_aligned,
        save_path=os.path.join(save_dir, "5_correlations.pdf"),
        corr_type='pearson'
    )
    
    print(f"[INFO] Comprehensive analysis complete! Results saved to {save_dir}")