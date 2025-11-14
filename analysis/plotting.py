# analysis/plotting.py
# This script contains plotting functions

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
from analysis.utils import compute_correlations

# Map drawing helpers (from your code)
def get_map_objects(size=10):
    height = size
    width = size * 2
    radius = height / 10
    return {
        'walls': [
            [(0, 0), (width, 0)], [(0, height), (width, height)],
            [(0, 0), (0, height)], [(width, 0), (width, height)],
            [(.6*height, 0), (height, height/2)],
            [(1.1*height, 0), (height, height/2)],
            [(1.8*height, height), (width, .8*height)],
            [(.8*height, height), (.8*height, .9*height)],
            [(1.2*height, height), (1.2*height, .9*height)],
            [(.8*height, .9*height), (1.2*height, .9*height)],
            [(.1*height, 0), (.1*height, .05*height)],
            [(.1*height, .05*height), (.05*height, .1*height)],
            [(0, .1*height), (.05*height, .1*height)],
        ],
        'circles': [
            {'center': (radius*2, height - radius*2), 'radius': radius*1.3},
            {'center': (width - radius*2, radius*2), 'radius': radius}
        ]
    }

def plot_map_overlay(ax, map_size=10):
    objs = get_map_objects(size=map_size)
    for wall in objs['walls']:
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', lw=2, alpha=0.6)
    for c in objs['circles']:
        ax.add_patch(plt.Circle(c['center'], c['radius'], fill=False, color='black', lw=2, alpha=0.6))

def plot_embeddings_on_map(x, y, rgb, map_size=40, max_points=None, save_path=None):
    """Plot embeddings overlaid on map."""
    if max_points:
        print(f"[INFO] Subsampling to {max_points} points for plotting... len={len(x)}")
        idxs = np.random.choice(len(x), min(max_points, len(x)), replace=False)
        x, y, rgb = x[idxs], y[idxs], rgb[idxs]

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_map_overlay(ax, map_size=map_size)
    ax.scatter(x, y, c=rgb, s=30, alpha=0.8)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf')
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_oriented_embeddings(x, y, theta, rgb, target_orientation=0, tolerance=10, map_size=40, save_path=None):
    """Plot embeddings filtered by orientation."""
    diff = np.abs((theta - target_orientation + 180) % 360 - 180)
    mask = diff <= tolerance
    if mask.sum() == 0:
        print("[WARN] No samples in orientation range.")
        return

    x_f, y_f, colors_f = x[mask], y[mask], rgb[mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_map_overlay(ax, map_size=map_size)
    ax.scatter(x_f, y_f, c=colors_f, s=15, alpha=0.8)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf')
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_random_cluster(x, y, embeddings, n_clusters=50, arrow_scale=None, map_size=40, save_path=None):
    """Plot spatial distribution of a random cluster, optionally with orientation arrows."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    unique, counts = np.unique(labels, return_counts=True)
    valid = unique[counts > 3]
    if len(valid) == 0:
        print("[WARN] No valid clusters.")
        return

    chosen = np.random.choice(valid)
    mask = labels == chosen

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_map_overlay(ax, map_size=map_size)
    ax.scatter(x, y, c='lightgray', s=10, alpha=0.5)
    ax.scatter(x[mask], y[mask], c='red', s=30, alpha=0.9)

    if arrow_scale is not None:
        # Assuming theta is available; pass it if needed
        pass  # Add quiver logic here if theta provided

    ax.axis('equal')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf')
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
        plt.savefig(save_path, format='pdf')
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)