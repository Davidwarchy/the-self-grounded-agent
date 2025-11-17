import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from create_parquet import load_data


# Configuration
DATA_DIR = "output/2025-11-14-111925_random_walk_100k"
N_STEPS = 100000
N_NEIGHBORS = 1000  # for original preservation test

print(f"""Loading data from {DATA_DIR}
Processing {N_STEPS} steps
Using {N_NEIGHBORS} neighbors for analysis
""")


# ----------------------------------------
# Encoder
# ----------------------------------------
def create_simple_encoder():
    import torch
    import torch.nn as nn

    class SimpleEncoder(nn.Module):
        def __init__(self, input_dim=100, hidden_dim=64, output_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            return self.net(x)

    return SimpleEncoder()


def get_embeddings(lidar_data):
    import torch

    model = create_simple_encoder()
    lidar_tensor = torch.from_numpy(lidar_data).float()
    with torch.no_grad():
        embeddings = model(lidar_tensor).numpy()

    return embeddings


# ----------------------------------------
# Neighborhood preservation
# ----------------------------------------
def calculate_neighborhood_preservation(xy_space, embedding_space, n_neighbors=10):
    nn_xy = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn_emb = NearestNeighbors(n_neighbors=n_neighbors + 1)

    nn_xy.fit(xy_space)
    nn_emb.fit(embedding_space)

    xy_neighbors = nn_xy.kneighbors(xy_space, return_distance=False)[:, 1:]
    emb_neighbors = nn_emb.kneighbors(embedding_space, return_distance=False)[:, 1:]

    overlaps = []
    for i in tqdm(range(len(xy_space)), desc="Calculating preservation"):
        xy_nn = set(xy_neighbors[i])
        emb_nn = set(emb_neighbors[i])
        overlap = len(xy_nn.intersection(emb_nn))
        overlaps.append(overlap / n_neighbors)

    return np.mean(overlaps), np.std(overlaps), overlaps


# ----------------------------------------
# Oriented preservation
# ----------------------------------------
def calculate_oriented_neighborhood_preservation(
    xy_space, orientations, embedding_space,
    target_orientation=None, tolerance=30, n_neighbors=10
):
    if target_orientation is None:
        mask = np.ones(len(xy_space), dtype=bool)
    else:
        theta_mod = np.mod(orientations, 360)
        diff = np.abs((theta_mod - target_orientation + 180) % 360 - 180)
        mask = diff <= tolerance

    if np.sum(mask) < n_neighbors + 1:
        return 0, 0, []

    filtered_xy = xy_space[mask]
    filtered_emb = embedding_space[mask]
    original_indices = np.where(mask)[0]

    idx_mapping = {i: orig_idx for i, orig_idx in enumerate(original_indices)}

    nn_xy = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(filtered_xy)))
    nn_emb = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(filtered_emb)))

    nn_xy.fit(filtered_xy)
    nn_emb.fit(filtered_emb)

    xy_neighbors = nn_xy.kneighbors(filtered_xy, return_distance=False)[:, 1:]
    emb_neighbors = nn_emb.kneighbors(filtered_emb, return_distance=False)[:, 1:]

    overlaps = []
    for i in tqdm(range(len(filtered_xy)), desc=f"Oriented preservation {target_orientation}°"):
        xy_nn_original = set(idx_mapping[idx] for idx in xy_neighbors[i])
        emb_nn_original = set(idx_mapping[idx] for idx in emb_neighbors[i])
        overlap = len(xy_nn_original.intersection(emb_nn_original))
        overlaps.append(overlap / min(n_neighbors, len(xy_neighbors[i])))

    return np.mean(overlaps), np.std(overlaps), overlaps


# ----------------------------------------
# MULTISCALE neighborhood preservation
# ----------------------------------------
def neighborhood_preservation_multiscale(
    xy_space,
    embedding_space,
    ks=[10, 100, 1000, 10000],
    n_targets=2000
):
    N = len(xy_space)
    idx_targets = np.random.choice(N, size=min(n_targets, N), replace=False)

    nn_xy = NearestNeighbors(n_neighbors=max(ks) + 1).fit(xy_space)
    nn_emb = NearestNeighbors(n_neighbors=max(ks) + 1).fit(embedding_space)

    xy_neighbors_all = nn_xy.kneighbors(xy_space[idx_targets], return_distance=False)[:, 1:]
    emb_neighbors_all = nn_emb.kneighbors(embedding_space[idx_targets], return_distance=False)[:, 1:]

    results = {}

    print("\n=== MULTISCALE NEIGHBORHOOD PRESERVATION ===")

    for k in ks:
        xy_k = xy_neighbors_all[:, :k]
        emb_k = emb_neighbors_all[:, :k]

        overlaps = []
        for i in tqdm(range(len(idx_targets)), desc=f"k={k}"):
            overlaps.append(len(set(xy_k[i]).intersection(set(emb_k[i]))) / k)

        mean_pres = float(np.mean(overlaps))
        std_pres = float(np.std(overlaps))
        random_baseline = k / N

        results[k] = {
            "mean": mean_pres,
            "std": std_pres,
            "random_baseline": random_baseline,
        }

        print(f"k={k:5d} | preservation={mean_pres:.4f} ± {std_pres:.4f} | random={random_baseline:.6f}")

    return results


# ----------------------------------------
# Plotting
# ----------------------------------------
def plot_neighborhood_analysis(preservation_results, oriented_results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    mean_pres, std_pres, individual_pres = preservation_results
    ax1.hist(individual_pres, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax1.axvline(mean_pres, color='red', linestyle='--', linewidth=2)
    ax1.set_title(f"Overall Neighborhood Preservation (n={N_NEIGHBORS})")

    orientations_to_test = [0, 90, 180, 270]
    means = [oriented_results[o][0] for o in orientations_to_test]
    errs = [oriented_results[o][1] for o in orientations_to_test]

    ax2.bar(range(4), means, yerr=errs, capsize=5, color='orange', edgecolor='black')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"{o}°" for o in orientations_to_test])
    ax2.set_title("Oriented Neighborhood Preservation")

    ax3.hist(individual_pres, bins=20, cumulative=True, density=True, edgecolor='black')
    ax3.set_title("CDF of Neighborhood Preservation")

    ax4.axis('off')
    txt = "Neighborhood Preservation Summary\n"
    txt += f"Mean: {mean_pres:.4f}\nStd: {std_pres:.4f}\n\n"
    for o in orientations_to_test:
        txt += f"{o}°: {oriented_results[o][0]:.4f}\n"
    ax4.text(0.05, 0.95, txt, va="top")

    plt.tight_layout()
    plt.show()


# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    parquet_path = load_data(DATA_DIR, n=1000)
    df = pd.read_parquet(parquet_path).head(N_STEPS)

    print(f"Loaded {len(df)} steps of data")

    lidar_cols = [f'ray_{i}' for i in range(100)]
    lidar_data = df[lidar_cols].values
    xy_positions = df[['x', 'y']].values
    orientations = df['orientation'].values

    embeddings = get_embeddings(lidar_data)

    print("Extracted embeddings")

    # Original preservation
    preservation_results = calculate_neighborhood_preservation(
        xy_positions, embeddings, N_NEIGHBORS
    )

    # Oriented preservation
    oriented_results = {}
    for o in [0, 90, 180, 270]:
        oriented_results[o] = calculate_oriented_neighborhood_preservation(
            xy_positions, orientations, embeddings,
            target_orientation=o, tolerance=30, n_neighbors=N_NEIGHBORS
        )

    # Quick summary
    print("\n=== QUICK SUMMARY ===")
    print(f"Overall preservation: {preservation_results[0]:.4f}")
    for o in [0, 90, 180, 270]:
        print(f"{o}°: {oriented_results[o][0]:.4f}")

    # MULTISCALE
    multiscale_results = neighborhood_preservation_multiscale(
        xy_positions, embeddings,
        ks=[10, 100, 1000, 10000],
        n_targets=2000
    )

    # Plot
    plot_neighborhood_analysis(preservation_results, oriented_results)
