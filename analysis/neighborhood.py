import numpy as np

def compare_random_vs_true_neighbors(n_points=10000, sample_size=100, k=100):
    # Generate points
    points = np.random.rand(n_points, 2) * 1000
    
    # Pick target point
    target_idx = np.random.randint(n_points)
    target = points[target_idx]

    # Pick 100 random points (ignore target idx if it appears)
    random_indices = np.random.choice(
        [i for i in range(n_points) if i != target_idx],
        sample_size,
        replace=False
    )

    # Compute all distances to target
    distances = np.sqrt(np.sum((points - target)**2, axis=1))
    
    # Find the k nearest points (excluding the target itself)
    nearest_indices = np.argsort(distances)[1:k+1]  # skip distance 0
    
    # How many of the random picks are in the true neighbors?
    overlap = len(set(random_indices).intersection(nearest_indices))
    
    return overlap / sample_size


# Run experiments
print("Experiment: Random vs. True Nearest Neighbors")
print("points\tsample\t% random in true 100")

for n_points in [1_000, 10_000, 100_000]:
    pct = compare_random_vs_true_neighbors(n_points=n_points)
    print(f"{n_points:,}\t100\t{pct:.4f}")
