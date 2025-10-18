# summarize.py
import numpy as np
import os
import glob
from collections import Counter

def summarize_data(timestamp_dir):
    """Load and summarize data from the output directory for a given timestamp."""
    # Find all relevant files with zero-padded naming
    path_files = sorted(glob.glob(os.path.join(timestamp_dir, "path_*.npy")))
    action_files = sorted(glob.glob(os.path.join(timestamp_dir, "actions_*.npy")))
    lidar_files = sorted(glob.glob(os.path.join(timestamp_dir, "lidar_*.npy")))

    if not path_files:
        print(f"No data files found in {timestamp_dir}")
        return

    # Initialize accumulators
    total_steps = 0
    action_counts = Counter()
    total_intersections = 0

    # Load and summarize path data
    print("\nPath Data Summary:")
    for path_file in path_files:
        path = np.load(path_file, allow_pickle=True)
        steps = path.shape[0]
        total_steps += steps
        print(f"{path_file}: {steps} steps")
        if steps > 0:
            x, y, theta = path[:, 0], path[:, 1], path[:, 2]
            print(f"  X range: {x.min():.2f} to {x.max():.2f}")
            print(f"  Y range: {y.min():.2f} to {y.max():.2f}")
            print(f"  Orientation range: {theta.min():.2f} to {theta.max():.2f} degrees")

    # Load and summarize action data
    print("\nAction Data Summary:")
    for action_file in action_files:
        actions = np.load(action_file, allow_pickle=True)
        action_counts.update(actions)
        print(f"{action_file}: {len(actions)} actions")
    print("Action distribution:", dict(action_counts))

    # Load and summarize lidar data
    print("\nLidar Data Summary:")
    for lidar_file in lidar_files:
        lidar_data = np.load(lidar_file, allow_pickle=True)
        steps = len(lidar_data)
        intersections = sum(len([i for i in step if i is not None]) for step in lidar_data)
        total_intersections += intersections
        print(f"{lidar_file}: {steps} steps, {intersections} intersections")

    print(f"\nTotal Summary:")
    print(f"Total steps: {total_steps}")
    print(f"Total lidar intersections: {total_intersections}")

if __name__ == "__main__":
    # Example: Replace with the actual timestamp directory
    timestamp = "20250825-221040"  # Update with the correct timestamp
    output_dir = os.path.join("output", timestamp)
    if os.path.exists(output_dir):
        summarize_data(output_dir)
    else:
        print(f"Directory {output_dir} does not exist. Please provide a valid timestamp.")