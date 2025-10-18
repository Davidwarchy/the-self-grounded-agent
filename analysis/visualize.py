# visualize.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.plotting_utils import get_map_objects

def visualize_data(timestamp_dir, interval_file="path_000100.npy", sample_step=0):
    """Visualize robot path and lidar data for a specific interval file."""
    # Load map objects
    map_size = 40
    room = get_map_objects(map_size)

    # Load path and lidar data
    path_file = os.path.join(timestamp_dir, interval_file)
    lidar_file = os.path.join(timestamp_dir, interval_file.replace("path_", "lidar_"))
    
    if not os.path.exists(path_file) or not os.path.exists(lidar_file):
        print(f"Files {path_file} or {lidar_file} not found.")
        return

    try:
        path = np.load(path_file, allow_pickle=True)
        lidar_data = np.load(lidar_file, allow_pickle=True)
        
        if path.size == 0 or lidar_data.size == 0:
            print(f"One or both files ({path_file}, {lidar_file}) are empty.")
            return

        # Plot map
        fig, ax = plt.subplots(figsize=(10, 5))
        for wall in room['walls']:
            ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', linewidth=2)
        for circle in room['circles']:
            circle_patch = plt.Circle(circle['center'], circle['radius'], fill=False, color='k', linewidth=2)
            ax.add_patch(circle_patch)

        # Plot robot path
        x, y = path[:, 0], path[:, 1]
        ax.plot(x, y, 'b-', label="Robot Path", alpha=0.5)

        # Plot lidar rays for a single step (sample_step)
        if sample_step < len(lidar_data):
            robot_x, robot_y = path[sample_step, 0], path[sample_step, 1]
            intersections = lidar_data[sample_step]
            for intersection in intersections:
                if intersection is not None:
                    ax.plot([robot_x, intersection[0]], [robot_y, intersection[1]], 'g-', alpha=0.3)
            ax.plot(robot_x, robot_y, 'ro', label="Robot Position")

        # Set plot properties
        ax.set_xlim(0, map_size * 2)
        ax.set_ylim(0, map_size)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Robot Path and Lidar Rays (Step {sample_step} of {interval_file})")
        ax.legend()
        ax.set_aspect('equal')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error processing {path_file} or {lidar_file}: {e}")

if __name__ == "__main__":
    # Update with the actual timestamp directory
    timestamp = "20250825-221040"  # Matches your output
    output_dir = os.path.join("output", timestamp)
    if os.path.exists(output_dir):
        visualize_data(output_dir, interval_file="path_000100.npy", sample_step=99)
    else:
        print(f"Directory {output_dir} does not exist. Please provide a valid timestamp.")