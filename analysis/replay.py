import os
import glob
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_data(output_dir, n):
    # Find and sort intermediate log files by step number
    log_files = glob.glob(os.path.join(output_dir, "log_*.csv"))
    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    
    # Select first n files
    selected_files = log_files[:n]

    print("Selected files for animation:")
    for f in selected_files:
        print(f)

    data = []
    for i, file_path in enumerate(selected_files):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            if i > 0:
                next(reader)  # Skip header for subsequent files
            else:
                header = next(reader)  # Read header only once
            for row in reader:
                data.append(row)
    return data

def animate_lidar_readings(data):
    num_rays = 100  # From the environment code
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_rays - 1)
    ax.set_ylim(0, 200)  # Ray length is 200, set as max
    ax.set_xlabel('Ray Index')
    ax.set_ylabel('Distance')
    line, = ax.plot(range(num_rays), [0] * num_rays)
    
    def update(frame):
        row = data[frame]
        step = row[0]
        distances = [float(d) for d in row[2:102]]  # Skip step and action, take ray_0 to ray_99
        line.set_ydata(distances)
        ax.set_title(f"LIDAR Readings at Step {step}")
        # print(f"Frame {frame}: Step {step}")  # Debug output to confirm step changes
        return line,
    
    ani = FuncAnimation(fig, update, frames=len(data), interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    output_dir = "output/2025-10-18-175300"
    n = 5

    data = load_data(output_dir, n)
    if not data:
        print("No data found in the selected files.")
    else:
        animate_lidar_readings(data)