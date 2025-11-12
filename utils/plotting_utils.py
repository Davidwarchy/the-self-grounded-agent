import numpy as np
import matplotlib.pyplot as plt

def get_map_objects(size=10):
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

def create_robot(x, y, orientation, size=1):
    angle = orientation * (3.14159 / 180)
    octagon = [
        (x + size * 0.707 * np.cos(angle + i * 3.14159 / 4), y + size * 0.707 * np.sin(angle + i * 3.14159 / 4))
        for i in range(8)
    ]
    octagon.append(octagon[0])
    front_x = x + size * .626 * np.cos(angle)
    front_y = y + size * .626 * np.sin(angle)
    lidar = [(x, y), (front_x, front_y)]
    return octagon, lidar

def plot_environment(map_size=40, save_path=None):
    """Plots the robot environment map with legible text."""
    
    room = get_map_objects(size=map_size)
    width = map_size * 2
    height = map_size

    # Compact figure but high resolution
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_aspect('equal')
    
    # Draw walls
    for wall in room['walls']:
        (x0, y0), (x1, y1) = wall
        ax.plot([x0, x1], [y0, y1], 'k-', linewidth=2)
    
    # Draw circles
    for circle in room['circles']:
        cx, cy = circle['center']
        r = circle['radius']
        ax.add_artist(plt.Circle((cx, cy), r, fill=False, color='b', linewidth=1.5))
    
    # Limits and labels
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel('X (units)', fontsize=10)
    ax.set_ylabel('Y (units)', fontsize=10)
    ax.set_title('Simulation Environment', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_environment(map_size=80)