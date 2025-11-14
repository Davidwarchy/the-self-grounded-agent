import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_map_image(map_image_path, target_size=(358, 358)):
    """Load and prepare map image for visualization"""
    image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {map_image_path}")
    
    # Resize to target dimensions
    image = cv2.resize(image, target_size)
    return image

def plot_image_environment(map_image_path, exploration_grid=None, robot_pos=None, save_path=None):
    """Plot the image-based environment with optional exploration overlay"""
    
    map_image = load_map_image(map_image_path)
    height, width = map_image.shape

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_aspect('equal')
    
    # Display the base map
    ax.imshow(map_image, cmap='gray', extent=[0, width, 0, height])
    
    # Overlay exploration data if provided
    if exploration_grid is not None:
        explored_free = np.ma.masked_where(exploration_grid != 0, exploration_grid >= 0)
        explored_obstacle = np.ma.masked_where(exploration_grid != 1, exploration_grid >= 0)
        
        ax.imshow(explored_free, cmap='Greens', alpha=0.3, extent=[0, width, 0, height])
        ax.imshow(explored_obstacle, cmap='Reds', alpha=0.3, extent=[0, width, 0, height])
    
    # Plot robot position if provided
    if robot_pos is not None:
        x, y, orientation = robot_pos
        ax.plot(x, y, 'bo', markersize=8, label='Robot')
        # Draw orientation arrow
        dx = 10 * np.cos(np.radians(orientation))
        dy = 10 * np.sin(np.radians(orientation))
        ax.arrow(x, y, dx, dy, head_width=3, head_length=2, fc='red', ec='red')
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Image-Based Simulation Environment')
    ax.grid(True, alpha=0.3)
    
    if robot_pos is not None:
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    map_path = r"C:\Users\HP\Desktop\Projects\navigation\9-daniel-cremers-random-motion-collect\environments\images\1.png"
    plot_image_environment(map_path)