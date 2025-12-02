import numpy as np
import math

def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's Line Algorithm to generate a list of integer coordinates 
    approximating a straight line between two points.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            
    points.append((x, y))
    return points

def cast_rays(robot_x, robot_y, robot_angle, obstacle_map, num_rays=100, max_range=200, fov_deg=90):
    """
    Casts rays from the robot's position to detect obstacles.
    
    Args:
        robot_x, robot_y: Robot position (pixels).
        robot_angle: Robot orientation (degrees).
        obstacle_map: 2D numpy array (0=free, 1=obstacle).
        num_rays: Number of rays to cast.
        max_range: Maximum distance of a ray.
        fov_deg: Field of view in degrees.
        
    Returns:
        distances: List of float distances for each ray.
        intersections: List of (x, y) tuples where rays hit (or None).
    """
    height, width = obstacle_map.shape
    distances = []
    
    # Calculate ray angles (centered around robot_angle)
    # Fov span: robot_angle - fov/2 to robot_angle + fov/2
    start_angle = robot_angle - (fov_deg / 2)
    end_angle = robot_angle + (fov_deg / 2)
    ray_angles = np.linspace(start_angle, end_angle, num_rays)

    for angle in ray_angles:
        rad = np.radians(angle)
        
        # Calculate target endpoint for the ray
        target_x = int(robot_x + max_range * np.cos(rad))
        target_y = int(robot_y + max_range * np.sin(rad))
        
        # Get all pixels along the line
        line_points = bresenham_line(int(robot_x), int(robot_y), target_x, target_y)
        
        hit_dist = max_range
        
        for px, py in line_points:
            # Check bounds
            if px < 0 or px >= width or py < 0 or py >= height:
                # Wall hit (bounds)
                hit_dist = math.sqrt((px - robot_x)**2 + (py - robot_y)**2)
                break
            
            # Check obstacle
            if obstacle_map[py, px] == 1:
                hit_dist = math.sqrt((px - robot_x)**2 + (py - robot_y)**2)
                break
        
        distances.append(hit_dist)

    return np.array(distances)