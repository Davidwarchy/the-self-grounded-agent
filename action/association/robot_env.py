import numpy as np
import cv2
import pygame
import os
import math
from numba import njit

# ==========================================
# 1. GEOMETRY: Get points on a line
# ==========================================
@njit
def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm for efficient pixel traversal.
    Returns a list of (x, y) coordinates connecting the two points.
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

# ==========================================
# 2. INTERSECTIONS: Find where rays hit walls
# ==========================================
@njit
def get_ray_intersections(x, y, angle, lidar_angles, max_range, map_data, width, height):
    """
    Casts rays from the robot's position and finds wall intersections.
    """
    num_rays = len(lidar_angles)
    hits = np.full((num_rays, 2), -1, dtype=np.int32)
    
    # Pre-calculate all ray angles in radians
    ray_angles = np.radians(angle + lidar_angles)
    
    for i in range(num_rays):
        ray_angle = ray_angles[i]
        
        # Calculate theoretical end of ray (max range)
        end_x = int(x + max_range * np.cos(ray_angle))
        end_y = int(y + max_range * np.sin(ray_angle))
        
        # 1. Get Geometry (Bresenham)
        line_points = bresenham_line(int(x), int(y), end_x, end_y)
        
        # 2. Check for Obstacles along the line
        for point in line_points:
            px, py = point
            
            # Check map bounds
            if not (0 <= px < width and 0 <= py < height):
                break
            
            # Check Obstacle (1 is wall)
            if map_data[py, px] == 1:
                hits[i][0] = px
                hits[i][1] = py
                break # Stop at first wall hit
                
    return hits

# ==========================================
# 3. DISTANCES: Convert hits to scalar readings
# ==========================================
@njit
def get_ray_distances(x, y, intersections, max_range):
    """
    Converts intersection coordinates into scalar distances for the agent.
    """
    num_rays = len(intersections)
    distances = np.zeros(num_rays, dtype=np.float32)
    
    for i in range(num_rays):
        ix, iy = intersections[i]
        
        if ix == -1: 
            # No hit -> Max Range
            distances[i] = max_range
        else:
            # Hit -> Calculate Euclidean distance
            dist = math.sqrt((ix - x)**2 + (iy - y)**2)
            distances[i] = dist
            
    return distances

# ==========================================
# 4. DOWN SAMPLE AND DISCRETIZE RAYS
# ==========================================
@njit
def downsample_and_discretize(ray_distances, num_outputs=3, num_levels=3, max_range=200):
    """
    Downsamples ray distances by averaging groups and discretizes to levels.
    
    Args:
        ray_distances: Array of length N (original ray distances)
        num_outputs: Number of output values (k)
        num_levels: Number of discrete levels (default 3: close, medium, far)
        max_range: Maximum range for normalization
        
    Returns:
        Array of length num_outputs with discretized values (0, 1, 2)
    """
    N = len(ray_distances)
    outputs = np.zeros(num_outputs, dtype=np.int32)
    
    # Determine group size
    group_size = N // num_outputs
    remainder = N % num_outputs
    
    # Calculate thresholds for discretization
    # Example for 3 levels: 
    # level 0: [0, max_range/3) - close
    # level 1: [max_range/3, 2*max_range/3) - medium
    # level 2: [2*max_range/3, max_range] - far
    thresholds = np.linspace(0, max_range, num_levels + 1)
    
    start_idx = 0
    for i in range(num_outputs):
        # Adjust group size for remainder
        actual_group_size = group_size + (1 if i < remainder else 0)
        end_idx = start_idx + actual_group_size
        
        # Calculate average of the group
        group_avg = np.mean(ray_distances[start_idx:end_idx])
        
        # Discretize based on thresholds
        level = 0
        for l in range(num_levels):
            if group_avg >= thresholds[l] and group_avg < thresholds[l+1]:
                level = l
                break
        # Handle edge case (exactly max_range)
        if group_avg >= thresholds[-1]:
            level = num_levels - 1
            
        outputs[i] = level
        start_idx = end_idx
        
    return outputs

# ==========================================
# 5. MAP UPDATE: Fog of War logic
# ==========================================
@njit
def update_exploration_grid_fast(x, y, angle, lidar_angles, intersections, ray_length, grid, width, height):
    """
    Updates the exploration grid based on ray hits.
    Marks cells along the ray as 0 (Free) and hit points as 1 (Obstacle).
    """
    new_cells = 0
    num_rays = len(lidar_angles)
    ray_angles = np.radians(angle + lidar_angles)
    
    for i in range(num_rays):
        ix, iy = intersections[i]
        
        # Determine the target point for the "clear" line
        if ix != -1:
            target_x, target_y = ix, iy
        else:
            target_x = int(x + ray_length * np.cos(ray_angles[i]))
            target_y = int(y + ray_length * np.sin(ray_angles[i]))

        # Get the full line of pixels
        line_points = bresenham_line(int(x), int(y), target_x, target_y)
        
        # 1. Mark Free Space
        for point in line_points:
            px, py = point
            
            # Bounds check
            if 0 <= px < width and 0 <= py < height:
                
                # If this is the obstacle hit point, skip marking it as free
                if px == ix and py == iy:
                    continue
                
                # If unexplored (-1), mark as free (0)
                if grid[px, py] == -1:
                    grid[px, py] = 0
                    new_cells += 1
        
        # 2. Mark Obstacle (only if we actually hit something)
        if ix != -1:
            if 0 <= ix < width and 0 <= iy < height:
                if grid[ix, iy] == -1:
                    grid[ix, iy] = 1
                    new_cells += 1
                    
    return new_cells

class DownsampledRobotEnv:
    def __init__(self, map_path, max_steps=1000, robot_radius=3, goal_radius=2, render_mode=False, scale_factor=20):
        # 1. Load Map
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map not found at: {map_path}")
            
        self.map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise ValueError("Failed to load map image.")
            
        # 2. Process Map (0=Free, 1=Obstacle)
        _, self.binary_map = cv2.threshold(self.map_img, 127, 1, cv2.THRESH_BINARY_INV)
        self.h, self.w = self.binary_map.shape
        self.obstacle_map = self.binary_map.astype(np.int32)
        
        # 3. Parameters
        self.max_steps = max_steps
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.render_mode = render_mode
        self.scale_factor = scale_factor  # NEW: Magnification factor
        
        self.step_count = 0
        self.episode = 0
        
        # Robot State (fixed start for environment 12)
        self.initial_robot_x, self.initial_robot_y, self.initial_robot_angle = 5, 15, np.pi/4
        self.x = self.initial_robot_x
        self.y = self.initial_robot_y 
        self.angle = self.initial_robot_angle
        
        # Goal State (Fixed for all episodes at 15, 5 as requested)
        self.initial_goal_x, self.initial_goal_y = 15, 5
        self.goal_x = self.initial_goal_x
        self.goal_y = self.initial_goal_y
        
        # Constants & Config
        self.SPEED = 5.0
        self.TURN_SPEED = 15.0 # Degrees
        self.num_rays = 100
        self.ray_length = 10
        
        # Downsampling parameters
        self.num_outputs = 3  # Output 3 rays after downsampling
        self.num_levels = 3   # 3 discretization levels
        
        # Lidar Angles
        self.lidar_angles = np.linspace(-45, 45, self.num_rays)
        self.last_intersections = None
        
        # Exploration Grid (-1=unexplored, 0=free, 1=obstacle)
        self.exploration_grid = None
        
        # Visualization
        self.screen = None
        self.clock = None

    def reset(self):
        """Resets the robot and exploration grid. Goal remains fixed."""
        self.step_count = 0
        self.episode += 1
        
        # Initialize exploration grid
        self.exploration_grid = np.full((self.w, self.h), -1, dtype=np.int32)
        
        # Reset robot to initial position
        self.x, self.y, self.angle = self.initial_robot_x, self.initial_robot_y, self.initial_robot_angle
        
        # Initial Lidar Update (Physics)
        intersections = get_ray_intersections(
            self.x, self.y, self.angle, 
            self.lidar_angles, self.ray_length, 
            self.obstacle_map, self.w, self.h
        )
        self.last_intersections = intersections
        
        # Update Visualization only if needed
        if self.render_mode:
            self._update_exploration_grid(intersections)
        
        return self._get_observation(intersections)

    def step(self, action):
        self.step_count += 1
        prev_x, prev_y = self.x, self.y
        
        # Movement (5 actions: 0=up, 1=down, 2=left, 3=right, 4=nothing)
        if action == 0:   # Up (forward relative to orientation)
            self.x += self.SPEED * np.cos(np.radians(self.angle))
            self.y += self.SPEED * np.sin(np.radians(self.angle))
        elif action == 1: # Down (backward)
            self.x -= self.SPEED * np.cos(np.radians(self.angle))
            self.y -= self.SPEED * np.sin(np.radians(self.angle))
        elif action == 2: # Left (turn left)
            self.angle = (self.angle - self.TURN_SPEED) % 360
        elif action == 3: # Right (turn right)
            self.angle = (self.angle + self.TURN_SPEED) % 360
        elif action == 4: # Nothing (stay in place)
            pass
            
        # Collision Check
        if not self._is_clear(self.x, self.y, self.robot_radius):
            self.x, self.y = prev_x, prev_y
            
        # 1. Physics: Get Lidar Intersections
        intersections = get_ray_intersections(
            self.x, self.y, self.angle, 
            self.lidar_angles, self.ray_length, 
            self.obstacle_map, self.w, self.h
        )
        self.last_intersections = intersections
        
        # 2. Visualization: Update Fog of War (Optional)
        if self.render_mode:
            self._update_exploration_grid(intersections)
            
        # 3. Observation: Get downsampled and discretized distances
        obs = self._get_observation(intersections)
        
        # Reward & Done
        dist_to_goal = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        touch_threshold = self.robot_radius + self.goal_radius
        
        reward = 0.0
        done = False
        
        if dist_to_goal < touch_threshold:
            reward = 1.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True
        
        return obs, reward, done, {}

    # --- Core Logic ---

    def _update_exploration_grid(self, intersections):
        """Updates the fog-of-war grid."""
        update_exploration_grid_fast(
            self.x, self.y, self.angle, 
            self.lidar_angles, intersections, self.ray_length,
            self.exploration_grid, self.w, self.h
        )

    def _get_observation(self, intersections):
        """
        Gets distance readings, downsamples to 3 outputs, and discretizes to 3 levels.
        Returns array of integers (0, 1, or 2) for each output.
        """
        # Get raw distances
        raw_distances = get_ray_distances(self.x, self.y, intersections, self.ray_length)
        
        # Downsample and discretize
        obs = downsample_and_discretize(
            raw_distances, 
            num_outputs=self.num_outputs, 
            num_levels=self.num_levels, 
            max_range=self.ray_length
        )
        
        return obs

    # --- Helpers ---

    def _is_clear(self, x, y, radius):
        """Check collision with map boundaries and obstacles."""
        ix, iy = int(x), int(y)
        if ix < radius or ix >= self.w - radius: return False
        if iy < radius or iy >= self.h - radius: return False
        
        if self.obstacle_map[iy, ix] == 1: return False
        
        # Check surrounding points (simplified circle check)
        r = int(radius)
        # Check cardinal directions
        if self.obstacle_map[min(self.h-1, iy+r), ix] == 1: return False
        if self.obstacle_map[max(0, iy-r), ix] == 1: return False
        if self.obstacle_map[iy, min(self.w-1, ix+r)] == 1: return False
        if self.obstacle_map[iy, max(0, ix-r)] == 1: return False
        
        return True

    # --- Rendering ---

    def render(self):
        # Prevent pygame initialization if rendering is disabled
        if not self.render_mode:
            return False

        if self.screen is None:
            pygame.init()
            # NEW: Create window with scaled dimensions
            self.screen = pygame.display.set_mode((self.w * self.scale_factor, self.h * self.scale_factor))
            pygame.display.set_caption(f"Robot Exploration - Downsampled (Scale: {self.scale_factor}x)")
            self.clock = pygame.time.Clock()
        
        self._draw_map()
        self._draw_entities()
        self._draw_lidar()
        
        pygame.display.flip()
        self.clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def _draw_map(self):
        # Base map
        vis_map = np.stack([self.map_img] * 3, axis=-1)
        base_surf = pygame.surfarray.make_surface(vis_map.swapaxes(0, 1))
        
        # NEW: Scale the surface
        scaled_surf = pygame.transform.scale(base_surf, (self.w * self.scale_factor, self.h * self.scale_factor))
        self.screen.blit(scaled_surf, (0, 0))
        
        # Draw Exploration Grid
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        grid_t = self.exploration_grid
        
        # Create masks
        free_mask = (grid_t == 0)
        obs_mask = (grid_t == 1)
        
        # Access pixel array directly
        rgb_ref = pygame.surfarray.pixels3d(overlay)
        alpha_ref = pygame.surfarray.pixels_alpha(overlay)
        
        # Green for Free
        rgb_ref[free_mask] = [0, 255, 0]
        alpha_ref[free_mask] = 100
        
        # Red for Obstacles
        rgb_ref[obs_mask] = [255, 0, 0]
        alpha_ref[obs_mask] = 100
        
        del rgb_ref
        del alpha_ref
        
        # NEW: Scale the overlay
        scaled_overlay = pygame.transform.scale(overlay, (self.w * self.scale_factor, self.h * self.scale_factor))
        self.screen.blit(scaled_overlay, (0, 0))

    def _draw_entities(self):
        # NEW: Scale all coordinates and sizes
        s = self.scale_factor
        
        # Goal (Blue Blob) at 15, 5
        pygame.draw.circle(self.screen, (0, 0, 255), 
                          (int(self.goal_x * s), int(self.goal_y * s)), 
                          self.goal_radius * s)
        
        # Robot (Yellow Circle)
        pygame.draw.circle(self.screen, (255, 255, 0), 
                          (int(self.x * s), int(self.y * s)), 
                          self.robot_radius * s)
        
        # Heading Line
        end_x = (self.x + 15 * np.cos(np.radians(self.angle))) * s
        end_y = (self.y + 15 * np.sin(np.radians(self.angle))) * s
        pygame.draw.line(self.screen, (0, 0, 0), 
                        (self.x * s, self.y * s), 
                        (end_x, end_y), 
                        max(2, int(2 * s / 10)))

    def _draw_lidar(self):
        if self.last_intersections is None:
            return

        s = self.scale_factor
        angles = self.angle + self.lidar_angles
        
        # Iterate through the numpy array of intersections
        for i, angle_deg in enumerate(angles):
            ix, iy = self.last_intersections[i]
            
            # Check if we have a valid hit (-1 check)
            if ix != -1:
                end_pos = (ix * s, iy * s)
            else:
                # Calculate max range end point for visualization
                angle_rad = np.radians(angle_deg)
                end_x = (self.x + self.ray_length * np.cos(angle_rad)) * s
                end_y = (self.y + self.ray_length * np.sin(angle_rad)) * s
                end_pos = (end_x, end_y)
            
            # Draw Ray
            pygame.draw.line(self.screen, (0, 255, 255), 
                           (self.x * s, self.y * s), 
                           end_pos, 
                           max(1, int(s / 20)))