import numpy as np
import cv2
import pygame
import os
import math
from numba import njit

@njit 
def _bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm for efficient pixel traversal.

    :param x0: Start x
    :param y0: Start y
    :param x1: End x
    :param y1: End y    

    :return: List of (x, y) points along the line
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

class SimpleRobotEnv:
    def __init__(self, map_path, max_steps=1000, robot_radius=3, goal_radius=10, render_mode=False):
        # 1. Load Map
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map not found at: {map_path}")
            
        self.map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise ValueError("Failed to load map image.")
            
        # 2. Process Map (0=Free, 1=Obstacle)
        _, self.binary_map = cv2.threshold(self.map_img, 127, 1, cv2.THRESH_BINARY_INV)
        self.h, self.w = self.binary_map.shape
        self.obstacle_map = self.binary_map 
        
        # 3. Parameters
        self.max_steps = max_steps
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.render_mode = render_mode
        
        self.step_count = 0
        self.episode = 0
        
        # Robot State
        self.initial_robot_x, self.initial_robot_y, self.initial_robot_angle = 50, 65, np.pi/4 # Fixed start
        self.x = self.initial_robot_x
        self.y = self.initial_robot_y 
        self.angle = self.initial_robot_angle
        
        # Goal State (Fixed for all episodes)
        self.initial_goal_x, self.initial_goal_y = 70, 10
        self.goal_x = self.initial_goal_x
        self.goal_y = self.initial_goal_y
        # self._spawn_goal()
        
        # Constants & Config
        self.SPEED = 5.0
        self.TURN_SPEED = 15.0 # Degrees
        self.num_rays = 100
        self.ray_length = 200
        
        # Lidar Angles
        self.lidar_angles = np.linspace(-45, 45, self.num_rays)
        self.last_intersections = []
        
        # Exploration Grid (-1=unexplored, 0=free, 1=obstacle)
        self.exploration_grid = None
        
        # Visualization
        self.screen = None
        self.clock = None

    def _spawn_goal(self):
        """Spawns the goal at a random valid location once."""
        while True:
            gx = np.random.randint(self.goal_radius, self.w - self.goal_radius)
            gy = np.random.randint(self.goal_radius, self.h - self.goal_radius)
            if self._is_clear(gx, gy, self.goal_radius):
                self.goal_x = float(gx)
                self.goal_y = float(gy)
                break

    def _spawn_robot(self):
        """Spawns the robot at a random valid location. But not near the goal."""
        while True:
            rx = np.random.randint(self.robot_radius, self.w - self.robot_radius)
            ry = np.random.randint(self.robot_radius, self.h - self.robot_radius)
            if self._is_clear(rx, ry, self.robot_radius):
                # Check distance to goal
                dist = math.hypot(rx - self.goal_x, ry - self.goal_y)
                if dist > (self.robot_radius + self.goal_radius + 20):
                    self.x = float(rx)
                    self.y = float(ry)
                    break
                
        self.angle = np.random.uniform(0, 360)

    def reset(self):
        """Resets the robot and exploration grid. Goal remains fixed."""
        self.step_count = 0
        self.episode += 1
        
        # Initialize exploration grid
        self.exploration_grid = np.full((self.w, self.h), -1, dtype=int)
        
        # # Spawn Robot (ensure it doesn't spawn too close to the fixed goal)
        # self._spawn_robot()

        self.x, self.y, self.angle = self.initial_robot_x, self.initial_robot_y, self.initial_robot_angle
        
        # Initial Lidar Update
        intersections, _ = self._update_map()
        self.last_intersections = intersections
        return self._get_observation(intersections)

    def step(self, action):
        self.step_count += 1
        prev_x, prev_y = self.x, self.y
        
        # Movement
        if action == 0:   # Forward
            self.x += self.SPEED * np.cos(np.radians(self.angle))
            self.y += self.SPEED * np.sin(np.radians(self.angle))
        elif action == 1: # Backward
            self.x -= self.SPEED * np.cos(np.radians(self.angle))
            self.y -= self.SPEED * np.sin(np.radians(self.angle))
        elif action == 2: # Left
            self.angle = (self.angle - self.TURN_SPEED) % 360
        elif action == 3: # Right
            self.angle = (self.angle + self.TURN_SPEED) % 360
            
        # Collision Check
        if not self._is_clear(self.x, self.y, self.robot_radius):
            self.x, self.y = prev_x, prev_y
            
        # Update Map & Get Lidar
        intersections, new_cells = self._update_map()
        self.last_intersections = intersections
        obs = self._get_observation(intersections)
        
        # Reward & Done
        dist_to_goal = math.hypot(self.x - self.goal_x, self.y - self.goal_y)
        touch_threshold = self.robot_radius + self.goal_radius + 4 # Slight buffer
        
        reward = 0.0
        done = False
        
        if dist_to_goal < touch_threshold:
            reward = 1.0
            done = True # Reset on touch
        elif self.step_count >= self.max_steps:
            done = True
        
        return obs, reward, done, {}

    # --- Original Robot Env Logic (Mapping & Simulation) ---

    def cast_lidar_rays_optimized(self, robot_x, robot_y, orientation):
        """Optimized LIDAR using Bresenham's line algorithm."""
        intersections = []
        angles = orientation + self.lidar_angles
        
        for angle_deg in angles:
            angle_rad = np.radians(angle_deg)
            end_x = int(robot_x + self.ray_length * np.cos(angle_rad))
            end_y = int(robot_y + self.ray_length * np.sin(angle_rad))
            
            line_points = _bresenham_line(int(robot_x), int(robot_y), end_x, end_y)
            
            closest_intersection = None
            for point in line_points:
                x, y = point
                if not (0 <= x < self.w and 0 <= y < self.h):
                    break
                    
                if self.obstacle_map[y, x] == 1:
                    closest_intersection = (x, y)
                    break
            
            intersections.append(closest_intersection)
        
        return intersections

    def _update_map(self):
        """Update exploration grid with latest LIDAR scan."""
        intersections = self.cast_lidar_rays_optimized(self.x, self.y, self.angle)
        angles = self.angle + self.lidar_angles
        new_cells = 0

        for angle_deg, inter in zip(angles, intersections):
            angle_rad = np.radians(angle_deg)

            if inter is not None:
                # Ray hit obstacle
                ox, oy = int(inter[0]), int(inter[1])

                # Free path up to obstacle
                line_points = _bresenham_line(int(self.x), int(self.y), ox, oy)
                for px, py in line_points[:-1]:
                    if (0 <= px < self.w and 0 <= py < self.h and
                        self.exploration_grid[px, py] == -1):
                        self.exploration_grid[px, py] = 0
                        new_cells += 1

                # Mark obstacle cell
                if (0 <= ox < self.w and 0 <= oy < self.h and
                    self.exploration_grid[ox, oy] == -1):
                    self.exploration_grid[ox, oy] = 1
                    new_cells += 1

            else:
                # No obstacle hit -> mark full ray as free
                end_x = int(self.x + self.ray_length * np.cos(angle_rad))
                end_y = int(self.y + self.ray_length * np.sin(angle_rad))
                line_points = _bresenham_line(int(self.x), int(self.y), end_x, end_y)

                for px, py in line_points:
                    if (0 <= px < self.w and 0 <= py < self.h and
                        self.exploration_grid[px, py] == -1):
                        self.exploration_grid[px, py] = 0
                        new_cells += 1

        return intersections, new_cells

    def _get_observation(self, intersections):
        """
        Gets distance readings from x,y intersections
        
        :param self: instance of SimpleRobotEnv class
        :param intersections: (x,y) coordinates of ray intersections or None
        :return: Numpy array of distances for each ray 
        """
        distances = []
        for inter in intersections:
            if inter:
                dist = math.sqrt((inter[0] - self.x)**2 + (inter[1] - self.y)**2)
            else:
                dist = self.ray_length
            distances.append(dist)
        return np.array(distances, dtype=np.float32)

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
            self.screen = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Robot Exploration")
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
        self.screen.blit(base_surf, (0, 0))
        
        # Draw Exploration Grid
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        rgb_ref = pygame.surfarray.pixels3d(overlay)
        alpha_ref = pygame.surfarray.pixels_alpha(overlay)
        
        free_mask = (self.exploration_grid == 0)
        obs_mask = (self.exploration_grid == 1)
        
        rgb_ref[free_mask] = [0, 255, 0]
        alpha_ref[free_mask] = 100
        
        rgb_ref[obs_mask] = [255, 0, 0]
        alpha_ref[obs_mask] = 100
        
        del rgb_ref
        del alpha_ref
        
        self.screen.blit(overlay, (0, 0))

    def _draw_entities(self):
        # Goal (Blue Blob)
        pygame.draw.circle(self.screen, (0, 0, 255), (int(self.goal_x), int(self.goal_y)), self.goal_radius)
        # Robot (Yellow Circle)
        pygame.draw.circle(self.screen, (255, 255, 0), (int(self.x), int(self.y)), self.robot_radius)
        # Heading Line
        end_x = self.x + 15 * np.cos(np.radians(self.angle))
        end_y = self.y + 15 * np.sin(np.radians(self.angle))
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y), (end_x, end_y), 2)

    def _draw_lidar(self):
        # Re-calculate ray ends for visualization if needed, or use intersections
        angles = self.angle + self.lidar_angles
        
        for i, angle_deg in enumerate(angles):
            # Check if we had a hit
            intersection = self.last_intersections[i] if i < len(self.last_intersections) else None
            
            if intersection:
                end_pos = intersection
            else:
                # Calculate max range end point
                angle_rad = np.radians(angle_deg)
                end_x = self.x + self.ray_length * np.cos(angle_rad)
                end_y = self.y + self.ray_length * np.sin(angle_rad)
                end_pos = (end_x, end_y)
            
            # Draw Ray
            pygame.draw.line(self.screen, (0, 255, 255), (self.x, self.y), end_pos, 1)