import numpy as np
import pygame
import os
from datetime import datetime
from math import cos, sin, radians, sqrt
from utils.lidar_utils import line_intersection, circle_line_intersection, cast_lidar_rays
from utils.plotting_utils import get_map_objects, create_robot

class RobotExplorationEnv:
    def __init__(self, map_size=40, grid_width=80, grid_height=40, scale=10, fps=10,
                 robot_size=None, num_rays=100, ray_length=200, max_steps=int(1000e3),
                 wheel_base=4.0, wheel_radius=0.75, dt=0.2, linear_speed=4.0, angular_speed=1.0,
                 save_interval=100, output_dir=None):
        
        # Environment parameters
        self.map_size = map_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.scale = scale
        self.window_width = grid_width * scale
        self.window_height = grid_height * scale
        self.fps = fps
        self.room = get_map_objects(map_size)
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.max_steps = max_steps
        self.save_interval = save_interval
        
        # Robot parameters
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.dt = dt
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.robot_size = robot_size if robot_size else map_size / 10
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        self.WHITE = (255, 255, 255)
        
        # State variables
        self.robot_x = None
        self.robot_y = None
        self.robot_orientation = None
        self.grid = None
        self.current_step = 0
        self.clock = pygame.time.Clock()
        
        # Data recording
        self.path = []
        self.lidar_data = []
        self.action_data = []
        self.output_dir = output_dir
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.output_dir = os.path.join("output", timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Pygame screen
        self.screen = None
        
    def reset(self):
        self.robot_x = 8
        self.robot_y = 8
        self.robot_orientation = 45
        self.grid = np.full((self.grid_width, self.grid_height), -1, dtype=int)
        
        for wall in self.room['walls']:
            x0, y0 = int(wall[0][0]), int(wall[0][1])
            x1, y1 = int(wall[1][0]), int(wall[1][1])
            steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
            for t in np.linspace(0, 1, steps):
                x = int(x0 + t * (x1 - x0))
                y = int(y0 + t * (y1 - y0))
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    self.grid[x, y] = 1
        for circle in self.room['circles']:
            cx, cy = int(circle['center'][0]), int(circle['center'][1])
            radius = int(circle['radius'])
            for x in range(max(0, cx - radius - 1), min(self.grid_width, cx + radius + 2)):
                for y in range(max(0, cy - radius - 1), min(self.grid_height, cy + radius + 2)):
                    if sqrt((x - cx)**2 + (y - cy)**2) <= radius:
                        self.grid[x, y] = 1
        
        self.path = []
        self.lidar_data = []
        self.action_data = []
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action):
        if not 0 <= action <= 3:
            raise ValueError("Action must be 0-3")
        
        actions_map = ['forward', 'back', 'left', 'right']
        action_str = actions_map[action]
        
        if action == 0:  # forward
            v_left = v_right = self.linear_speed
        elif action == 1:  # back
            v_left = v_right = -self.linear_speed
        elif action == 2:  # left
            v_left = -self.linear_speed
            v_right = self.linear_speed
        elif action == 3:  # right
            v_left = self.linear_speed
            v_right = -self.linear_speed
        
        self.robot_x, self.robot_y, self.robot_orientation = self._update_robot_position(
            self.robot_x, self.robot_y, self.robot_orientation, v_left, v_right
        )
        
        intersections, new_cells = self._update_map()
        
        self.path.append((self.robot_x, self.robot_y, self.robot_orientation))
        self.lidar_data.append(intersections)
        self.action_data.append(action_str)
        
        if (self.current_step + 1) % self.save_interval == 0:
            self._save_data(self.current_step + 1)
            self.path = []
            self.lidar_data = []
            self.action_data = []
        
        self.current_step += 1
        
        obs = self._get_observation(intersections)
        reward = new_cells
        done = self.current_step >= self.max_steps or self._get_coverage() >= 99.0
        info = {"new_cells": new_cells, "coverage": self._get_coverage(), "action": action_str}
        
        return obs, reward, done, info
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Robot Exploration")
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        self.screen.fill(self.WHITE)
        self._draw_grid()
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        if self.path:
            self._save_data(self.current_step, final=True)
    
    def _update_robot_position(self, x, y, orientation, v_left, v_right):
        linear_velocity = (v_left + v_right) / 2 * self.wheel_radius
        angular_velocity = (v_right - v_left) / self.wheel_base * self.wheel_radius
        angular_velocity = np.degrees(angular_velocity)
        new_x = x + linear_velocity * cos(radians(orientation)) * self.dt
        new_y = y + linear_velocity * sin(radians(orientation)) * self.dt
        new_orientation = orientation + angular_velocity * self.dt
        new_robot, _ = create_robot(new_x, new_y, new_orientation, size=self.robot_size)
        if self._check_collision(x, y, new_x, new_y, new_robot):
            return x, y, orientation
        return new_x, new_y, new_orientation
    
    def _check_collision(self, current_x, current_y, new_x, new_y, new_robot):
        path_start = (current_x, current_y)
        path_end = (new_x, new_y)
        for wall in self.room['walls']:
            if line_intersection(path_start, path_end, wall[0], wall[1]):
                return True
        for circle in self.room['circles']:
            if len(circle_line_intersection(circle['center'], circle['radius'], path_start, path_end)) > 0:
                return True
        for vertex in new_robot[:-1]:
            for circle in self.room['circles']:
                if sqrt((vertex[0] - circle['center'][0])**2 + (vertex[1] - circle['center'][1])**2) < circle['radius']:
                    return True
            for wall in self.room['walls']:
                p1, p2 = wall[0], wall[1]
                wall_vec = (p2[0] - p1[0], p2[1] - p1[1])
                point_vec = (vertex[0] - p1[0], vertex[1] - p1[1])
                wall_length_sq = wall_vec[0]**2 + wall_vec[1]**2
                if wall_length_sq == 0: continue
                t = max(0, min(1, (point_vec[0] * wall_vec[0] + point_vec[1] * wall_vec[1]) / wall_length_sq))
                closest = (p1[0] + t * wall_vec[0], p1[1] + t * wall_vec[1])
                if sqrt((vertex[0] - closest[0])**2 + (vertex[1] - closest[1])**2) < self.robot_size * 0.1:
                    return True
        return False
    
    def _update_map(self):
        robot, lidar = create_robot(self.robot_x, self.robot_y, self.robot_orientation, size=self.robot_size)
        intersections = cast_lidar_rays(lidar, self.room, self.num_rays, self.ray_length)
        new_cells = 0
        for intersection in intersections:
            if intersection:
                x0, y0 = int(self.robot_x), int(self.robot_y)
                x1, y1 = int(intersection[0]), int(intersection[1])
                steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
                for t in np.linspace(0, 1, steps):
                    x = int(x0 + t * (x1 - x0))
                    y = int(y0 + t * (y1 - y0))
                    if 0 <= x < self.grid_width and 0 <= y < self.grid_height and self.grid[x, y] == -1:
                        self.grid[x, y] = 0
                        new_cells += 1
                if 0 <= x1 < self.grid_width and 0 <= y1 < self.grid_height and self.grid[x1, y1] != 1:
                    self.grid[x1, y1] = 1
                    new_cells += 1
        return intersections, new_cells
    
    def _get_observation(self, intersections=None):
        if intersections is None:
            _, lidar = create_robot(self.robot_x, self.robot_y, self.robot_orientation, size=self.robot_size)
            intersections = cast_lidar_rays(lidar, self.room, self.num_rays, self.ray_length)
        distances = []
        for inter in intersections:
            if inter is not None:
                dx = inter[0] - self.robot_x
                dy = inter[1] - self.robot_y
                dist = sqrt(dx**2 + dy**2)
            else:
                dist = self.ray_length
            distances.append(dist)
        return np.array(distances, dtype=np.float32)
    
    def _get_coverage(self):
        return 100 * np.sum(self.grid >= 0) / (self.grid_width * self.grid_height)
    
    def _save_data(self, step, final=False):
        step_str = f"{step:06d}"
        prefix = "final_" if final else ""
        np.save(os.path.join(self.output_dir, f"path_{prefix}{step_str}.npy"), np.array(self.path, dtype=np.float32))
        np.save(os.path.join(self.output_dir, f"lidar_{prefix}{step_str}.npy"), np.array(self.lidar_data, dtype=object))
        np.save(os.path.join(self.output_dir, f"actions_{prefix}{step_str}.npy"), np.array(self.action_data))
    
    def _draw_grid(self):
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                color = self.GRAY if self.grid[x, y] == -1 else self.GREEN if self.grid[x, y] == 0 else self.BLACK
                pygame.draw.rect(self.screen, color, (x * self.scale, self.window_height - (y + 1) * self.scale, self.scale, self.scale))

