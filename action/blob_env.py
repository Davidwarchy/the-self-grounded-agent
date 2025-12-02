import sys
import os
import numpy as np
import math
import random
import pygame
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_env import RobotExplorationEnv

class FixedBlobEnv(RobotExplorationEnv):
    def __init__(self, map_image_path, max_run=15, **kwargs):
        super().__init__(map_image_path, **kwargs)
        
        self.blob_radius = 12
        self.max_run = max_run  # Used for normalization or info
        
        # Hardcoded Locations
        self.start_x = 50 
        self.start_y = 65
        self.goal_x = 75
        self.goal_y = 0
        
        self.clean_obstacle_map = self.obstacle_map.copy()
        
        self.observation_dim = self.num_rays 
        self.action_dim = 4

    def reset(self):
        self.obstacle_map = self.clean_obstacle_map.copy()
        super().reset()
        
        self.robot_x = self.start_x
        self.robot_y = self.start_y
        self.robot_orientation = 0
        
        cv2.circle(self.obstacle_map, (self.goal_x, self.goal_y), self.blob_radius, 1, -1)
        return self._get_observation()

    def _calculate_reward(self):
        """
        Calculates the immediate reward for the current state.
        This overrides the base environment's method.
        """
        dist = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        threshold = self.robot_radius + self.blob_radius + 4.0
        
        # Time step penalty
        reward = -0.01
        
        # Check Touch
        if dist < threshold:
            reward += 10.0
            
        return reward

    def step(self, action_tuple):
        """
        action_tuple: (direction_index, duration_index)
        """
        direction, duration_idx = action_tuple
        
        # Convert index (0..N) to actual steps (1..N+1)
        steps_to_take = duration_idx + 1
        
        total_reward = 0
        done = False
        dist = 0
        
        for _ in range(steps_to_take):
            # Calling super().step() will internally call self._calculate_reward() 
            # and log the reward, step, action, etc.
            obs, reward, done, _ = super().step(direction)
            
            total_reward += reward
            
            # Check if we reached goal (reward > 0 implies goal hit based on _calculate_reward logic)
            # Or re-check done condition explicitly if needed
            dist = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
            threshold = self.robot_radius + self.blob_radius + 4.0
            if dist < threshold:
                done = True
                break
            
            if done: # If max steps reached
                break
        
        next_obs = obs / self.ray_length
        
        info = {
            "dist_to_goal": dist,
            "steps_taken": steps_to_take
        }
        
        return next_obs, total_reward, done, info

    def render(self):
        super().render()
        if self.screen:
            gx = int(self.goal_x * self.scale)
            gy = int(self.goal_y * self.scale)
            r = int(self.blob_radius * self.scale)
            pygame.draw.circle(self.screen, (0, 0, 0), (gx, gy), r)
            pygame.draw.circle(self.screen, (0, 255, 0), (gx, gy), r, 2)
            pygame.display.flip()