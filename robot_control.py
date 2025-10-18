import pygame
import numpy as np
from math import cos, sin, radians, sqrt
import random
import os
from datetime import datetime
from utils import plotting_utils, lidar_utils
import platform
import asyncio

# Initialize Pygame
pygame.init()
MAP_SIZE = 40
GRID_WIDTH, GRID_HEIGHT = 80, 40
SCALE = 10
WINDOW_WIDTH = GRID_WIDTH * SCALE
WINDOW_HEIGHT = GRID_HEIGHT * SCALE
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Random Robot Exploration")
BLACK, BLUE, GREEN, GRAY = (0, 0, 0), (0, 0, 255), (0, 255, 0), (128, 128, 128)
FPS = 10
clock = pygame.time.Clock()

# Robot parameters
robot_x, robot_y, robot_orientation = 8, 8, 45
wheel_base, wheel_radius, dt = 4.0, 0.75, 0.2
linear_speed, angular_speed = 4.0, 1.0
ROBOT_SIZE = MAP_SIZE / 10
room = plotting_utils.get_map_objects(MAP_SIZE)
num_rays, ray_length = 100, 200

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join("output", timestamp)
os.makedirs(output_dir, exist_ok=True)

# Occupancy grid
grid = np.full((GRID_WIDTH, GRID_HEIGHT), -1, dtype=int)  # -1: unknown, 0: free, 1: occupied

# Initialize grid with known obstacles (no buffer)
for wall in room['walls']:
    x0, y0 = int(wall[0][0]), int(wall[0][1])
    x1, y1 = int(wall[1][0]), int(wall[1][1])
    steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    for t in np.linspace(0, 1, steps):
        x = int(x0 + t * (x1 - x0))
        y = int(y0 + t * (y1 - y0))
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            grid[x, y] = 1
for circle in room['circles']:
    cx, cy = int(circle['center'][0]), int(circle['center'][1])
    radius = int(circle['radius'])
    for x in range(max(0, cx - radius - 1), min(GRID_WIDTH, cx + radius + 2)):
        for y in range(max(0, cy - radius - 1), min(GRID_HEIGHT, cy + radius + 2)):
            if sqrt((x - cx)**2 + (y - cy)**2) <= radius:
                grid[x, y] = 1

def update_robot_position(x, y, orientation, v_left, v_right, dt, wheel_base, wheel_radius, room):
    linear_velocity = (v_left + v_right) / 2 * wheel_radius
    angular_velocity = (v_right - v_left) / wheel_base * wheel_radius
    angular_velocity = np.degrees(angular_velocity)
    new_x = x + linear_velocity * cos(radians(orientation)) * dt
    new_y = y + linear_velocity * sin(radians(orientation)) * dt
    new_orientation = orientation + angular_velocity * dt
    new_robot, _ = plotting_utils.create_robot(new_x, new_y, new_orientation, size=ROBOT_SIZE)
    if check_collision(x, y, new_x, new_y, new_robot, room):
        return x, y, orientation
    return new_x, new_y, new_orientation

def check_collision(current_x, current_y, new_x, new_y, new_robot, room):
    path_start = (current_x, current_y)
    path_end = (new_x, new_y)
    for wall in room['walls']:
        if lidar_utils.line_intersection(path_start, path_end, wall[0], wall[1]):
            return True
    for circle in room['circles']:
        center, radius = circle['center'], circle['radius']
        if lidar_utils.circle_line_intersection(center, radius, path_start, path_end):
            return True
    for vertex in new_robot[:-1]:
        for circle in room['circles']:
            if sqrt((vertex[0] - circle['center'][0])**2 + (vertex[1] - circle['center'][1])**2) < circle['radius']:
                return True
        for wall in room['walls']:
            p1, p2 = wall[0], wall[1]
            wall_vec = (p2[0] - p1[0], p2[1] - p1[1])
            point_vec = (vertex[0] - p1[0], vertex[1] - p1[1])
            wall_length_sq = wall_vec[0]**2 + wall_vec[1]**2
            if wall_length_sq == 0: continue
            t = max(0, min(1, (point_vec[0] * wall_vec[0] + point_vec[1] * wall_vec[1]) / wall_length_sq))
            closest = (p1[0] + t * wall_vec[0], p1[1] + t * wall_vec[1])
            if sqrt((vertex[0] - closest[0])**2 + (vertex[1] - closest[1])**2) < ROBOT_SIZE * 0.1:
                return True
    return False

def draw_grid(screen):
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            color = GRAY if grid[x, y] == -1 else GREEN if grid[x, y] == 0 else BLACK
            pygame.draw.rect(screen, color, (x*SCALE, WINDOW_HEIGHT - (y+1)*SCALE, SCALE, SCALE))

def draw_map(screen, room):
    for wall in room['walls']:
        start = (int(wall[0][0] * SCALE), int(WINDOW_HEIGHT - wall[0][1] * SCALE))
        end = (int(wall[1][0] * SCALE), int(WINDOW_HEIGHT - wall[1][1] * SCALE))
        pygame.draw.line(screen, BLACK, start, end, 2)
    for circle in room['circles']:
        center = (int(circle['center'][0] * SCALE), int(WINDOW_HEIGHT - circle['center'][1] * SCALE))
        radius = int(circle['radius'] * SCALE)
        pygame.draw.circle(screen, BLACK, center, radius, 2)

def draw_robot(screen, robot, lidar):
    robot_points = [(int(x * SCALE), int(WINDOW_HEIGHT - y * SCALE)) for x, y in robot]
    pygame.draw.polygon(screen, BLUE, robot_points, 2)
    lidar_start = (int(lidar[0][0] * SCALE), int(WINDOW_HEIGHT - lidar[0][1] * SCALE))
    lidar_end = (int(lidar[1][0] * SCALE), int(WINDOW_HEIGHT - lidar[1][1] * SCALE))
    pygame.draw.line(screen, BLUE, lidar_start, lidar_end, 3)

def draw_lidar_rays(screen, lidar, intersections):
    lidar_start = (int(lidar[0][0] * SCALE), int(WINDOW_HEIGHT - lidar[0][1] * SCALE))
    for intersection in intersections:
        if intersection:
            end = (int(intersection[0] * SCALE), int(WINDOW_HEIGHT - intersection[1] * SCALE))
            pygame.draw.line(screen, GREEN, lidar_start, end, 1)
            pygame.draw.circle(screen, GREEN, end, 3)

def update_map(robot_x, robot_y, robot_orientation):
    robot, lidar = plotting_utils.create_robot(robot_x, robot_y, robot_orientation, size=ROBOT_SIZE)
    intersections = lidar_utils.cast_lidar_rays(lidar, room, num_rays, ray_length)
    new_cells = 0
    for intersection in intersections:
        if intersection:
            x0, y0 = int(robot_x), int(robot_y)
            x1, y1 = int(intersection[0]), int(intersection[1])
            steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
            for t in np.linspace(0, 1, steps):
                x = int(x0 + t * (x1 - x0))
                y = int(y0 + t * (y1 - y0))
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and grid[x, y] == -1:
                    grid[x, y] = 0
                    new_cells += 1
            if 0 <= x1 < GRID_WIDTH and 0 <= y1 < GRID_HEIGHT and grid[x1, y1] != 1:
                grid[x1, y1] = 1
                new_cells += 1
    return intersections, new_cells

async def main():
    global robot_x, robot_y, robot_orientation
    actions = ['forward', 'back', 'left', 'right']
    steps = 0
    max_steps = int(1000e3)
    path = []
    lidar_data = []
    action_data = []
    interval_size = 100  # Save every 100 steps

    while steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # Choose random action
        action = random.choice(actions)
        if action == 'forward':
            v_left = v_right = linear_speed
        elif action == 'back':
            v_left = v_right = -linear_speed
        elif action == 'left':
            v_left = -linear_speed
            v_right = linear_speed
        elif action == 'right':
            v_left = linear_speed
            v_right = -linear_speed

        # Update robot position
        robot_x, robot_y, robot_orientation = update_robot_position(
            robot_x, robot_y, robot_orientation, v_left, v_right, dt, wheel_base, wheel_radius, room
        )

        # Update map and get lidar measurements
        intersections, new_cells = update_map(robot_x, robot_y, robot_orientation)

        # Record data for current interval
        path.append((robot_x, robot_y, robot_orientation))
        lidar_data.append(intersections)
        action_data.append(action)

        # Save data every 100 steps with zero-padded naming
        if (steps + 1) % interval_size == 0:
            step_str = f"{steps + 1:06d}"  # Zero-pad to 6 digits
            np.save(os.path.join(output_dir, f"path_{step_str}.npy"), np.array(path, dtype=np.float32))
            np.save(os.path.join(output_dir, f"lidar_{step_str}.npy"), np.array(lidar_data, dtype=object))
            np.save(os.path.join(output_dir, f"actions_{step_str}.npy"), np.array(action_data))
            # Reset lists to start collecting data for the next interval
            path = []
            lidar_data = []
            action_data = []

        # Visualize
        screen.fill((255, 255, 255))
        draw_grid(screen)
        robot, lidar = plotting_utils.create_robot(robot_x, robot_y, robot_orientation, size=ROBOT_SIZE)
        # draw_map(screen, room)
        # draw_robot(screen, robot, lidar)
        # draw_lidar_rays(screen, lidar, intersections)
        pygame.display.flip()
        steps += 1
        print(f"Step {steps}, Action: {action}, New cells: {new_cells}, Coverage: {100 * np.sum(grid >= 0) / (GRID_WIDTH * GRID_HEIGHT):.2f}%")
        # await asyncio.sleep(1.0 / FPS)

    # Save any remaining data with zero-padded naming
    if path:  # Only save if there is unsaved data
        step_str = f"{steps:06d}"  # Zero-pad to 6 digits
        np.save(os.path.join(output_dir, f"path_final_{step_str}.npy"), np.array(path, dtype=np.float32))
        np.save(os.path.join(output_dir, f"lidar_final_{step_str}.npy"), np.array(lidar_data, dtype=object))
        np.save(os.path.join(output_dir, f"actions_final_{step_str}.npy"), np.array(action_data))

    coverage = 100 * np.sum(grid >= 0) / (GRID_WIDTH * GRID_HEIGHT)
    print(f"Final Coverage: {coverage:.2f}% after {steps} steps")
    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())