import os
import time
import csv
import random
import datetime
import pandas as pd
from robot_env import SimpleRobotEnv

# --- Configuration ---
MAP_PATH = "environments/images/6.png" # Adjust if your map is elsewhere
TOTAL_STEPS = 500000        # Total frames to record across all episodes
MAX_EPISODE_STEPS = 1000   # Reset env after this many steps
MIN_RUN_LENGTH = 1         # Min steps to hold an action
MAX_RUN_LENGTH = 30        # Max steps to hold an action

def setup_logging():
    """Creates the timestamped output directory and returns the path."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    base_dir = os.path.join("output", "action", timestamp)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def run_collection():
    # 1. Setup
    out_dir = setup_logging()
    print(f"Starting collection. Output: {out_dir}")
    
    # Check map existence, fallback if needed
    if not os.path.exists(MAP_PATH):
        print(f"Error: Map not found at {MAP_PATH}")
        print("Please ensure 'environments/images/6.png' exists or update MAP_PATH.")
        return

    env = SimpleRobotEnv(MAP_PATH, max_steps=MAX_EPISODE_STEPS)

    render = False  # Set to True to visualize during collection
    
    # 2. Header Definition
    # step, strategy, action, run_start, run_length, ray_0...ray_99, episode, reward, x, y, orientation
    lidar_cols = [f"ray_{i}" for i in range(100)]
    header = [
        "step", "strategy", "action", 
        "run_start", "run_length", 
        *lidar_cols, 
        "episode", "reward", "x", "y", "orientation"
    ]
    
    buffer = []
    global_step = 0
    file_index = 0
    buffer_size = 100  # Save to CSV every 1k steps
    
    # Strategy State
    current_action = 0
    steps_remaining_in_run = 0
    is_run_start = False
    run_total_length = 0

    # 3. Main Loop
    obs = env.reset()
    
    while global_step < TOTAL_STEPS:
        
        # --- Strategy: Random Uniform Motion ---
        # If the previous run is finished, pick a new action and duration
        if steps_remaining_in_run <= 0:
            current_action = random.randint(0, 3)
            run_total_length = random.randint(MIN_RUN_LENGTH, MAX_RUN_LENGTH)
            steps_remaining_in_run = run_total_length
            is_run_start = True
        else:
            is_run_start = False
        
        # --- Execution ---
        obs, reward, done, _ = env.step(current_action)
        if render:
            env.render() # Optional visualization
        
        # --- Data Recording ---
        
        # Format run metadata (only strictly 1 or int on start, else blank)
        # Note: CSV standard usually prefers actual NaNs or consistent types, 
        # but requested format is specific: "1 or blank"
        log_run_start = 1 if is_run_start else ""
        log_run_length = run_total_length if is_run_start else ""
        
        row = [
            global_step,                # step
            "random_uniform",           # strategy
            current_action,             # action
            log_run_start,              # run_start
            log_run_length,             # run_length
            *obs,                       # ray_0 ... ray_99
            env.episode,                # episode
            reward,                     # reward
            round(env.x, 2),            # x
            round(env.y, 2),            # y
            round(env.angle, 2)         # orientation
        ]
        
        buffer.append(row)
        
        # --- Counters Update ---
        global_step += 1
        steps_remaining_in_run -= 1
        
        # --- Flushing to Disk ---
        if len(buffer) >= buffer_size:
            file_path = os.path.join(out_dir, f"log_{file_index}.csv")
            df = pd.DataFrame(buffer, columns=header)
            df.to_csv(file_path, index=False)
            
            buffer = []
            file_index += 1
            
        # --- Episode Reset ---
        if done:
            print(f"Episode {env.episode} finished after {env.step_count} steps.")
            obs = env.reset()
            # We do NOT reset global_step, as that tracks total dataset size
            # We strictly might reset the run logic if we want a fresh run per episode,
            # but usually preserving flow is fine. Let's reset run logic for cleanliness.
            steps_remaining_in_run = 0 

    # 4. Final Flush
    if buffer:
        file_path = os.path.join(out_dir, f"log_{file_index}.csv")
        df = pd.DataFrame(buffer, columns=header)
        df.to_csv(file_path, index=False)
        print(f"Saved final {len(buffer)} steps to {file_path}")

    print("Collection Complete.")

if __name__ == "__main__":
    run_collection()