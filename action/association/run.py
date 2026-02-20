import os
import time
import random
import datetime
import numpy as np
import pandas as pd
from robot_env import DownsampledRobotEnv

# --- Configuration ---
MAP_PATH = "../environments/images/12.png"  # Using environment 12.png as requested
TOTAL_STEPS = 100_000       # Total frames to record across all episodes
MAX_EPISODE_STEPS = 10_000     # Reset env after this many steps

def setup_logging():
    """Creates the timestamped output directory and returns the path."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    base_dir = os.path.join("output", "downsampled", timestamp)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def run_collection():
    # 1. Setup
    out_dir = setup_logging()
    print(f"Starting downsampled collection. Output: {out_dir}")
    
    # Check map existence, fallback if needed
    if not os.path.exists(MAP_PATH):
        print(f"Error: Map not found at {MAP_PATH}")
        print("Please ensure 'environments/images/12.png' exists or update MAP_PATH.")
        return

    render_sim = False  # Set to True to visualize during collection
    env = DownsampledRobotEnv(MAP_PATH, max_steps=MAX_EPISODE_STEPS, render_mode=render_sim)
    
    # 2. Header Definition
    # We have 3 discretized ray values (0, 1, or 2)
    lidar_cols = [f"ray_{i}" for i in range(3)]  # Only 3 rays after downsampling
    header = [
        "step", "strategy", "action", 
        *lidar_cols, 
        "episode", "reward", "x", "y", "orientation"
    ]
    
    buffer = []
    global_step = 0
    file_index = 0
    buffer_size = 100  # Save to CSV every 100 steps
    
    # --- Metrics State ---
    episodes_completed = 0
    success_count = 0
    success_lengths = []

    # 3. Main Loop
    obs = env.reset()
    
    while global_step < TOTAL_STEPS:
        
        # --- Strategy: Random Uniform Motion with 5 actions ---
        # 0=up, 1=down, 2=left, 3=right, 4=nothing
        current_action = random.randint(0, 4)
        
        # --- Execution ---
        obs, reward, done, _ = env.step(current_action)
        
        if render_sim:
            env.render()
        
        # --- Data Recording ---
        row = [
            global_step,                # step
            "random_uniform_5actions",  # strategy
            current_action,             # action (0-4)
            int(obs[0]),                # ray_0 (discretized)
            int(obs[1]),                # ray_1 (discretized)
            int(obs[2]),                # ray_2 (discretized)
            env.episode,                # episode
            reward,                     # reward (1.0 if goal reached, else 0)
            round(env.x, 2),            # x
            round(env.y, 2),            # y
            round(env.angle, 2)         # orientation
        ]
        
        buffer.append(row)
        
        # --- Counters Update ---
        global_step += 1
        
        # --- Flushing to Disk ---
        if len(buffer) >= buffer_size:
            file_path = os.path.join(out_dir, f"log_{global_step:06d}.csv")
            df = pd.DataFrame(buffer, columns=header)
            df.to_csv(file_path, index=False)
            
            buffer = []
            file_index += 1
            
        # --- Episode Reset & Metrics Calculation ---
        if done:
            episodes_completed += 1
            
            # Check for Success (reward is 1.0 ONLY if goal is reached)
            is_success = (reward > 0.0)
            
            if is_success:
                success_count += 1
                success_lengths.append(env.step_count)

            # Calculate Stats
            success_rate = (success_count / episodes_completed) * 100 if episodes_completed > 0 else 0
            avg_success_len = np.mean(success_lengths) if success_lengths else 0.0

            # Console Output
            print(f"Episode {env.episode} finished after {env.step_count} steps. "
                  f"Success: {'Yes' if is_success else 'No'}, "
                  f"Success Rate: {success_rate:.2f}%, "
                  f"Avg Success Path Length: {avg_success_len:.1f}")

            obs = env.reset()

    # 4. Final Flush
    if buffer:
        file_path = os.path.join(out_dir, f"log_{global_step:06d}.csv")
        df = pd.DataFrame(buffer, columns=header)
        df.to_csv(file_path, index=False)
        print(f"Saved final {len(buffer)} steps to {file_path}")

    # 5. Summary Statistics
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Total steps collected: {global_step}")
    print(f"Total episodes: {episodes_completed}")
    print(f"Success rate: {success_count}/{episodes_completed} = {success_rate:.2f}%")
    if success_lengths:
        print(f"Average success path length: {np.mean(success_lengths):.1f} steps")
        print(f"Min success path length: {np.min(success_lengths):.0f} steps")
        print(f"Max success path length: {np.max(success_lengths):.0f} steps")
    print(f"Data saved to: {out_dir}")
    print("="*60)

if __name__ == "__main__":
    run_collection()