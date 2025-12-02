import os
import numpy as np
from tqdm import tqdm
from blob_env import FixedBlobEnv
import sys

EPISODES = 200
MAX_STEPS_PER_EPISODE = 500

def random_action(env):
    """Sample a random (direction, duration) action."""
    direction = np.random.randint(env.action_dim)
    duration = np.random.randint(env.max_run)
    return direction, duration

def run_random_walk():
    map_path = "environments/images/6.png"
    if not os.path.exists(map_path):
        map_path = "../environments/images/6.png"

    env = FixedBlobEnv(map_path, max_run=15)
    
    episode_lengths = []
    successes = 0

    print("Running random-walk benchmark...")
    for ep in tqdm(range(EPISODES), desc="Episodes"):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = random_action(env)
            _, _, done, info = env.step(action)
            steps += 1

            # Inline step info
            sys.stdout.write(f"\rEpisode {ep+1}/{EPISODES} | Step {steps} | Dist to goal: {info['dist_to_goal']:.2f}")
            sys.stdout.flush()

        episode_lengths.append(steps)
        if done:
            successes += 1

        # Clear line after episode ends
        print()

    # ---- Stats -----------------------------------------------------------
    episode_lengths = np.array(episode_lengths)
    avg_steps = episode_lengths.mean()
    med_steps = np.median(episode_lengths)
    success_rate = successes / EPISODES

    print("\n===== RANDOM WALK STATS =====")
    print(f"Episodes:        {EPISODES}")
    print(f"Successes:       {successes}")
    print(f"Success rate:    {success_rate*100:.1f}%")
    print(f"Avg. steps:      {avg_steps:.2f}")
    print(f"Median steps:    {med_steps:.2f}")
    print(f"Min steps:       {episode_lengths.min()}")
    print(f"Max steps:       {episode_lengths.max()}")
    print("==============================")

    return {
        "steps": episode_lengths,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
    }

if __name__ == "__main__":
    run_random_walk()
