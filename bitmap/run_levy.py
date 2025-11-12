import random
import asyncio
import numpy as np
from robot_env import RobotExplorationEnv

# Lévy parameters
LEVY_ALPHA = 1.6
LEVY_MIN = 1.0
LEVY_MAX = 200.0

def sample_truncated_pareto(alpha, xm, xmax):
    """Sample from a truncated Pareto distribution."""
    u = np.random.uniform()
    x = xm * (1 - u)**(-1.0 / alpha)
    while x > xmax:
        u = np.random.uniform()
        x = xm * (1 - u)**(-1.0 / alpha)
    return int(max(1, round(x)))

async def main():
    map_image_path = r"C:\Users\HP\Desktop\Projects\navigation\9-daniel-cremers-random-motion-collect\environments\images\1.png"
    
    env = RobotExplorationEnv(
        map_image_path=map_image_path,
        grid_width=358,
        grid_height=358,
        robot_radius=10,
        render=True,
        max_steps=int(1000e3)
    )
    
    obs = env.reset()
    done = False

    current_run_length = 0
    current_direction = random.randint(0, 3)

    while not done:
        # Start a new run if finished the last one
        if current_run_length <= 0:
            current_direction = random.randint(0, 3)
            current_run_length = sample_truncated_pareto(LEVY_ALPHA, LEVY_MIN, LEVY_MAX)

        # Step environment
        obs, reward, done, info = env.step(current_direction)
        env.render()

        # Update run counter
        current_run_length -= 1

        # Print step info occasionally
        if env.current_step % 1 == 0:
            print(
                f"Step {env.current_step}, "
                f"Direction: {info['action']}, "
                f"Run steps left: {current_run_length}, "
                f"Coverage: {info['coverage']:.2f}%"
            )

        # await asyncio.sleep(1.0 / env.fps)  # Uncomment to slow down

    env.close()
    print(f"Final Coverage: {env._get_coverage():.2f}% after {env.current_step} steps")

if __name__ == "__main__":
    asyncio.run(main())