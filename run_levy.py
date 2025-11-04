import random
import asyncio
import numpy as np
from robot_env import RobotExplorationEnv

# =========================================
# Lévy parameters
# =========================================
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

# =========================================
# Lévy walk loop
# =========================================
async def main():
    env = RobotExplorationEnv(render=False, max_steps=int(1000e3))
    obs = env.reset()
    done = False

    current_run_length = 0
    current_direction = random.randint(0, 3)

    while not done:
        # Start a new run if finished the last one
        if current_run_length <= 0:
            current_direction = random.randint(0, 3)
            current_run_length = sample_truncated_pareto(
                LEVY_ALPHA, LEVY_MIN, LEVY_MAX
            )

        # Step environment
        obs, reward, done, info = env.step(current_direction)
        env.render()

        # Update run counter
        current_run_length -= 1

        # Print step info occasionally
        if env.current_step % 500 == 0:
            print(
                f"Step {env.current_step}, "
                f"Direction: {info['action']}, "
                f"Run steps left: {current_run_length}, "
                f"Coverage: {info['coverage']:.2f}%"
            )

        # Optionally slow down visualization
        # await asyncio.sleep(1.0 / env.fps)

    env.close()
    print(f"Final Coverage: {env._get_coverage():.2f}% after {env.current_step} steps")

if __name__ == "__main__":
    asyncio.run(main())
