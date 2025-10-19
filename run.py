import random
import asyncio
from robot_env import RobotExplorationEnv

async def main():
    env = RobotExplorationEnv(render=False, max_steps=int(1000e3))  # Set render=True to see visualization
    obs = env.reset()
    done = False
    while not done:
        action = random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Step {env.current_step}, Action: {info['action']}, New cells: {reward}, Coverage: {info['coverage']:.2f}%")
        # await asyncio.sleep(1.0 / env.fps)
    env.close()
    print(f"Final Coverage: {env._get_coverage():.2f}% after {env.current_step} steps")

if __name__ == "__main__":
    asyncio.run(main())
