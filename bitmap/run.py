import random
import asyncio
from robot_env import RobotExplorationEnv

async def main():
    map_image_path = r"C:\Users\HP\Desktop\Projects\navigation\9-daniel-cremers-random-motion-collect\environments\images\1.png"
    
    env = RobotExplorationEnv(
        map_image_path=map_image_path,
        grid_width=358,
        grid_height=357,
        robot_radius=10,
        render=True,  # Set to True to see visualization
        max_steps=int(1000e3)
    )
    
    obs = env.reset()
    done = False
    
    while not done:
        action = random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        env.render()
        
        print(f"Step {env.current_step}, Action: {info['action']}, New cells: {reward}, Coverage: {info['coverage']:.2f}%")
        
        # await asyncio.sleep(1.0 / env.fps)  # Uncomment to slow down
    
    env.close()
    print(f"Final Coverage: {env._get_coverage():.2f}% after {env.current_step} steps")

if __name__ == "__main__":
    asyncio.run(main())