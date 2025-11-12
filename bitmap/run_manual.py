#### .\run_manual.py
import pygame
import asyncio
from robot_env import RobotExplorationEnv

async def main():
    map_image_path = r"C:\Users\HP\Desktop\Projects\navigation\9-daniel-cremers-random-motion-collect\environments\images\1.png"
    
    env = RobotExplorationEnv(
        map_image_path=map_image_path,
        grid_width=358,
        grid_height=357,
        robot_radius=10,
        render=True,
        max_steps=int(1000e3)
    )

    obs = env.reset()
    done = False
    print("Manual control: use arrow keys to move. Press ESC to quit.")

    # Key mapping
    key_to_action = {
        pygame.K_UP: 0,     # Forward
        pygame.K_DOWN: 1,   # Backward
        pygame.K_LEFT: 2,   # Turn left
        pygame.K_RIGHT: 3,  # Turn right
    }

    while not done:
        env.render()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    break
                elif event.key in key_to_action:
                    action = key_to_action[event.key]
                    obs, reward, done, info = env.step(action)
                    print(f"Step {env.current_step}, Action: {info['action']}, "
                          f"New cells: {reward}, Coverage: {info['coverage']:.2f}%")

        await asyncio.sleep(1.0 / env.fps)

    env.close()
    print(f"Manual control ended. Final coverage: {env._get_coverage():.2f}% after {env.current_step} steps")

if __name__ == "__main__":
    asyncio.run(main())
