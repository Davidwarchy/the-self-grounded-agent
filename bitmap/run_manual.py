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
    print("Manual control: hold arrow keys to move. Press ESC to quit.")

    while not done:
        env.render()

        # Handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        # Continuous key state check
        keys = pygame.key.get_pressed()
        action = None

        # Movement mapping
        if keys[pygame.K_UP]:
            action = 0  # forward
        elif keys[pygame.K_DOWN]:
            action = 1  # backward
        elif keys[pygame.K_LEFT]:
            action = 2  # turn left
        elif keys[pygame.K_RIGHT]:
            action = 3  # turn right

        # Only step when a key is pressed
        if action is not None:
            obs, reward, done, info = env.step(action)
            print(f"Step {env.current_step}, Action: {info['action']}, "
                  f"New cells: {reward}, Coverage: {info['coverage']:.2f}%")

        await asyncio.sleep(1.0 / env.fps)

    env.close()
    print(f"Manual control ended. Final coverage: {env._get_coverage():.2f}% after {env.current_step} steps")

if __name__ == "__main__":
    asyncio.run(main())
