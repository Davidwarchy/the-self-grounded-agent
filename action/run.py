import torch
import pygame
import os
import time
from .blob_env import FixedBlobEnv
from .model import MultiHeadActorCritic

EPISODES = 100
def run():
    map_path = "environments/images/6.png"
    if not os.path.exists(map_path): map_path = "../environments/images/6.png"
    
    MAX_RUN = 15
    env = FixedBlobEnv(map_path, max_run=MAX_RUN, render=True)
    policy = MultiHeadActorCritic(env.observation_dim, env.action_dim, max_run_steps=MAX_RUN)
    
    path = "output/rl_models/multihead_blob_agent.pth"
    if os.path.exists(path):
        policy.load_state_dict(torch.load(path))
        print("Model loaded.")
    
    policy.eval()
    
    # Action map for print debugging
    
    for _ in range(EPISODES):
        state = env.reset()
        done = False
        print(f"New Episode {_+1}")
        
        while not done:
            env.render()
            for e in pygame.event.get():
                if e.type == pygame.QUIT: return

            with torch.no_grad():
                # Get both decisions
                dir_idx, time_idx, _ = policy.act(torch.FloatTensor(state))
            
            steps = time_idx + 1
            
            state, reward, done, _ = env.step((dir_idx, time_idx))
            
        print(f"Goal Reached. Steps taken: {env.current_step}, Reward: {reward:.2f}")

if __name__ == "__main__":
    run()
    # python -m action.run