import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
from blob_env import FixedBlobEnv
from model import MultiHeadActorCritic

# Parameters
LR = 0.002
GAMMA = 0.99
K_EPOCHS = 4
EPS_CLIP = 0.2
MAX_TIMESTEPS = 5000
UPDATE_TIMESTEP = 1000
MAX_RUN_STEPS = 15 # The agent can choose 1 to 15 steps

class Buffer:
    def __init__(self):
        self.states = []
        self.actions_dir = []  # Store Direction
        self.actions_time = [] # Store Duration
        self.logprobs = []
        self.rewards = []
        self.dones = []
        
    def clear(self):
        self.states = []
        self.actions_dir = []
        self.actions_time = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

def update(policy, buffer, optimizer):
    rewards = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.dones)):
        if is_terminal: discounted_reward = 0
        discounted_reward = reward + (GAMMA * discounted_reward)
        rewards.insert(0, discounted_reward)
        
    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    
    # Stack data
    old_states = torch.stack([torch.FloatTensor(s) for s in buffer.states]).detach()
    old_actions_dir = torch.stack([torch.tensor(a) for a in buffer.actions_dir]).detach()
    old_actions_time = torch.stack([torch.tensor(a) for a in buffer.actions_time]).detach()
    old_logprobs = torch.stack(buffer.logprobs).detach()
    
    mse_loss = nn.MSELoss()
    
    for _ in range(K_EPOCHS):
        # Evaluate using both action components
        logprobs, values, entropy = policy.evaluate(old_states, old_actions_dir, old_actions_time)
        values = values.squeeze()
        
        ratios = torch.exp(logprobs - old_logprobs)
        advantages = rewards - values.detach()
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
        
        loss = -torch.min(surr1, surr2) + 0.5*mse_loss(values, rewards) - 0.01*entropy
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

def train():
    map_path = "environments/images/6.png"
    if not os.path.exists(map_path): map_path = "../environments/images/6.png"
    
    env = FixedBlobEnv(map_path, max_run=MAX_RUN_STEPS)
    
    # Init Multi-Head Policy
    policy = MultiHeadActorCritic(env.observation_dim, env.action_dim, max_run_steps=MAX_RUN_STEPS)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    buffer = Buffer()
    
    print(f"Training Multi-Head Agent (Dir + Duration)...")
    
    timestep = 0
    scores = []
    
    while timestep < MAX_TIMESTEPS:
        state = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            # Act returns direction, duration, and combined logprob
            dir_idx, time_idx, logprob = policy.act(torch.FloatTensor(state))
            
            # Step environment with tuple
            next_state, reward, done, _ = env.step((dir_idx, time_idx))
            
            buffer.states.append(state)
            buffer.actions_dir.append(dir_idx)   # Save Separately
            buffer.actions_time.append(time_idx) # Save Separately
            buffer.logprobs.append(logprob)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            
            state = next_state
            ep_reward += reward
            timestep += 1
            
            if timestep % UPDATE_TIMESTEP == 0:
                update(policy, buffer, optimizer)
                buffer.clear()
                print(f"Update at step {timestep}")

        scores.append(ep_reward)
        if len(scores) % 10 == 0:
            print(f"Ep {len(scores)} | Score: {ep_reward:.2f}")

    os.makedirs("output/rl_models", exist_ok=True)
    torch.save(policy.state_dict(), "output/rl_models/multihead_blob_agent.pth")
    
    plt.plot(scores)
    plt.title("Multi-Head Policy Training")
    plt.savefig("output/rl_models/multihead_training.png")
    print("Done!")

if __name__ == "__main__":
    train()