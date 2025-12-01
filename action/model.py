import torch
import torch.nn as nn
from torch.distributions import Categorical

class MultiHeadActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_run_steps=15):
        super().__init__()
        
        # Shared Feature Extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # --- HEAD 1: DIRECTION (Up, Down, Left, Right) ---
        self.actor_dir = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # --- HEAD 2: DURATION (1 step to max_run_steps) ---
        self.actor_time = nn.Sequential(
            nn.Linear(64, max_run_steps),
            nn.Softmax(dim=-1)
        )
        
        # Critic (Value)
        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )
        
    def act(self, state):
        features = self.feature_net(state)
        
        # Get Direction
        probs_dir = self.actor_dir(features)
        dist_dir = Categorical(probs_dir)
        action_dir = dist_dir.sample()
        
        # Get Duration
        probs_time = self.actor_time(features)
        dist_time = Categorical(probs_time)
        action_time = dist_time.sample()
        
        # Combine Log Probabilities (Summing logs = Multiplying probs)
        # We assume direction and duration choices are conditionally independent given state
        logprob = dist_dir.log_prob(action_dir) + dist_time.log_prob(action_time)
        
        return action_dir.item(), action_time.item(), logprob
    
    def evaluate(self, state, action_dir, action_time):
        features = self.feature_net(state)
        
        # Direction Logic
        probs_dir = self.actor_dir(features)
        dist_dir = Categorical(probs_dir)
        logprobs_dir = dist_dir.log_prob(action_dir)
        entropy_dir = dist_dir.entropy()
        
        # Duration Logic
        probs_time = self.actor_time(features)
        dist_time = Categorical(probs_time)
        logprobs_time = dist_time.log_prob(action_time)
        entropy_time = dist_time.entropy()
        
        # Combined metrics
        total_logprobs = logprobs_dir + logprobs_time
        total_entropy = entropy_dir + entropy_time
        
        state_values = self.critic(features)
        
        return total_logprobs, state_values, total_entropy