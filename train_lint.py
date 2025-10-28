import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from datetime import datetime
import json
from clustering import sample_clusters_and_inspect, plot_clusters_grid
import csv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LidarDataset(Dataset):
    """
    Dataset for Lidar-Goal Navigation.
    Returns: (current_lidar, goal_lidar, action_sequence, temporal_distance)
    """
    def __init__(self, parquet_path, num_rays=100, horizon=5, min_dist=5, max_dist=20):
        self.df = pd.read_parquet(parquet_path)
        self.num_rays = num_rays
        self.horizon = horizon
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.ray_cols = [f'ray_{i}' for i in range(num_rays)]
        # Normalize lidar distances by max ray length (200)
        self.lidar_data = self.df[self.ray_cols].values.astype(np.float32) / 200.0
        self.actions = self.df['action'].values.astype(np.int64)

    def __len__(self):
        # We can't sample goals past max_dist + horizon from the end
        return len(self.df) - self.max_dist - self.horizon

    def __getitem__(self, idx):
        # 1. Sample Temporal Distance (d)
        # d is the time steps needed to reach the goal
        temporal_distance = np.random.randint(self.min_dist, self.max_dist + 1)

        # 2. Define Indices
        anchor_idx = idx
        goal_idx = idx + temporal_distance
        
        # Action sequence starts at anchor_idx and has length 'horizon'
        action_sequence_indices = slice(anchor_idx, anchor_idx + self.horizon)
        
        # 3. Extract Data
        current_lidar = torch.from_numpy(self.lidar_data[anchor_idx])
        goal_lidar = torch.from_numpy(self.lidar_data[goal_idx])
        
        # Actions are 0-3 (up, down, left, right). We use them as discrete class labels.
        action_sequence = torch.from_numpy(self.actions[action_sequence_indices])
        
        # 4. Temporal Distance (d) - Normalized
        # Normalize d for prediction (e.g., to range [0, 1])
        d_norm = torch.tensor([(temporal_distance - self.min_dist) / (self.max_dist - self.min_dist)], dtype=torch.float32)
        
        # We only need the current Lidar for context for this simple model
        return current_lidar, goal_lidar, action_sequence, d_norm, temporal_distance


class LidarGoalModel(nn.Module):
    """
    The Lidar Navigation Transformer (LidarNT) analog.
    It performs Goal Fusion and predicts actions and temporal distance.
    """

    def __init__(self, input_dim=100, embedding_dim=64, num_actions=4, action_horizon=5):
        super().__init__()
        
        # 1. Observation Encoder (psi) - Processes current Lidar
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # 2. Goal Fusion Encoder (phi) - Processes current + goal Lidar
        # Input dimension is 2 * input_dim (concatenated current and goal scan)
        self.goal_fusion_encoder = nn.Sequential(
            nn.Linear(2 * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        # 3. Transformer/MLP Head (f) - Processes combined token
        # Total context is the embedding from the Observation Encoder + the Goal Token (2 * embedding_dim)
        self.transformer_head = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Output Heads
        # Predicts H actions (each is one of 4 classes)
        self.action_head = nn.Linear(64, action_horizon * num_actions)
        # Predicts 1 normalized temporal distance
        self.distance_head = nn.Linear(64, 1)

        self.action_horizon = action_horizon
        self.num_actions = num_actions

    def forward(self, current_lidar, goal_lidar):
        # 1. Encode Context (current_lidar)
        obs_emb = self.obs_encoder(current_lidar) # [B, embedding_dim]

        # 2. Goal Fusion (phi)
        fused_input = torch.cat([current_lidar, goal_lidar], dim=1) # [B, 2*input_dim]
        goal_token = self.goal_fusion_encoder(fused_input) # [B, embedding_dim]

        # 3. Combine and Pass through Head
        combined_token = torch.cat([obs_emb, goal_token], dim=1) # [B, 2*embedding_dim]
        
        features = self.transformer_head(combined_token) # [B, 64]

        # 4. Predict Actions and Distance
        action_logits = self.action_head(features) # [B, H * num_actions]
        # Reshape logits to [B * H, num_actions] for CrossEntropyLoss
        action_logits = action_logits.view(-1, self.num_actions)

        predicted_distance = self.distance_head(features) # [B, 1]

        return action_logits, predicted_distance


def loss_function(action_logits, target_actions, predicted_distance, target_distance, lambd=0.01):
    """Custom loss function combining action and distance predictions (ViNT's L_Vixr)"""
    
    # 1. Action Loss (Cross-Entropy for multi-step discrete actions)
    # target_actions is [B * H]
    action_loss = F.cross_entropy(action_logits, target_actions.view(-1))
    
    # 2. Distance Loss (L1/L2 loss for regression)
    distance_loss = F.mse_loss(predicted_distance, target_distance)
    
    # 3. Combined Loss (ViNT's L_Vixr)
    # Note: ViNT used log p(a) + lambda * log p(d). We use standard CE and MSE.
    total_loss = action_loss + lambd * distance_loss
    return total_loss, action_loss, distance_loss


def train_model(model, dataloader, optimizer, num_epochs=50):
    model.train()
    losses = []
    lambd = 0.01

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_total, running_action, running_dist = 0.0, 0.0, 0.0
        
        for current_lidar, goal_lidar, target_actions, target_distance, _ in dataloader:
            
            # Send data to device
            current_lidar, goal_lidar, target_actions, target_distance = (
                current_lidar.to(device), 
                goal_lidar.to(device), 
                target_actions.to(device), 
                target_distance.to(device)
            )

            optimizer.zero_grad()
            
            # Forward pass
            action_logits, predicted_distance = model(current_lidar, goal_lidar)

            # Compute loss
            total_loss, action_loss, distance_loss = loss_function(
                action_logits, target_actions, predicted_distance, target_distance, lambd=lambd
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Log statistics
            running_total += total_loss.item()
            running_action += action_loss.item()
            running_dist += distance_loss.item()

        avg_loss = running_total / len(dataloader)
        losses.append(avg_loss)
        tqdm.write(f"[epoch {epoch+1}/{num_epochs}] total_loss={avg_loss:.4f} (action={running_action/len(dataloader):.4f}, dist={running_dist/len(dataloader):.4f})")

    return losses

def create_output_directory():
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Created output directory: {output_dir}")
    return output_dir

def load_data(data_dir, n, num_rays=100):
    """
    Returns path to merged_<n>.parquet.
    If it doesn't exist, creates it from the first n CSV logs.
    """
    parquet_path = os.path.join(data_dir, f"merged_{n}.parquet")
    if os.path.exists(parquet_path):
        print(f"[INFO] Found existing parquet: {parquet_path}")
        return parquet_path

    # --- otherwise build from CSVs ---
    log_files = glob.glob(os.path.join(data_dir, "log_*.csv"))
    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    selected_files = log_files[:n]
    print(f"[INFO] Building parquet from {len(selected_files)} CSV logs")

    rows = []
    for i, file_path in enumerate(selected_files):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            if i == 0:
                header = next(reader)
            else:
                next(reader)  # skip header
            rows.extend(list(reader))

    df = pd.DataFrame(rows, columns=header)

    # important: convert ray columns and action to numeric
    ray_cols = [f"ray_{i}" for i in range(num_rays)]
    numeric_cols = ray_cols + ["action"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Saved parquet -> {parquet_path}")
    return parquet_path

def main():
    # Configuration
    config = {
        "data_dir": "output/2025-10-24-074223", # <<< REPLACE with your actual log directory
        "n": 50,  # Number of log files to process (50*100 steps = 5000 steps)
        "num_rays": 100,
        "batch_size": 128,
        "num_epochs": 10, # Increased epochs for better training
        "learning_rate": 0.001,
        "embedding_dim": 64,
        "action_horizon": 5, # Predict 5 future steps (ViNT default)
        "min_goal_dist": 5,  # Min steps to target
        "max_goal_dist": 20, # Max steps to target
        "lambd": 0.01         # Loss balance
    }

    # Create timestamped output directory
    output_dir = create_output_directory()
    start_time = datetime.now()

    # Load data
    parquet_path = load_data(config["data_dir"], config["n"])
    dataset = LidarDataset(
        parquet_path, 
        num_rays=config["num_rays"], 
        horizon=config["action_horizon"],
        min_dist=config["min_goal_dist"],
        max_dist=config["max_goal_dist"]
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    print(f"Dataset size: {len(dataset)} samples")

    # Create model
    model = LidarGoalModel(
        input_dim=config["num_rays"],
        embedding_dim=config["embedding_dim"],
        action_horizon=config["action_horizon"]
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train
    print("\nTraining LidarGoalModel...")
    losses = train_model(model, dataloader, optimizer, num_epochs=config["num_epochs"])

    # Save model
    model_path = os.path.join(output_dir, 'lidar_goal_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save run information
    end_time = datetime.now()
    # (Simplified save_run_info for this context)
    
    print(f"\n{'='*60}")
    print(f"Training Complete. Model saved to: {model_path}")
    print(f"To run the robot, use: goal_navigation.py with this path.")
    print(f"{'='*60}")

if __name__ == "__main__":
    # --- The rest of the original main function for analysis is removed for focus ---
    # The existing helper functions (load_data, create_output_directory) are assumed available.
    # The functions for plotting embeddings (plot_embeddings, analyze_place_recognition) 
    # are removed to keep the focus on the navigation task.
    main()

# The original helper functions from the user's train.py (like load_data) are assumed to remain below.
