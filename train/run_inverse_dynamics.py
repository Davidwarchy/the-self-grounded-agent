# inverse_dynamics.py 
"""
Predict action given state_t and state_t+1 using a simple neural network.
Author: David Warutumo 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import datetime

# Reuse your existing parquet loader
from train.dataset import load_or_create_parquet

# ==========================================
# 1. The Super Simple Model (Classification)
# ==========================================
class ActionPredictor(nn.Module):
    def __init__(self, input_dim=100, num_classes=4):
        super().__init__()
        # Input is state_t (100) + state_t+1 (100) = 200
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output: logits for 4 actions (0=Up, 1=Down, 2=Left, 3=Right)
        )

    def forward(self, state_t, state_t1):
        # Concatenate s_t and s_{t+1}
        x = torch.cat([state_t, state_t1], dim=1)
        return self.net(x)

# ==========================================
# 2. The Dataset (Discrete Action 0-3)
# ==========================================
class ActionDataset(Dataset):
    def __init__(self, parquet_path, num_rays=100):
        self.df = pd.read_parquet(parquet_path)
        
        # 1. Load LiDAR (State)
        ray_cols = [f'ray_{i}' for i in range(num_rays)]
        self.lidar = (self.df[ray_cols].values.astype(np.float32) / 200.0)
        
        # 2. Load Actions (Integers 0-3)
        if 'action' in self.df.columns:
            self.actions = self.df['action'].values.astype(np.int64)
        else:
            print("[WARN] 'action' column not found! Using zeros.")
            self.actions = np.zeros(len(self.df), dtype=np.int64)

    def __len__(self):
        # We need t and t+1, so length is N-1
        return len(self.df) - 1

    def __getitem__(self, idx):
        # We want to predict the action that caused the transition s_t -> s_{t+1}.
        # In the logs: row 'i' contains state 'i' and the action 'i' that caused it.
        # So, transition (lidar[idx] -> lidar[idx+1]) is caused by actions[idx+1].
        
        state_t = torch.from_numpy(self.lidar[idx])
        state_t1 = torch.from_numpy(self.lidar[idx + 1])
        
        # Use idx + 1 for the action target
        action_t = torch.tensor(self.actions[idx + 1], dtype=torch.long)
        
        return state_t, state_t1, action_t

# ==========================================
# 3. Training Loop
# ==========================================
def train_inverse_model():
    # --- Config ---
    DATA_DIR = "output/2025-11-19-081615_uniform_100k_env8"
    # Fallback to generic output if specific folder doesn't exist
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "output"
        
    N_FILES = 1000
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")

    # --- Data ---
    parquet_path = load_or_create_parquet(DATA_DIR, N_FILES)
    dataset = ActionDataset(parquet_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model ---
    model = ActionPredictor(num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Changed to CrossEntropyLoss because we are predicting classes 0-3
    criterion = nn.CrossEntropyLoss() 

    # --- Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for s_t, s_t1, target_action in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            s_t, s_t1, target_action = s_t.to(device), s_t1.to(device), target_action.to(device)
            
            # Forward (outputs logits)
            logits = model(s_t, s_t1)
            
            # Loss
            loss = criterion(logits, target_action)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += target_action.size(0)
            correct += (predicted == target_action).sum().item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | Acc: {accuracy:.2f}%")

    # --- Save ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")  # YYYY-MM-DD-HHMMSS 
    output_dir = f"output/inverse_dynamics_model/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "inverse_model.pth"))
    print(f"[INFO] Model saved to {os.path.join(output_dir, 'inverse_model.pth')}")

    # --- Simple Test ---
    print("\n[TESTING] Checking a few predictions...")
    model.eval()
    action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
    
    with torch.no_grad():
        # Test on 5 random samples
        indices = np.random.choice(len(dataset), 5)
        for idx in indices:
            s_t, s_t1, real_a = dataset[idx]
            s_t, s_t1 = s_t.unsqueeze(0).to(device), s_t1.unsqueeze(0).to(device)
            
            logits = model(s_t, s_t1)
            pred_a = torch.argmax(logits, dim=1).item()
            
            real_label = action_map.get(real_a.item(), str(real_a.item()))
            pred_label = action_map.get(pred_a, str(pred_a))
            
            print(f"Idx {idx} | Real: {real_label:<5} | Pred: {pred_label:<5} | {'CORRECT' if real_a.item() == pred_a else 'WRONG'}")

if __name__ == "__main__":
    train_inverse_model()
    # python -m train.run_inverse_dynamics