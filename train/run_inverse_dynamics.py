# inverse_dynamics.py 
"""
Predict action given state_t and state_t+1 using a simple neural network.
Author: David Warutumo 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import datetime

# --- PATH FIX: Allow running this script directly ---
# This adds the project root (one level up) to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
    VAL_SPLIT = 0.2  # 20% for validation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")

    # --- Data Loading & Splitting ---
    try:
        parquet_path = load_or_create_parquet(DATA_DIR, N_FILES)
        full_dataset = ActionDataset(parquet_path)
        
        # Calculate split sizes
        val_size = int(len(full_dataset) * VAL_SPLIT)
        train_size = len(full_dataset) - val_size
        
        # Random split
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"[INFO] Dataset size: {len(full_dataset)}")
        print(f"[INFO] Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        print(f"Ensure you are running from the project root or that '{DATA_DIR}' exists.")
        return

    # --- Model Setup ---
    model = ActionPredictor(num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss() 

    # Output directory setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_dir = f"output/inverse_dynamics_model/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "inverse_model_best.pth")

    best_val_acc = 0.0

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        # 1. TRAIN
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for s_t, s_t1, target_action in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            s_t, s_t1, target_action = s_t.to(device), s_t1.to(device), target_action.to(device)
            
            # Forward
            logits = model(s_t, s_t1)
            
            # Loss
            loss = criterion(logits, target_action)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += target_action.size(0)
            train_correct += (predicted == target_action).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # 2. VALIDATE
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for s_t, s_t1, target_action in val_loader:
                s_t, s_t1, target_action = s_t.to(device), s_t1.to(device), target_action.to(device)
                
                logits = model(s_t, s_t1)
                loss = criterion(logits, target_action)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += target_action.size(0)
                val_correct += (predicted == target_action).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # 3. LOG & SAVE
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    >>> New Best Validation Accuracy! Saved to {best_model_path}")

    print(f"\n[INFO] Training Complete. Best Validation Accuracy: {best_val_acc:.2f}%")

    # --- Simple Test on Validation Set ---
    print("\n[TESTING] Checking predictions on Validation Set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
    
    with torch.no_grad():
        # Create an iterator to grab a batch
        val_iter = iter(val_loader)
        s_t, s_t1, real_a = next(val_iter)
        
        # Take first 5 samples from the batch
        s_t, s_t1, real_a = s_t[:5].to(device), s_t1[:5].to(device), real_a[:5].to(device)
        
        logits = model(s_t, s_t1)
        predictions = torch.argmax(logits, dim=1)
        
        for i in range(5):
            real_label = action_map.get(real_a[i].item(), str(real_a[i].item()))
            pred_label = action_map.get(predictions[i].item(), str(predictions[i].item()))
            is_correct = 'CORRECT' if real_a[i] == predictions[i] else 'WRONG'
            print(f"Sample {i+1} | Real: {real_label:<5} | Pred: {pred_label:<5} | {is_correct}")

if __name__ == "__main__":
    train_inverse_model()