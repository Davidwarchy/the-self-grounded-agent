"""
Forward Dynamics Model: Predict LiDAR at t+1 from LiDAR at t
With optional action and history context
Author: David Warutumo (with Claude assistance)
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
import matplotlib.pyplot as plt
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train.dataset import load_or_create_parquet

# ==========================================
# 1. Forward Dynamics Models
# ==========================================

class ForwardDynamicsBase(nn.Module):
    """Base model: LiDAR(t) -> LiDAR(t+1)"""
    def __init__(self, input_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, lidar_t):
        return self.net(lidar_t)


class ForwardDynamicsWithAction(nn.Module):
    """Model with action: [LiDAR(t), action] -> LiDAR(t+1)"""
    def __init__(self, input_dim=100, num_actions=4):
        super().__init__()
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 16)
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, lidar_t, action):
        action_emb = self.action_embed(action)
        x = torch.cat([lidar_t, action_emb], dim=1)
        return self.net(x)


class ForwardDynamicsWithHistory(nn.Module):
    """Model with history: LiDAR(t-n:t) -> LiDAR(t+1)"""
    def __init__(self, input_dim=100, history_len=3):
        super().__init__()
        self.history_len = history_len
        
        # Process history with 1D convolutions
        self.conv = nn.Sequential(
            nn.Conv1d(history_len, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(128 + input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, lidar_history):
        # lidar_history: (batch, history_len, input_dim)
        batch_size = lidar_history.shape[0]
        
        # Conv processing
        conv_out = self.conv(lidar_history)  # (batch, 128, 1)
        conv_out = conv_out.squeeze(-1)  # (batch, 128)
        
        # Concat with current frame
        current = lidar_history[:, -1, :]  # (batch, input_dim)
        x = torch.cat([conv_out, current], dim=1)
        
        return self.fc(x)


class ForwardDynamicsComplete(nn.Module):
    """Complete model: [LiDAR(t-n:t), action] -> LiDAR(t+1)"""
    def __init__(self, input_dim=100, history_len=3, num_actions=4):
        super().__init__()
        self.history_len = history_len
        
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 16)
        
        # Process history
        self.conv = nn.Sequential(
            nn.Conv1d(history_len, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(128 + input_dim + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, lidar_history, action):
        # Conv processing
        conv_out = self.conv(lidar_history).squeeze(-1)
        
        # Current frame + action
        current = lidar_history[:, -1, :]
        action_emb = self.action_embed(action)
        
        x = torch.cat([conv_out, current, action_emb], dim=1)
        return self.fc(x)


# ==========================================
# 2. Datasets
# ==========================================

class ForwardDynamicsDataset(Dataset):
    """Base dataset: predict lidar(t+1) from lidar(t)"""
    def __init__(self, parquet_path, num_rays=100):
        self.df = pd.read_parquet(parquet_path)
        ray_cols = [f'ray_{i}' for i in range(num_rays)]
        self.lidar = (self.df[ray_cols].values.astype(np.float32) / 200.0)
    
    def __len__(self):
        return len(self.df) - 1
    
    def __getitem__(self, idx):
        lidar_t = torch.from_numpy(self.lidar[idx])
        lidar_t1 = torch.from_numpy(self.lidar[idx + 1])
        return lidar_t, lidar_t1


class ForwardDynamicsWithActionDataset(Dataset):
    """Dataset with action conditioning"""
    def __init__(self, parquet_path, num_rays=100):
        self.df = pd.read_parquet(parquet_path)
        ray_cols = [f'ray_{i}' for i in range(num_rays)]
        self.lidar = (self.df[ray_cols].values.astype(np.float32) / 200.0)
        
        if 'action' in self.df.columns:
            self.actions = self.df['action'].values.astype(np.int64)
        else:
            print("[WARN] No action column, using zeros")
            self.actions = np.zeros(len(self.df), dtype=np.int64)
    
    def __len__(self):
        return len(self.df) - 1
    
    def __getitem__(self, idx):
        lidar_t = torch.from_numpy(self.lidar[idx])
        action = torch.tensor(self.actions[idx + 1], dtype=torch.long)
        lidar_t1 = torch.from_numpy(self.lidar[idx + 1])
        return lidar_t, action, lidar_t1


class ForwardDynamicsWithHistoryDataset(Dataset):
    """Dataset with history context"""
    def __init__(self, parquet_path, num_rays=100, history_len=3):
        self.df = pd.read_parquet(parquet_path)
        ray_cols = [f'ray_{i}' for i in range(num_rays)]
        self.lidar = (self.df[ray_cols].values.astype(np.float32) / 200.0)
        self.history_len = history_len
    
    def __len__(self):
        return len(self.df) - self.history_len
    
    def __getitem__(self, idx):
        history = torch.from_numpy(self.lidar[idx:idx + self.history_len])
        target = torch.from_numpy(self.lidar[idx + self.history_len])
        return history, target


class ForwardDynamicsCompleteDataset(Dataset):
    """Dataset with both history and action"""
    def __init__(self, parquet_path, num_rays=100, history_len=3):
        self.df = pd.read_parquet(parquet_path)
        ray_cols = [f'ray_{i}' for i in range(num_rays)]
        self.lidar = (self.df[ray_cols].values.astype(np.float32) / 200.0)
        self.history_len = history_len
        
        if 'action' in self.df.columns:
            self.actions = self.df['action'].values.astype(np.int64)
        else:
            print("[WARN] No action column, using zeros")
            self.actions = np.zeros(len(self.df), dtype=np.int64)
    
    def __len__(self):
        return len(self.df) - self.history_len
    
    def __getitem__(self, idx):
        history = torch.from_numpy(self.lidar[idx:idx + self.history_len])
        action = torch.tensor(self.actions[idx + self.history_len], dtype=torch.long)
        target = torch.from_numpy(self.lidar[idx + self.history_len])
        return history, action, target


# ==========================================
# 3. Visualization Functions
# ==========================================

def plot_prediction_comparison(model, dataset, device, save_path, n_samples=5):
    """Plot predicted vs actual LiDAR scans"""
    model.eval()
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i in range(n_samples):
            idx = np.random.randint(len(dataset))
            
            # Get prediction based on dataset type
            if isinstance(dataset, ForwardDynamicsCompleteDataset):
                history, action, target = dataset[idx]
                history = history.unsqueeze(0).to(device)
                action = action.unsqueeze(0).to(device)
                pred = model(history, action).cpu().squeeze()
            elif isinstance(dataset, ForwardDynamicsWithHistoryDataset):
                history, target = dataset[idx]
                history = history.unsqueeze(0).to(device)
                pred = model(history).cpu().squeeze()
            elif isinstance(dataset, ForwardDynamicsWithActionDataset):
                lidar_t, action, target = dataset[idx]
                lidar_t = lidar_t.unsqueeze(0).to(device)
                action = action.unsqueeze(0).to(device)
                pred = model(lidar_t, action).cpu().squeeze()
            else:  # Base dataset
                lidar_t, target = dataset[idx]
                lidar_t = lidar_t.unsqueeze(0).to(device)
                pred = model(lidar_t).cpu().squeeze()
            
            target = target.numpy()
            pred = pred.numpy()
            
            # Calculate error
            mse = np.mean((target - pred) ** 2)
            
            # Plot
            ax = axes[i]
            ax.plot(target, label='Actual', linewidth=2, alpha=0.8)
            ax.plot(pred, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
            ax.fill_between(range(len(target)), target, pred, alpha=0.2)
            ax.set_xlabel('Ray Index')
            ax.set_ylabel('Distance (normalized)')
            ax.set_title(f'Sample {i+1} | MSE: {mse:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved prediction comparison to {save_path}")


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    
    # Mark best validation loss
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.scatter([best_epoch], [best_val], color='red', s=100, zorder=5)
    ax.text(best_epoch, best_val, f'  Best: {best_val:.4f}', 
            verticalalignment='bottom')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved training curves to {save_path}")


# ==========================================
# 4. Training Function
# ==========================================

def train_forward_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")
    
    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_type = config['model_type']
    output_dir = f"{config['data_dir']}/forward_dynamics/{model_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        parquet_path = load_or_create_parquet(config['data_dir'], config['n_files'])
        
        # Create appropriate dataset
        if model_type == 'base':
            full_dataset = ForwardDynamicsDataset(parquet_path)
            model = ForwardDynamicsBase().to(device)
        elif model_type == 'action':
            full_dataset = ForwardDynamicsWithActionDataset(parquet_path)
            model = ForwardDynamicsWithAction().to(device)
        elif model_type == 'history':
            full_dataset = ForwardDynamicsWithHistoryDataset(
                parquet_path, history_len=config['history_len']
            )
            model = ForwardDynamicsWithHistory(
                history_len=config['history_len']
            ).to(device)
        elif model_type == 'complete':
            full_dataset = ForwardDynamicsCompleteDataset(
                parquet_path, history_len=config['history_len']
            )
            model = ForwardDynamicsComplete(
                history_len=config['history_len']
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"[INFO] Dataset size: {len(full_dataset)}")
        
    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        return
    
    # Split dataset
    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=4
    )
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_model_path = os.path.join(output_dir, "best_model.pth")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
            # Handle different input formats
            if model_type == 'base':
                lidar_t, target = [x.to(device) for x in batch]
                pred = model(lidar_t)
            elif model_type == 'action':
                lidar_t, action, target = [x.to(device) for x in batch]
                pred = model(lidar_t, action)
            elif model_type == 'history':
                history, target = [x.to(device) for x in batch]
                pred = model(history)
            else:  # complete
                history, action, target = [x.to(device) for x in batch]
                pred = model(history, action)
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'base':
                    lidar_t, target = [x.to(device) for x in batch]
                    pred = model(lidar_t)
                elif model_type == 'action':
                    lidar_t, action, target = [x.to(device) for x in batch]
                    pred = model(lidar_t, action)
                elif model_type == 'history':
                    history, target = [x.to(device) for x in batch]
                    pred = model(history)
                else:
                    history, action, target = [x.to(device) for x in batch]
                    pred = model(history, action)
                
                loss = criterion(pred, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, best_model_path)
            print(f"    >>> New Best! Saved to {best_model_path}")
    
    # Load best model and generate plots
    print("\n[INFO] Generating visualizations...")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        os.path.join(output_dir, "training_curves.png")
    )
    
    # Plot prediction comparisons
    plot_prediction_comparison(
        model, val_dataset.dataset, device,
        os.path.join(output_dir, "predictions.png"),
        n_samples=5
    )
    
    # Save config
    import json
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")


# ==========================================
# 5. Main Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Forward Dynamics Model")
    
    parser.add_argument("--model_type", type=str, required=True,
                       choices=['base', 'action', 'history', 'complete'],
                       help="Model variant to train")
    parser.add_argument("--data_dir", type=str,
                       default="output/2025-11-19-081615_uniform_100k_env8",
                       help="Data directory")
    parser.add_argument("--n_files", type=int, default=1000,
                       help="Number of log files to load")
    parser.add_argument("--history_len", type=int, default=3,
                       help="History length for history/complete models")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.2)
    
    args = parser.parse_args()
    
    config = {
        'model_type': args.model_type,
        'data_dir': args.data_dir,
        'n_files': args.n_files,
        'history_len': args.history_len,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'val_split': args.val_split
    }
    
    print("="*60)
    print(f"FORWARD DYNAMICS TRAINING - {args.model_type.upper()}")
    print("="*60)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k:20s}: {v}")
    print("="*60 + "\n")
    
    train_forward_model(config)

    """
    # Base model (no conditioning)
    python -m train.run_forward_dynamics --model_type base --epochs 50

    # With action only
    python -m train.run_forward_dynamics --model_type action --epochs 50

    # With history (default 3 frames)
    python -m train.run_forward_dynamics --model_type history --history_len 5 --epochs 50

    # Complete model (history + action)
    python -m train.run_forward_dynamics --model_type complete --history_len 3 --epochs 50
    """