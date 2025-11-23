"""
Let's try to see if we can reconstruct lidar readings. 

The setup is simple: 
- Inputs lidar reading
- Encode to latent dimension 
- Output reconstruction 
"""
"""
LiDAR Autoencoder: Reconstruct LiDAR readings from compressed latent space
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
import matplotlib.pyplot as plt
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train.dataset import load_or_create_parquet


# ==========================================
# 1. Autoencoder Model
# ==========================================

class LidarAutoencoder(nn.Module):
    """Simple autoencoder for LiDAR reconstruction"""
    def __init__(self, input_dim=100, latent_dim=16):
        super().__init__()
        
        # Encoder: compress to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, latent_dim)
        )
        
        # Decoder: reconstruct from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output in [0, 1] range like normalized input
        )
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent space to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode then decode"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


# ==========================================
# 2. Dataset
# ==========================================

class LidarAutoencoderDataset(Dataset):
    """Dataset for autoencoder training"""
    def __init__(self, parquet_path, num_rays=100):
        self.df = pd.read_parquet(parquet_path)
        ray_cols = [f'ray_{i}' for i in range(num_rays)]
        # Normalize to [0, 1]
        self.lidar = (self.df[ray_cols].values.astype(np.float32) / 200.0)
        
        # Also store position for visualization
        self.x = self.df['x'].values
        self.y = self.df['y'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        lidar = torch.from_numpy(self.lidar[idx])
        return lidar, self.x[idx], self.y[idx]


# ==========================================
# 3. Visualization Functions
# ==========================================

def plot_reconstructions(model, dataset, device, save_path, n_samples=5):
    """Plot original vs reconstructed LiDAR scans"""
    model.eval()
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    mse_list = []
    mae_list = []
    
    with torch.no_grad():
        for i in range(n_samples):
            idx = np.random.randint(len(dataset))
            original, _, _ = dataset[idx]
            original_input = original.unsqueeze(0).to(device)
            
            reconstruction, latent = model(original_input)
            reconstruction = reconstruction.cpu().squeeze().numpy()
            original = original.numpy()
            
            # Calculate errors
            mse = np.mean((original - reconstruction) ** 2)
            mae = np.mean(np.abs(original - reconstruction))
            mse_list.append(mse)
            mae_list.append(mae)
            
            # Plot
            ax = axes[i]
            ax.plot(original, label='Original', linewidth=2.5, alpha=0.8, color='blue')
            ax.plot(reconstruction, label='Reconstructed', linewidth=2, 
                   alpha=0.8, linestyle='--', color='red')
            ax.fill_between(range(len(original)), original, reconstruction, 
                           alpha=0.2, color='purple')
            
            ax.set_xlabel('Ray Index', fontsize=10)
            ax.set_ylabel('Normalized Distance', fontsize=10)
            ax.set_title(f'Sample {i+1} | MSE: {mse:.6f} | MAE: {mae:.6f} | '
                        f'Latent norm: {latent.norm().item():.2f}', 
                        fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
    
    # Summary statistics
    fig.text(0.5, 0.02, 
             f'Average MSE: {np.mean(mse_list):.6f} | Average MAE: {np.mean(mae_list):.6f}',
             ha='center', fontsize=12, weight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved reconstructions to {save_path}")
    print(f"[INFO] Avg MSE: {np.mean(mse_list):.6f} | Avg MAE: {np.mean(mae_list):.6f}")
    
    return np.mean(mse_list), np.mean(mae_list)


def plot_latent_space(model, dataset, device, save_path, n_samples=2000):
    """Visualize 2D projection of latent space colored by position"""
    model.eval()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    latents = []
    positions = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Encoding samples", leave=False):
            lidar, x, y = dataset[idx]
            lidar_input = lidar.unsqueeze(0).to(device)
            _, latent = model(lidar_input)
            latents.append(latent.cpu().numpy())
            positions.append([x, y])
    
    latents = np.vstack(latents)
    positions = np.array(positions)
    
    # If latent_dim > 2, use PCA for visualization
    if latents.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        title_suffix = f" (PCA, explained var: {pca.explained_variance_ratio_.sum():.2%})"
    else:
        latents_2d = latents
        title_suffix = ""
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Latent space colored by X position
    scatter1 = ax1.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                          c=positions[:, 0], cmap='viridis', 
                          s=10, alpha=0.6)
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_title(f'Latent Space (colored by X position){title_suffix}')
    plt.colorbar(scatter1, ax=ax1, label='X Position')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Latent space colored by Y position
    scatter2 = ax2.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                          c=positions[:, 1], cmap='plasma', 
                          s=10, alpha=0.6)
    ax2.set_xlabel('Latent Dimension 1')
    ax2.set_ylabel('Latent Dimension 2')
    ax2.set_title(f'Latent Space (colored by Y position){title_suffix}')
    plt.colorbar(scatter2, ax=ax2, label='Y Position')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved latent space visualization to {save_path}")


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Linear scale
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2, marker='s', markersize=4)
    
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax1.scatter([best_epoch], [best_val], color='red', s=150, zorder=5, marker='*')
    ax1.text(best_epoch, best_val, f'  Best: {best_val:.6f}', 
            verticalalignment='bottom', fontsize=10)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_losses, label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss (log scale)')
    ax2.set_title('Training Progress (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved training curves to {save_path}")


# ==========================================
# 4. Training Function
# ==========================================

def train_autoencoder(config):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")
    
    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_dir = f"{config['data_dir']}/autoencoder/latent{config['latent_dim']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        parquet_path = load_or_create_parquet(config['data_dir'], config['n_files'])
        full_dataset = LidarAutoencoderDataset(parquet_path)
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
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    model = LidarAutoencoder(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model has {total_params:,} parameters")
    print(f"[INFO] Latent dimension: {config['latent_dim']}")
    print(f"[INFO] Compression ratio: {config['input_dim']/config['latent_dim']:.1f}x\n")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_model_path = os.path.join(output_dir, "best_model.pth")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        
        for batch_data in tqdm(train_loader, 
                              desc=f"Epoch {epoch+1}/{config['epochs']} [Train]",
                              leave=False):
            lidar, _, _ = batch_data
            lidar = lidar.to(device)
            
            # Forward pass
            reconstruction, _ = model(lidar)
            loss = criterion(reconstruction, lidar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                lidar, _, _ = batch_data
                lidar = lidar.to(device)
                
                reconstruction, _ = model(lidar)
                loss = criterion(reconstruction, lidar)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:02d} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, best_model_path)
            print(f"    ✓ New Best! Saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for visualization
    print("\n[INFO] Generating visualizations...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        os.path.join(output_dir, "training_curves.png")
    )
    
    # Plot reconstructions
    plot_reconstructions(
        model, val_dataset.dataset, device,
        os.path.join(output_dir, "reconstructions.png"),
        n_samples=8
    )
    
    # Plot latent space
    plot_latent_space(
        model, val_dataset.dataset, device,
        os.path.join(output_dir, "latent_space.png"),
        n_samples=3000
    )
    
    # Save config
    import json
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save final embeddings
    print("\n[INFO] Saving latent embeddings for full dataset...")
    model.eval()
    all_latents = []
    all_positions = []
    
    full_loader = DataLoader(full_dataset, batch_size=512, shuffle=False)
    with torch.no_grad():
        for lidar, x, y in tqdm(full_loader, desc="Encoding full dataset"):
            lidar = lidar.to(device)
            _, latent = model(lidar)
            all_latents.append(latent.cpu().numpy())
            all_positions.append(np.stack([x, y], axis=1))
    
    all_latents = np.vstack(all_latents)
    all_positions = np.vstack(all_positions)
    
    np.save(os.path.join(output_dir, "latent_embeddings.npy"), all_latents)
    np.save(os.path.join(output_dir, "positions.npy"), all_positions)
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Latent embeddings shape: {all_latents.shape}")


# ==========================================
# 5. Main Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train LiDAR Autoencoder")
    
    parser.add_argument("--data_dir", type=str,
                       default="output/2025-11-19-081615_uniform_100k_env8",
                       help="Data directory")
    parser.add_argument("--n_files", type=int, default=1000)
    parser.add_argument("--input_dim", type=int, default=100,
                       help="Number of LiDAR rays")
    parser.add_argument("--latent_dim", type=int, default=16,
                       help="Latent space dimension")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    config = vars(args)
    
    print("="*60)
    print("LIDAR AUTOENCODER TRAINING")
    print("="*60)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k:20s}: {v}")
    print("="*60 + "\n")
    
    train_autoencoder(config)


if __name__ == "__main__":
    main()
    """
    Usage examples:
    
    # Basic training with 16D latent space
    python -m train.reconstruction.train --latent_dim 16 --epochs 50
    
    # Try different compression ratios
    python -m train.reconstruction.train --latent_dim 8 --epochs 50   # 12.5x compression
    python -m train.reconstruction.train --latent_dim 32 --epochs 50  # 3.1x compression
    python -m train.reconstruction.train --latent_dim 64 --epochs 50  # 1.56x compression
    
    # Longer training
    python -m train.reconstruction.train --latent_dim 16 --epochs 100 --patience 20
    """