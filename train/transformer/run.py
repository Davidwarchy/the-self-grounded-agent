import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd

# Reuse existing robust infrastructure
from train.temporal_model import TemporalLidarDataset
from train.transformer.model import TemporalLidarTransformer
from train.loss import ContrastiveLoss
from train.utils import create_output_dir, save_run_info
from train.plotting import plot_train_val_loss
from train.dataset import load_or_create_parquet
from train.train import run_visualization

def train_transformer(config):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Transformer on {device}")
    
    run_dir = create_output_dir(base_dir=config['data_dir'])

        
    # --- Create Visualization Directories explicitly ---
    os.makedirs(os.path.join(run_dir, "final_embedding_map"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "clusters"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "oriented_embeddings"), exist_ok=True)
    # ------------------------------------------------------
    
    
    # Save Config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # 2. Data
    print("[INFO] Loading Data...")
    parquet_path = load_or_create_parquet(config['data_dir'], config['n_files'])
    df = pd.read_parquet(parquet_path)
    split_idx = int(0.8 * len(df))
    
    train_ds = TemporalLidarDataset(df, config['num_rays'], config['n_timesteps'], 0, split_idx)
    val_ds = TemporalLidarDataset(df, config['num_rays'], config['n_timesteps'], split_idx)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # 3. Model
    model = TemporalLidarTransformer(
        num_rays=config['num_rays'],
        n_timesteps=config['n_timesteps'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        embedding_dim=config['embedding_dim']
    ).to(device)
    
    print(f"[INFO] Transformer Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Optimization (AdamW is usually better for Transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    # Warmup + Cosine Decay scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = ContrastiveLoss(margin=config['margin'])

    # 5. Training Loop
    best_val = float('inf')
    train_losses, val_losses = [], []
    
    # Get map image path for visualization (reused logic)
    metadata_path = os.path.join(config['data_dir'], "metadata.json")
    map_image_path = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        img = meta.get("environment_parameters", {}).get("map_image")
        if img: map_image_path = os.path.join(config['data_dir'], img)

    print("[INFO] Starting Training...")
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        
        for anchor, pos, neg, _, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            
            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)
            
            loss = criterion(emb_a, emb_p, emb_n)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping is crucial for Transformers to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, pos, neg, _, _, _ in val_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                val_loss += criterion(model(anchor), model(pos), model(neg)).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            
        if (epoch + 1) % config['vis_interval'] == 0:
            run_visualization(model, epoch+1, type('Config', (), config), run_dir, map_image_path, val_ds)

    # Save Results
    plot_train_val_loss(train_losses, val_losses, best_val, os.path.join(run_dir, "loss.png"))
    print(f"[INFO] Complete. Output: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="output/2025-11-14-111925_random_walk_100k") # Update this default
    parser.add_argument("--n_files", type=int, default=1000)
    parser.add_argument("--n_timesteps", type=int, default=10) # Longer context for Transformer
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--vis_interval", type=int, default=5)
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'n_files': args.n_files,
        'num_rays': 100,
        'n_timesteps': args.n_timesteps,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'embedding_dim': 64,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'margin': 1.0,
        'num_epochs': args.num_epochs,
        'vis_interval': args.vis_interval,
        'learning_rate': args.lr # Alias for visualization compat
    }
    
    train_transformer(config)