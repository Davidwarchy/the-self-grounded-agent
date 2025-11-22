# train/run_temporal_train.py
"""
Training script for Temporal CNN LiDAR Encoder.
Uses sequences of LiDAR scans instead of single frames.

Usage:
    python -m train.run_temporal_train --n_timesteps 5 --num_epochs 50
"""
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd

from train.temporal_model import TemporalLidarEncoderCNN, TemporalLidarDataset
from train.loss import ContrastiveLoss
from train.utils import create_output_dir, save_run_info
from train.plotting import plot_train_val_loss
from train.dataset import load_or_create_parquet
from train.spatial import (
    plot_embedding_distribution, 
    plot_oriented_embedding_distribution,
    plot_cluster_spatial_distribution
)
from train.clustering import sample_clusters_and_inspect, plot_clusters_grid
from sklearn.decomposition import PCA


def extract_embeddings(model, dataset, batch_size=512):
    """Extract embeddings for all sequences in dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    
    with torch.no_grad():
        for anchor, _, _, _, _, _ in tqdm(loader, desc="Extracting embeddings", leave=False):
            emb = model(anchor.to(model.device)).cpu().numpy()
            embeddings.append(emb)
    
    return np.vstack(embeddings)


def validate(model, loader, criterion):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for anchor, pos, neg, _, _, _ in loader:
            a = anchor.to(model.device)
            p = pos.to(model.device)
            n = neg.to(model.device)
            
            loss = criterion(model(a), model(p), model(n))
            total_loss += loss.item()
    
    return total_loss / len(loader)


def generate_visualizations(model, val_ds, epoch, run_dir, map_image_path,
                           emb_map_dir, cluster_dir, orientation_dirs,
                           target_orientations, orientation_tolerance):
    """Generate and save all visualizations for a given epoch."""
    
    # Extract embeddings
    val_emb = extract_embeddings(model, val_ds)
    np.save(os.path.join(run_dir, f"val_emb_epoch_{epoch}.npy"), val_emb)
    valid = val_ds.valid_slice
    
    # Fit PCA once for consistent coloring across all plots
    pca_global = PCA(n_components=3, random_state=42)
    rgb_3d = pca_global.fit_transform(val_emb)
    rgb_min = rgb_3d.min(axis=0)
    rgb_ptp = np.ptp(rgb_3d, axis=0) + 1e-8
    rgb_global = (rgb_3d - rgb_min) / rgb_ptp
    
    # 1. Global embedding map
    emb_path = os.path.join(emb_map_dir, f"epoch_{epoch}.png")
    plot_embedding_distribution(
        val_emb, valid['x'], valid['y'],
        emb_path,
        map_image_path=map_image_path,
        rgb_precomputed=rgb_global
    )
    
    # 2. Orientation-filtered maps
    for orientation in target_orientations:
        oriented_path = os.path.join(
            orientation_dirs[orientation],
            f"epoch_{epoch}.png"
        )
        plot_oriented_embedding_distribution(
            val_emb, valid['x'], valid['y'], valid['theta'],
            oriented_path,
            target_orientation=orientation,
            tolerance=orientation_tolerance,
            map_image_path=map_image_path,
            pca_global=pca_global,
            rgb_min=rgb_min,
            rgb_ptp=rgb_ptp
        )
    
    # 3. Clustering visualization
    cluster_results = sample_clusters_and_inspect(val_emb, valid['lidar'])
    cluster_path = os.path.join(cluster_dir, f"epoch_{epoch}.png")
    plot_clusters_grid(cluster_results, save_path=cluster_path)
    
    print(f"[INFO] Visualizations for epoch {epoch} saved")


def train_temporal_model(config):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}")
    
    # Create output directory
    run_dir = create_output_dir(base_dir=config['data_dir'])
    start_time = datetime.now()
    
    # Create visualization directories
    emb_map_dir = os.path.join(run_dir, "final_embedding_map")
    cluster_dir = os.path.join(run_dir, "clusters")
    oriented_emb_dir = os.path.join(run_dir, "oriented_embeddings")
    os.makedirs(emb_map_dir, exist_ok=True)
    os.makedirs(cluster_dir, exist_ok=True)
    os.makedirs(oriented_emb_dir, exist_ok=True)
    
    # Define orientations to visualize
    target_orientations = [0, 90, 180, 270]
    orientation_tolerance = 15
    orientation_dirs = {}
    for ori in target_orientations:
        ori_dir = os.path.join(oriented_emb_dir, str(ori))
        os.makedirs(ori_dir, exist_ok=True)
        orientation_dirs[ori] = ori_dir
    
    # Load map image
    metadata_path = os.path.join(config['data_dir'], "metadata.json")
    map_image_path = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        map_image_name = metadata.get("environment_parameters", {}).get("map_image")
        if map_image_name:
            possible_paths = [
                os.path.join(config['data_dir'], map_image_name),
                os.path.join("environments", "images", map_image_name),
                os.path.join("..", "environments", "images", map_image_name),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    map_image_path = p
                    break
    if map_image_path:
        print(f"[INFO] Using map image: {map_image_path}")
    else:
        print("[WARN] Map image not found - overlay will be omitted")
    
    # Save config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Config saved to {config_path}")
    
    # Load data
    print("[INFO] Loading data...")
    parquet_path = load_or_create_parquet(config['data_dir'], config['n_files'])
    df = pd.read_parquet(parquet_path)
    
    # Split train/val
    split_idx = int(0.8 * len(df))
    
    train_ds = TemporalLidarDataset(
        df, 
        num_rays=config['num_rays'],
        n_timesteps=config['n_timesteps'],
        start_idx=0,
        end_idx=split_idx
    )
    
    val_ds = TemporalLidarDataset(
        df,
        num_rays=config['num_rays'],
        n_timesteps=config['n_timesteps'],
        start_idx=split_idx
    )
    
    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples: {len(val_ds)}")
    print(f"[INFO] Temporal window: {config['n_timesteps']} frames")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("[INFO] Creating model...")
    model = TemporalLidarEncoderCNN(
        input_dim=config['num_rays'],
        n_timesteps=config['n_timesteps'],
        embedding_dim=config['embedding_dim']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=config['margin'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler (optional but helpful)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    best_model_path = os.path.join(run_dir, "best_model.pth")
    
    print("\n[INFO] Starting training...")
    
    # Generate initial visualization (epoch 0)
    print("[INFO] Generating initial visualizations (epoch 0)...")
    generate_visualizations(
        model, val_ds, 0, run_dir, map_image_path,
        emb_map_dir, cluster_dir, orientation_dirs,
        target_orientations, orientation_tolerance
    )
    
    for epoch in range(config['num_epochs']):
        # Train
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for anchor, pos, neg, _, _, _ in pbar:
            a = anchor.to(device)
            p = pos.to(device)
            n = neg.to(device)
            
            # Forward pass
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            
            # Compute loss
            loss = criterion(emb_a, emb_p, emb_n)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_model_path)
            print(f"    ✓ New best model saved (val_loss: {val_loss:.4f})")
        
        # Generate visualizations every vis_interval epochs
        if (epoch + 1) % config['vis_interval'] == 0 or epoch == config['num_epochs'] - 1:
            print(f"[INFO] Generating visualizations (epoch {epoch+1})...")
            generate_visualizations(
                model, val_ds, epoch + 1, run_dir, map_image_path,
                emb_map_dir, cluster_dir, orientation_dirs,
                target_orientations, orientation_tolerance
            )
    
    # Save final embeddings
    print("\n[INFO] Extracting final embeddings...")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    train_emb = extract_embeddings(model, train_ds)
    val_emb = extract_embeddings(model, val_ds)
    
    np.save(os.path.join(run_dir, "train_embeddings.npy"), train_emb)
    np.save(os.path.join(run_dir, "val_embeddings.npy"), val_emb)
    print(f"[INFO] Embeddings saved to {run_dir}")
    
    # Save loss curve
    plot_train_val_loss(
        train_losses, val_losses, best_val_loss,
        os.path.join(run_dir, "loss_curve.png")
    )
    
    # Save run info
    end_time = datetime.now()
    save_run_info(
        run_dir, config, len(train_ds), len(val_ds),
        split_idx, best_val_loss, start_time, end_time
    )
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"Results saved to: {run_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal CNN LiDAR Encoder")
    
    # Data
    parser.add_argument("--data_dir", type=str, 
                       default="output/2025-11-19-081615_uniform_100k_env8",
                       help="Directory containing log files")
    parser.add_argument("--n_files", type=int, default=1000,
                       help="Number of log files to load")
    parser.add_argument("--num_rays", type=int, default=100,
                       help="Number of LiDAR rays per scan")
    
    # Temporal
    parser.add_argument("--n_timesteps", type=int, default=5,
                       help="Number of consecutive timesteps per sequence")
    
    # Model
    parser.add_argument("--embedding_dim", type=int, default=64,
                       help="Dimension of output embedding")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Margin for contrastive loss")
    parser.add_argument("--vis_interval", type=int, default=5,
                       help="Generate visualizations every N epochs")
    
    args = parser.parse_args()
    
    # Create config dict
    config = {
        'data_dir': args.data_dir,
        'n_files': args.n_files,
        'num_rays': args.num_rays,
        'n_timesteps': args.n_timesteps,
        'embedding_dim': args.embedding_dim,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'margin': args.margin,
        'vis_interval': args.vis_interval
    }
    
    print("="*60)
    print("TEMPORAL CNN LIDAR ENCODER - TRAINING")
    print("="*60)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k:20s}: {v}")
    print("="*60 + "\n")
    
    train_temporal_model(config)