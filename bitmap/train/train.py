# train/train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import pandas as pd

from train.dataset import LidarDataset, load_or_create_parquet
from train.model import LidarEncoder
from train.loss import ContrastiveLoss
from train.utils import create_output_dir, save_run_info
from train.plotting import plot_train_val_loss
from train.clustering import sample_clusters_and_inspect, plot_clusters_grid
from train.spatial import plot_embedding_distribution

def extract_embeddings(model, dataset, batch_size=512):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for anchor, _, _, _, _, _ in tqdm(loader, desc="Embeddings", leave=False):
            emb = model(anchor.to(model.device)).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)

def validate(model, loader, criterion):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for a, p, n, _, _, _ in loader:
            a, p, n = a.to(model.device), p.to(model.device), n.to(model.device)
            loss += criterion(model(a), model(p), model(n)).item()
    return loss / len(loader)

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_output_dir()
    start_time = datetime.now()

    # Data
    parquet_path = load_or_create_parquet(config.data_dir, config.n_files)
    df = pd.read_parquet(parquet_path)
    split_idx = int(0.8 * (len(df) - 1))

    train_ds = LidarDataset(parquet_path, config.num_rays, 0, split_idx)
    val_ds = LidarDataset(parquet_path, config.num_rays, split_idx)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Model
    model = LidarEncoder(config.num_rays, config.hidden_dims, config.embedding_dim).to(device)
    criterion = ContrastiveLoss(config.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train
    best_val = float('inf')
    train_losses, val_losses = [], []
    best_path = os.path.join(run_dir, "best_model.pth")

    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        for a, p, n, _, _, _ in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % config.vis_interval == 0:
            val_emb = extract_embeddings(model, val_ds)
            np.save(os.path.join(run_dir, f"val_emb_epoch_{epoch+1}.npy"), val_emb)
            valid = val_ds.valid_slice
            plot_embedding_distribution(val_emb, valid['x'], valid['y'],
                os.path.join(run_dir, f"emb_epoch_{epoch+1}.png"))

    # Final
    plot_train_val_loss(train_losses, val_losses, best_val,
                        os.path.join(run_dir, "loss_curve.png"))

    # Final embeddings
    model.load_state_dict(torch.load(best_path))
    final_emb = extract_embeddings(model, val_ds)
    np.save(os.path.join(run_dir, "val_embeddings_final.npy"), final_emb)
    valid = val_ds.valid_slice
    plot_embedding_distribution(final_emb, valid['x'], valid['y'],
                                os.path.join(run_dir, "final_embedding_map.png"))

    # Clustering
    cluster_results = sample_clusters_and_inspect(final_emb, valid['lidar'])
    plot_clusters_grid(cluster_results, save_path=os.path.join(run_dir, "clusters.png"))

    end_time = datetime.now()
    save_run_info(run_dir, config, len(train_ds), len(val_ds), split_idx, best_val, start_time, end_time)
    print(f"\nTraining complete! Results in: {run_dir}")