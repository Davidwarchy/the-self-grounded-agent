# train/train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd
from sklearn.decomposition import PCA

from train.dataset import LidarDataset, load_or_create_parquet
from train.model import LidarEncoder
from train.loss import ContrastiveLoss
from train.utils import create_output_dir, save_run_info
from train.plotting import plot_train_val_loss
from train.clustering import sample_clusters_and_inspect, plot_clusters_grid
from train.spatial import plot_embedding_distribution, plot_oriented_embedding_distribution


def extract_embeddings(model, dataset, batch_size=512):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for anchor, _, _, _, _, _ in tqdm(loader, desc="Extracting embeddings", leave=False):
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

def run_visualization(model, epoch, config, run_dir, map_image_path, val_ds):
    """Generates and saves all embedding visualizations for a given epoch."""
    
    # 1. Setup paths and targets (redefining variables from train() for clarity)
    emb_map_dir = os.path.join(run_dir, "final_embedding_map")
    cluster_dir = os.path.join(run_dir, "clusters")
    oriented_emb_dir = os.path.join(run_dir, "oriented_embeddings")
    target_orientations = [0, 90, 180, 270]          
    orientation_tolerance = 15
    orientation_dirs = {ori: os.path.join(oriented_emb_dir, str(ori)) for ori in target_orientations}
    for ori_dir in orientation_dirs.values():
        os.makedirs(ori_dir, exist_ok=True)

    # 2. Extract embeddings
    model.eval()
    val_emb = extract_embeddings(model, val_ds)
    # Save embeddings for this epoch (useful for later analysis)
    if config.save_embeddings:
        np.save(os.path.join(run_dir, f"val_emb_epoch_{epoch}.npy"), val_emb)
    valid = val_ds.valid_slice

    # 3. Fit PCA once on *all* validation embeddings (for consistent coloring)
    pca_global = PCA(n_components=3, random_state=42)
    rgb_3d = pca_global.fit_transform(val_emb)
    rgb_min = rgb_3d.min(axis=0)
    rgb_ptp = np.ptp(rgb_3d, axis=0) + 1e-8
    rgb_global = (rgb_3d - rgb_min) / rgb_ptp

    # 4. Global embedding map – use pre-computed colors
    emb_path = os.path.join(emb_map_dir, f"epoch_{epoch}.png")
    plot_embedding_distribution(
        val_emb, valid['x'], valid['y'],
        emb_path,
        map_image_path=map_image_path,
        rgb_precomputed=rgb_global
    )

    # 5. Orientation-filtered maps – reuse same PCA & normalization
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

    # 6. Clustering grid
    cluster_results = sample_clusters_and_inspect(val_emb, valid['lidar'])
    cluster_path = os.path.join(cluster_dir, f"epoch_{epoch}.png")
    plot_clusters_grid(cluster_results, save_path=cluster_path)

    # print(f"[INFO] Epoch {epoch}/{config.num_epochs} Visualization Complete. Plots saved to {emb_map_dir} and related folders.")


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_output_dir(base_dir=config.data_dir)
    start_time = datetime.now()

    # ------------------------------------------------------------------
    # 1. Load map image ONCE (from metadata.json)
    # ------------------------------------------------------------------
    metadata_path = os.path.join(config.data_dir, "metadata.json")
    map_image_path = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        map_image_name = metadata.get("environment_parameters", {}).get("map_image")
        if map_image_name:
            possible_paths = [
                os.path.join(config.data_dir, map_image_name),
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
        print("[WARN] Map image not found – overlay will be omitted")

    # ------------------------------------------------------------------
    # 2. Create periodic output directories
    # ------------------------------------------------------------------
    emb_map_dir = os.path.join(run_dir, "final_embedding_map")
    cluster_dir = os.path.join(run_dir, "clusters")
    oriented_emb_dir = os.path.join(run_dir, "oriented_embeddings")
    os.makedirs(emb_map_dir, exist_ok=True)
    os.makedirs(cluster_dir, exist_ok=True)
    os.makedirs(oriented_emb_dir, exist_ok=True)

    # Define orientations to visualize (e.g., cardinal directions)
    target_orientations = [0, 90, 180, 270]          # degrees
    orientation_tolerance = 15                      # ±15 degrees

    # **NEW**: create a sub-folder for *each* orientation
    orientation_dirs = {}
    for ori in target_orientations:
        ori_dir = os.path.join(oriented_emb_dir, str(ori))
        os.makedirs(ori_dir, exist_ok=True)
        orientation_dirs[ori] = ori_dir

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    parquet_path = load_or_create_parquet(config.data_dir, config.n_files)
    df = pd.read_parquet(parquet_path)
    split_idx = int(0.8 * (len(df) - 1))

    train_ds = LidarDataset(parquet_path, config.num_rays, 0, split_idx)
    val_ds = LidarDataset(parquet_path, config.num_rays, split_idx)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = LidarEncoder(config.num_rays, config.hidden_dims, config.embedding_dim).to(device)
    criterion = ContrastiveLoss(config.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    best_val = float('inf')
    train_losses, val_losses = [], []
    best_path = os.path.join(run_dir, "best_model.pth")

    run_visualization(epoch=-1, model=model, config=config, run_dir=run_dir,
                      map_image_path=map_image_path, val_ds=val_ds)

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
        
        tqdm.write(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        # --------------------------------------------------------------
        # 6. Periodic visualization (every vis_interval or last epoch)
        # --------------------------------------------------------------
        if (epoch + 1) % config.vis_interval == 0 or epoch == config.num_epochs - 1:
            run_visualization(
                model, epoch + 1, config, run_dir,
                map_image_path, val_ds
            )
        
    # ------------------------------------------------------------------
    # 7. Final saves (symlinks for backward compatibility)
    # ------------------------------------------------------------------
    final_emb_path = os.path.join(emb_map_dir, f"epoch_{config.num_epochs}.png")
    final_cluster_path = os.path.join(cluster_dir, f"epoch_{config.num_epochs}.png")

    if os.path.exists(final_emb_path):
        try:
            os.symlink(os.path.relpath(final_emb_path, run_dir),
                       os.path.join(run_dir, "final_embedding_map.png"))
        except OSError:  # Windows may need admin or use copy
            import shutil
            shutil.copy2(final_emb_path, os.path.join(run_dir, "final_embedding_map.png"))

    if os.path.exists(final_cluster_path):
        try:
            os.symlink(os.path.relpath(final_cluster_path, run_dir),
                       os.path.join(run_dir, "clusters.png"))
        except OSError:
            import shutil
            shutil.copy2(final_cluster_path, os.path.join(run_dir, "clusters.png"))

    # ------------------------------------------------------------------
    # 8. Loss curve & run info
    # ------------------------------------------------------------------
    plot_train_val_loss(train_losses, val_losses, best_val,
                        os.path.join(run_dir, "loss_curve.png"))

    end_time = datetime.now()
    save_run_info(run_dir, config, len(train_ds), len(val_ds), split_idx,
                  best_val, start_time, end_time)
    
    # After training, load best model and save full final embeddings
    print("[INFO] Saving final embeddings from best model...")
    model.load_state_dict(torch.load(best_path))
    model.eval()

    full_ds = LidarDataset(parquet_path, config.num_rays)
    final_embeddings = extract_embeddings(model, full_ds)
    final_emb_path = os.path.join(run_dir, "final_embeddings.npy")
    np.save(final_emb_path, final_embeddings)
    print(f"[INFO] Final embeddings saved: {final_emb_path}")

    print(f"\nTraining complete! Results in: {run_dir}")
    print(f"Best validation loss: {best_val:.4f}")