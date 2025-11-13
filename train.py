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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LidarDataset(Dataset):
    """Dataset returning anchor, positive, negative samples for contrastive learning"""

    def __init__(self, parquet_path, num_rays=100):
        self.df = pd.read_parquet(parquet_path)
        self.num_rays = num_rays
        self.ray_cols = [f'ray_{i}' for i in range(num_rays)]
        self.lidar_data = self.df[self.ray_cols].values.astype(np.float32) / 200.0
        # Spatial coordinates
        self.x = self.df['x'].values
        self.y = self.df['y'].values
        self.theta = self.df['orientation'].values

    def __len__(self):
        # Last item has no next item, so length-1
        return len(self.df) - 1

    def __getitem__(self, idx):
        anchor = torch.from_numpy(self.lidar_data[idx])
        positive = torch.from_numpy(self.lidar_data[idx + 1])

        # Pick a random negative (not the next item)
        neg_idx = idx
        while neg_idx == idx or neg_idx == idx + 1:
            neg_idx = np.random.randint(len(self.df))
        negative = torch.from_numpy(self.lidar_data[neg_idx])

        x, y, theta = self.x[idx], self.y[idx], self.theta[idx]

        return anchor, positive, negative, x, y, theta



class LidarEncoder(nn.Module):
    """Encoder network for LIDAR readings"""

    def __init__(self, input_dim=100, hidden_dims=[256, 128], embedding_dim=64):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for one positive and one negative per anchor"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # anchor, positive, negative: [B, D]
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = pos_dist + F.relu(self.margin - neg_dist)
        return loss.mean()



def create_output_directory():
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Created output directory: {output_dir}")
    return output_dir


def save_run_info(output_dir, config, dataset_size, start_time, end_time):
    """Save run information and parameters to JSON file"""
    info = {
        "description": "LIDAR Place Recognition using Contrastive Learning",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "device": str(device),
        "dataset": {
            "num_log_files": config["n"],
            "dataset_size": dataset_size,
            "num_rays": config["num_rays"]
        },
        "model": {
            "architecture": "LidarEncoder",
            "input_dim": config["num_rays"],
            "hidden_dims": config["hidden_dims"],
            "embedding_dim": config["embedding_dim"]
        },
        "training": {
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
            "learning_rate": config["learning_rate"],
            "optimizer": "Adam"
        },
        "loss": {
            "type": "ContrastiveLoss",
            "margin": config["margin"],
            "temporal_threshold": config["temporal_threshold"]
        },
        "analysis": {
            "distance_threshold": config["distance_threshold"]
        }
    }
    
    info_path = os.path.join(output_dir, "run_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"[INFO] Saved run information to {info_path}")


def load_data(data_dir, output_dir, n):
    """Load or create merged parquet file"""
    parquet_path = os.path.join(data_dir, f"merged_{n}.parquet")

    if os.path.exists(parquet_path):
        print(f"[INFO] Using cached parquet: {parquet_path}")
        return parquet_path

    print("[INFO] Merged parquet not found. Reading CSV logs...")
    log_files = glob.glob(os.path.join(data_dir, "log_*.csv"))
    log_files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    log_files = log_files[:n]

    dfs = []
    for f in tqdm(log_files, desc="Reading CSVs", unit="file"):
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Saved merged parquet: {parquet_path}")

    return parquet_path


def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    losses = []

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running = 0.0
        for anchor, positive, negative, _, _, _ in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_loss = running / len(dataloader)
        losses.append(avg_loss)
        tqdm.write(f"[epoch {epoch+1}/{num_epochs}] loss={avg_loss:.4f}")

    return losses


def extract_embeddings(model, dataset):
    """Extract embeddings for all data points"""
    model.eval()
    embeddings = []

    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    with torch.no_grad():
        for anchor, _, _, _, _, _ in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
            anchor = anchor.to(device)
            embed_batch = model(anchor)
            embeddings.append(embed_batch.cpu().numpy())


    return np.vstack(embeddings)


def plot_embeddings(embeddings, x_coords, y_coords, save_path=None):
    """Plot embeddings in physical space using PCA"""
    print("Applying PCA to embeddings...")

    # Reduce to 3D for color mapping
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    # Normalize to [0, 1] for RGB
    embeddings_rgb = (embeddings_3d - embeddings_3d.min(axis=0)) / (embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0))

    print("Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Spatial positions colored by embeddings
    scatter = axes[0].scatter(x_coords, y_coords, c=embeddings_rgb, s=10, alpha=0.6)
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[0].set_title('Robot Path Colored by Place Embeddings (PCA RGB)')
    axes[0].set_aspect('equal')

    # Plot 2: Embedding space (first 2 PCA components)
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)

    scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                               c=np.sqrt(x_coords**2 + y_coords**2),
                               s=10, alpha=0.6, cmap='viridis')
    axes[1].set_xlabel('Embedding Dimension 1')
    axes[1].set_ylabel('Embedding Dimension 2')
    axes[1].set_title('Embedding Space (colored by distance from origin)')
    plt.colorbar(scatter2, ax=axes[1], label='Distance from origin')

    plt.tight_layout()
    if save_path:
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_place_recognition(embeddings, x_coords, y_coords, distance_threshold=5.0, output_dir=None):
    """Analyze how well embeddings capture place identity"""
    n_samples = min(1000, len(embeddings))  # Sample for efficiency
    indices = np.random.choice(len(embeddings), n_samples, replace=False)

    spatial_dists = []
    embedding_dists = []

    # Progress bar for distance computation
    total_pairs = len(indices) * (len(indices) - 1) // 2
    pbar = tqdm(total=total_pairs, desc="Computing pairwise distances", unit="pair")

    for i in indices:
        for j in indices:
            if i >= j:
                continue

            # Spatial distance
            spatial_dist = np.sqrt((x_coords[i] - x_coords[j])**2 +
                                   (y_coords[i] - y_coords[j])**2)

            # Embedding distance
            embed_dist = np.linalg.norm(embeddings[i] - embeddings[j])

            spatial_dists.append(spatial_dist)
            embedding_dists.append(embed_dist)
            pbar.update(1)

    pbar.close()

    spatial_dists = np.array(spatial_dists)
    embedding_dists = np.array(embedding_dists)

    # Plot correlation
    plt.figure(figsize=(10, 6))
    plt.hexbin(spatial_dists, embedding_dists, gridsize=50, cmap='Blues', mincnt=1)
    plt.xlabel('Spatial Distance')
    plt.ylabel('Embedding Distance')
    plt.title('Spatial vs Embedding Distance\n(Should show correlation: close in space = close in embedding)')
    plt.colorbar(label='Count')

    # Add threshold line
    plt.axvline(x=distance_threshold, color='r', linestyle='--', label=f'Spatial threshold ({distance_threshold})')
    plt.legend()
    
    if output_dir:
        save_path = os.path.join(output_dir, 'place_recognition_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved place recognition analysis to {save_path}")
    plt.close()

    # Compute metrics
    close_in_space = spatial_dists < distance_threshold
    close_in_embedding = embedding_dists < np.median(embedding_dists)

    accuracy = np.mean(close_in_space == close_in_embedding)
    print(f"\nPlace recognition accuracy: {accuracy:.2%}")
    print(f"(Fraction of pairs correctly classified as same/different place)")
    
    return accuracy


def main():
    # Configuration
    config = {
        "data_dir": "bitmap/output/2025-11-13-110326_random_walk",
        "n": 5000,  # Number of log files to process
        "num_rays": 100,
        "batch_size": 256,
        "num_epochs": 500,
        "learning_rate": 0.001,
        "hidden_dims": [256, 128],
        "embedding_dim": 64,
        "margin": 1.0,
        "temporal_threshold": 5,
        "distance_threshold": 5.0
    }

    # Create timestamped output directory
    output_dir = create_output_directory()
    start_time = datetime.now()

    # Load data
    parquet_path = load_data(config["data_dir"], output_dir, config["n"])
    dataset = LidarDataset(parquet_path, num_rays=config["num_rays"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    print(f"Dataset size: {len(dataset)} samples")

    # Create model
    model = LidarEncoder(
        input_dim=config["num_rays"],
        hidden_dims=config["hidden_dims"],
        embedding_dim=config["embedding_dim"]
    ).to(device)

    # Loss and optimizer
    criterion = ContrastiveLoss(
        margin=config["margin"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train
    print("\nTraining model...")
    losses = train_model(model, dataloader, criterion, optimizer, num_epochs=config["num_epochs"])

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved training loss plot to {loss_plot_path}")
    plt.close()

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, dataset)

    # Plot embeddings
    print("\nPlotting embeddings...")
    plot_embeddings(
        embeddings,
        dataset.x[:-1],
        dataset.y[:-1],
        save_path=os.path.join(output_dir, 'place_embeddings.png')
    )

    # Save model and embeddings
    model_path = os.path.join(output_dir, 'place_encoder.pth')
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    torch.save(model.state_dict(), model_path)
    np.save(embeddings_path, embeddings)
    print(f"Model saved to {model_path}")
    print(f"Embeddings saved to {embeddings_path}")

    # Analyze embedding similarity for revisited places
    print("\nAnalyzing place recognition...")
    accuracy = analyze_place_recognition(
        embeddings, 
        dataset.x, 
        dataset.y,
        distance_threshold=config["distance_threshold"],
        output_dir=output_dir
    )

    # Save run information
    end_time = datetime.now()
    save_run_info(output_dir, config, len(dataset), start_time, end_time)
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}")

    # plot clusters of embeddings and inspect LiDAR rays
    results = sample_clusters_and_inspect(embeddings, dataset.lidar_data, k=20, n_samples_per=5)
    plot_clusters_grid(results)


if __name__ == "__main__":
    main()