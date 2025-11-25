import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset import load_or_create_parquet, LidarDataset
from train.model import LidarEncoder
from train.spatial import plot_embedding_distribution
from analysis.map_utils import get_map_image_path
from analysis.plotting import plot_correlations

def calculate_map_expanse(x_coords, y_coords, map_image_path):
    """
    Calculates the horizontal and vertical 'thickness' (expanse) of the free space 
    at the given coordinates by casting rays on the map image.
    """
    print(f"[INFO] Calculating map expanse features from {map_image_path}...")
    
    # Load map: 0=free, 1=obstacle (inverted from typical images where 0=black=obstacle)
    # We assume standard image: 0 (black) is obstacle, 255 (white) is free.
    img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load map: {map_image_path}")
    
    # Binarize: Obstacles = 0, Free = 1
    _, binary_map = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    
    h_expanse = []
    v_expanse = []
    
    h, w = binary_map.shape
    
    # We use vectorization where possible, but for geometric raycasting on grid, 
    # per-point lookups with np.where are reasonably fast for 100k points.
    for x, y in tqdm(zip(x_coords, y_coords), total=len(x_coords), desc="Geo-features"):
        ix, iy = int(x), int(y)
        
        # Clamp to bounds
        ix = max(0, min(w-1, ix))
        iy = max(0, min(h-1, iy))
        
        # If the robot is inside a wall (simulation glitch), return 0
        if binary_map[iy, ix] == 0:
            h_expanse.append(0)
            v_expanse.append(0)
            continue

        # --- Horizontal Expanse ---
        row = binary_map[iy, :]
        obstacles = np.where(row == 0)[0]
        
        # Find nearest wall to the left
        left_walls = obstacles[obstacles < ix]
        left_bound = left_walls[-1] if len(left_walls) > 0 else 0
        
        # Find nearest wall to the right
        right_walls = obstacles[obstacles > ix]
        right_bound = right_walls[0] if len(right_walls) > 0 else w
        
        h_expanse.append(right_bound - left_bound)

        # --- Vertical Expanse ---
        col = binary_map[:, ix]
        obstacles_v = np.where(col == 0)[0]
        
        # Find nearest wall above
        up_walls = obstacles_v[obstacles_v < iy]
        up_bound = up_walls[-1] if len(up_walls) > 0 else 0
        
        # Find nearest wall below
        down_walls = obstacles_v[obstacles_v > iy]
        down_bound = down_walls[0] if len(down_walls) > 0 else h
        
        v_expanse.append(down_bound - up_bound)

    return np.array(h_expanse), np.array(v_expanse)

def load_trained_model(train_dir, device):
    """Loads model and config from training directory."""
    info_path = os.path.join(train_dir, "run_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"run_info.json not found in {train_dir}")
        
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    # Extract model config
    model_cfg = info.get("model", {})
    hidden_dims = model_cfg.get("hidden_dims", [256, 128])
    embedding_dim = model_cfg.get("embedding_dim", 64)
    # Fallback if config structure is different
    if "config" in info: 
         hidden_dims = info["config"].get("hidden_dims", [256, 128])
         embedding_dim = info["config"].get("embedding_dim", 64)

    # Initialize model
    # We assume 100 rays as standard, but ideally this comes from config too
    model = LidarEncoder(input_dim=100, hidden_dims=hidden_dims, embedding_dim=embedding_dim)
    
    # Load weights
    model_path = os.path.join(train_dir, "best_model.pth")
    # Handle checkpoint dictionary vs direct state dict
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {model_path}")
    return model

def analyze_correlations(data_dir, train_dir, n_files=1000, batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    parquet_path = load_or_create_parquet(data_dir, n_files)
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Data loaded: {len(df)} samples")

    # 2. Extract Basic Features
    x = df['x'].values
    y = df['y'].values
    theta = df['orientation'].values
    actions = df['action'].values

    # Lidar Data
    lidar_cols = [c for c in df.columns if c.startswith('ray_')]
    lidar_data = df[lidar_cols].values.astype(np.float32) / 200.0 # Normalize
    
    # 3. Compute Embeddings
    model = load_trained_model(train_dir, device)
    
    print("[INFO] Computing embeddings...")
    embeddings = []
    # Process in batches
    for i in tqdm(range(0, len(lidar_data), batch_size)):
        batch = torch.from_numpy(lidar_data[i:i+batch_size]).to(device)
        with torch.no_grad():
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    # 4. Compute Derived Features
    print("[INFO] Computing derived features...")
    
    # A. Average Lidar Distance (Distance to wall proxy)
    avg_lidar_dist = np.mean(lidar_data, axis=1)
    
    # B. Map Thickness/Expanse
    map_path = get_map_image_path(data_dir)
    if map_path:
        h_expanse, v_expanse = calculate_map_expanse(x, y, map_path)
    else:
        print("[WARN] Map image not found. Skipping expanse calculation.")
        h_expanse = np.zeros(len(x))
        v_expanse = np.zeros(len(y))

    # 5. Prepare Correlation Dictionary
    features = {
        "Avg Wall Dist": avg_lidar_dist,
        "Horizontal Expanse": h_expanse,
        "Vertical Expanse": v_expanse,
        "Position X": x,
        "Position Y": y,
        "Orientation": theta,
        "Cos(Orientation)": np.cos(np.deg2rad(theta)), # Often correlates better
        "Sin(Orientation)": np.sin(np.deg2rad(theta)),
        "Action": actions
    }

    # 6. Run Correlation Analysis
    save_dir = os.path.join(train_dir, "correlation_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    # Reduce embeddings to RGB (3 components) for intuitive correlation
    pca = PCA(n_components=2)
    embeddings_rgb = pca.fit_transform(embeddings)
    
    # Normalize RGB for visualization logic consistency (though correlation doesn't care about scaling)
    rgb_norm = (embeddings_rgb - embeddings_rgb.min(0)) / (embeddings_rgb.max(0) - embeddings_rgb.min(0))

    # Use existing plotting utility
    print("[INFO] Plotting correlations with PCA Components (RGB)...")
    plot_correlations(
        features, 
        rgb=rgb_norm, 
        use_pca=True, 
        save_path=os.path.join(save_dir, "correlations_pca.pdf"),
        show_plot=False
    )
    
    # Optional: Plot correlation with raw dimensions (summed or average)
    # We create a dummy "Magnitude" feature to see if overall activation correlates
    emb_magnitude = np.linalg.norm(embeddings, axis=1)
    
    # Plot feature vs Embedding Magnitude
    plt.figure(figsize=(10, 6))
    corrs = []
    labels = []
    for name, data in features.items():
        corr, _ = pearsonr(data, emb_magnitude)
        corrs.append(corr)
        labels.append(name)
        
    plt.bar(labels, corrs, color='teal', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Pearson Correlation")
    plt.title("Correlation: Features vs Embedding Vector Norm")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_magnitude.png"))
    plt.close()

    print(f"[SUCCESS] Analysis complete. Results saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlate embeddings with physical attributes")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw data output")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training output containing model")
    parser.add_argument("--n_files", type=int, default=1000)
    
    args = parser.parse_args()
    
    analyze_correlations(args.data_dir, args.train_dir, args.n_files)