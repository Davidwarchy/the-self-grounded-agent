"""
Analysis Script to test "Random Structure" theory by using an untrained network 
with ARBITRARY architecture to generate embeddings from LIDAR data.

Generates visualizations of the spatial layout and oriented embeddings.
"""
import os, sys, json, torch, argparse
import pandas as pd
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_parquet import load_data
# from train.model import LidarEncoder  <-- Replaced with flexible local class below
from analysis.plotting import plot_embeddings_rgb, plot_oriented_embeddings_rgb

# --- Flexible Random Network Definition ---
class RandomEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # Note: No final activation implies a linear projection for the embedding
        )

    def forward(self, x):
        return self.net(x)

def analyze_random_structure(data_dir, hidden_dim, output_dim, n_files=1000):
    # 1. Load Metadata & Data
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Error: Metadata not found at {meta_path}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    print("Loaded metadata:", meta)
    
    num_rays = meta.get("environment_parameters", {}).get("num_rays", 100)
    df_path = load_data(data_dir, n=n_files)
    df = pd.read_parquet(df_path)

    print(f"Loaded data with {len(df)} samples.")
    
    # 2. Random Projection (Variable Architecture)
    print(f"\n--- Initializing Random Network ---")
    print(f"Input: {num_rays} -> Hidden: {hidden_dim} -> Output: {output_dim}")
    
    # Normalize inputs (divide by max range, usually 200.0 in your config)
    lidar_tensor = torch.tensor(df[[f'ray_{i}' for i in range(num_rays)]].values / 200.0).float()
    
    # Initialize the flexible model
    model = RandomEncoder(input_dim=num_rays, hidden_dim=hidden_dim, output_dim=output_dim).eval()
    
    print(f"Generating random embeddings...")
    with torch.no_grad():
        embeddings = model(lidar_tensor).numpy()

    # 3. Visualize
    out_dir = os.path.join(data_dir, "analysis_random_structure")
    os.makedirs(out_dir, exist_ok=True)
    
    # Create unique filenames based on the architecture
    arch_tag = f"h{hidden_dim}_o{output_dim}"
    
    # Plot A: General Spatial Layout
    print("Plotting spatial layout...")
    plot_embeddings_rgb(
        embeddings, df.x, df.y, data_dir=data_dir,
        save_path=os.path.join(out_dir, f"random_spatial_rgb_{arch_tag}.pdf"), 
        show_plot=True
    )

    # Plot B: Oriented (Facing East/0 rads)
    print("Plotting oriented view...")
    plot_oriented_embeddings_rgb(
        embeddings, df.x, df.y, df.orientation, target_orientation=0, data_dir=data_dir,
        save_path=os.path.join(out_dir, f"random_oriented_0_rgb_{arch_tag}.pdf"), 
        show_plot=True
    )
    
    print(f"Analysis saved to: {out_dir} with tag {arch_tag}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze random network projections") 
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--n", type=int, help="Number of files to load")

    
    # Added arguments for arbitrary network sizes
    parser.add_argument("--hidden", type=int, default=128, help="Number of neurons in the hidden layer")
    parser.add_argument("--output", type=int, default=64, help="Dimension of the output embedding")
    
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        analyze_random_structure(args.data_dir, args.hidden, args.output, n_files=args.n)
    else:
        print("Directory not found.")