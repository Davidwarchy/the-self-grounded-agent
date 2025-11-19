import os, sys, json, torch, argparse, pandas as pd
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_parquet import load_data
from train.model import LidarEncoder
from analysis.plotting import plot_embeddings_rgb, plot_oriented_embeddings_rgb

def analyze_random_structure(data_dir, n_files=1000):
    # 1. Load Metadata & Data
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)

    print("Loaded metadata:", meta)
    
    num_rays = meta.get("environment_parameters", {}).get("num_rays", 100)
    df_path = load_data(data_dir, n=n_files)
    df = pd.read_parquet(df_path)

    print(f"Loaded data with {len(df)} samples.")
    
    # 2. Random Projection (Untrained Network)
    # We initialize the model but skip loading weights to test the "Random Structure" theory
    lidar_tensor = torch.tensor(df[[f'ray_{i}' for i in range(num_rays)]].values / 200.0).float()
    model = LidarEncoder(input_dim=num_rays, embedding_dim=64).eval()
    
    print(f"Generating random embeddings for {len(df)} samples...")
    with torch.no_grad():
        embeddings = model(lidar_tensor).numpy()

    # 3. Visualize
    out_dir = os.path.join(data_dir, "analysis_random_structure")
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot A: General Spatial Layout
    plot_embeddings_rgb(
        embeddings, df.x, df.y, data_dir=data_dir,
        save_path=os.path.join(out_dir, "random_spatial_rgb.pdf"), show_plot=True
    )

    # Plot B: Oriented (Facing East/0 rads)
    plot_oriented_embeddings_rgb(
        embeddings, df.x, df.y, df.orientation, target_orientation=0, data_dir=data_dir,
        save_path=os.path.join(out_dir, "random_oriented_0_rgb.pdf"), show_plot=True
    )
    
    print(f"Analysis saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        analyze_random_structure(args.data_dir)
    else:
        print("Directory not found.")

    # python .\analysis\random_net.py "output/2025-11-14-111925_random_walk_100k" 