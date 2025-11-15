# bitmap/train/run_train.py
import argparse
from train import TrainingConfig, train
from train.config import TrainingConfig
from dataclasses import fields
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train LiDAR Embedding Model with Contrastive Loss")

    # Get all fields from TrainingConfig to auto-generate args
    config_fields = {f.name: f for f in fields(TrainingConfig)}

    # Data
    parser.add_argument("--data_dir", type=str, default=config_fields['data_dir'].default,
                        help="Directory containing log_*.csv files and metadata.json")
    parser.add_argument("--n_files", type=int, default=config_fields['n_files'].default,
                        help="Number of log files to load")
    parser.add_argument("--num_rays", type=int, default=config_fields['num_rays'].default,
                        help="Number of LiDAR rays per scan")

    # Model
    parser.add_argument("--hidden_dims", type=str, default=None,
                        help='Comma-separated hidden layer sizes, e.g., "256,128"')
    parser.add_argument("--embedding_dim", type=int, default=config_fields['embedding_dim'].default,
                        help="Dimension of output embedding")

    # Training
    parser.add_argument("--batch_size", type=int, default=config_fields['batch_size'].default)
    parser.add_argument("--num_epochs", type=int, default=config_fields['num_epochs'].default)
    parser.add_argument("--learning_rate", "--lr", type=float, default=config_fields['learning_rate'].default)
    parser.add_argument("--margin", type=float, default=config_fields['margin'].default)
    parser.add_argument("--vis_interval", type=int, default=config_fields['vis_interval'].default,
                        help="Save visualizations every N epochs")

    # Evaluation (optional overrides)
    parser.add_argument("--distance_threshold", type=float, default=config_fields['distance_threshold'].default)
    parser.add_argument("--temporal_window", type=int, default=config_fields['temporal_window'].default)
    parser.add_argument("--far_threshold", type=int, default=config_fields['far_threshold'].default)

    args = parser.parse_args()

    # Parse hidden_dims if provided
    if args.hidden_dims is not None:
        try:
            args.hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
        except:
            raise ValueError("hidden_dims must be comma-separated integers, e.g., 256,128")

    return args


def main():
    args = parse_args()

    # Create config from defaults + CLI overrides
    config = TrainingConfig(
        data_dir=args.data_dir,
        n_files=args.n_files,
        num_rays=args.num_rays,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        margin=args.margin,
        vis_interval=args.vis_interval,
        distance_threshold=args.distance_threshold,
        temporal_window=args.temporal_window,
        far_threshold=args.far_threshold,
    )

    print("[CONFIG]")
    print(json.dumps(config.__dict__, indent=2))

    train(config)


if __name__ == "__main__":
    main()

    # --------------------------------------------------------------
    # python -m train.run_train --num_epochs 20 --vis_interval 1 --batch_size 64 --data_dir output/2025-11-14-213949_random_walk_10k
    # --------------------------------------------------------------