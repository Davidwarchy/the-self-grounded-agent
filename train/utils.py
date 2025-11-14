# train/utils.py
import os
import json
from datetime import datetime
import torch

def create_output_dir(base_dir="output"):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir = os.path.join(base_dir, f"train_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Training output: {run_dir}")
    return run_dir

def save_run_info(run_dir, config, train_size, val_size, split_idx, best_val_loss, start_time, end_time):
    info = {
        "timestamp": datetime.now().isoformat(),
        "duration_sec": (end_time - start_time).total_seconds(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset": {"n_files": config.n_files, "train": train_size, "val": val_size, "split_idx": split_idx},
        "model": {"hidden_dims": config.hidden_dims, "embedding_dim": config.embedding_dim},
        "training": {"batch_size": config.batch_size, "epochs": config.num_epochs, "lr": config.learning_rate},
        "loss": {"margin": config.margin},
        "best_val_loss": best_val_loss
    }
    path = os.path.join(run_dir, "run_info.json")
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"[INFO] Run info saved: {path}")