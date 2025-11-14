# train/plotting.py
import matplotlib.pyplot as plt
import numpy as np
from train.spatial import plot_embedding_distribution, plot_cluster_spatial_distribution

def plot_train_val_loss(train_losses, val_losses, best_val, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.axhline(best_val, color='green', linestyle='--', label=f"Best: {best_val:.4f}")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Loss plot saved: {save_path}")