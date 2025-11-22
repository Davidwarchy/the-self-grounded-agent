# train/temporal_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class TemporalLidarEncoderCNN(nn.Module):
    """
    CNN-based encoder for temporal LiDAR sequences.
    Treats n_timesteps as channels and applies 1D convolutions across the ray dimension.
    """
    def __init__(self, input_dim=100, n_timesteps=5, embedding_dim=64):
        super().__init__()
        
        # First conv block: extract features from temporal patterns
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_timesteps, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 -> 50
            nn.Dropout(0.1)
        )
        
        # Second conv block: learn higher-level features
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 50 -> 25
            nn.Dropout(0.1)
        )
        
        # Third conv block: compress features
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling -> (batch, 256, 1)
        )
        
        # Fully connected layers for final embedding
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_timesteps, input_dim) - temporal sequence of LiDAR scans
        Returns:
            normalized embedding: (batch, embedding_dim)
        """
        # x: (batch, n_timesteps, input_dim)
        # Conv1d expects (batch, channels, length)
        # Here: channels = n_timesteps, length = input_dim (ray dimension)
        
        x = self.conv1(x)   # (batch, 64, 50)
        x = self.conv2(x)   # (batch, 128, 25)
        x = self.conv3(x)   # (batch, 256, 1)
        
        x = x.squeeze(-1)   # (batch, 256)
        x = self.fc(x)      # (batch, embedding_dim)
        
        # L2 normalization for contrastive learning
        return F.normalize(x, p=2, dim=1)
    
    @property
    def device(self):
        """Return the device of the first parameter."""
        return next(self.parameters()).device


class TemporalLidarDataset(Dataset):
    """
    Dataset for temporal sequences of LiDAR scans.
    Each sample contains n_timesteps consecutive scans.
    """
    def __init__(self, df, num_rays=100, n_timesteps=5, start_idx=0, end_idx=None):
        """
        Args:
            df: DataFrame with LiDAR data (already loaded)
            num_rays: Number of LiDAR rays per scan
            n_timesteps: Number of consecutive timesteps to use
            start_idx: Start index for this split
            end_idx: End index for this split
        """
        self.num_rays = num_rays
        self.n_timesteps = n_timesteps
        
        # Extract LiDAR data
        ray_cols = [f'ray_{i}' for i in range(num_rays)]
        self.lidar = (df[ray_cols].values.astype(np.float32) / 200.0)
        
        # Extract position/orientation
        self.x = df['x'].values
        self.y = df['y'].values
        self.theta = df['orientation'].values
        
        # Set valid range
        if end_idx is None:
            end_idx = len(df)
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        # Length accounts for needing n_timesteps consecutive frames
        self.length = max(0, end_idx - start_idx - n_timesteps)
        self.total_samples = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            anchor: (n_timesteps, num_rays) - sequence of scans at time t
            positive: (n_timesteps, num_rays) - sequence starting at t+1
            negative: (n_timesteps, num_rays) - random sequence
            x, y, theta: position/orientation of the LAST frame in anchor
        """
        # Actual index in the full dataset
        actual_idx = self.start_idx + idx
        
        # Anchor: frames [t, t+1, ..., t+n-1]
        anchor_seq = self.lidar[actual_idx:actual_idx + self.n_timesteps]
        
        # Positive: frames [t+1, t+2, ..., t+n] (shifted by 1)
        positive_seq = self.lidar[actual_idx + 1:actual_idx + self.n_timesteps + 1]
        
        # Negative: random sequence from anywhere in dataset
        # Make sure we have enough room for n_timesteps
        max_neg_start = self.total_samples - self.n_timesteps
        neg_idx = np.random.randint(0, max_neg_start)
        negative_seq = self.lidar[neg_idx:neg_idx + self.n_timesteps]
        
        # Convert to tensors
        anchor = torch.from_numpy(anchor_seq)
        positive = torch.from_numpy(positive_seq)
        negative = torch.from_numpy(negative_seq)
        
        # Use the LAST frame's position for spatial tracking
        last_frame_idx = actual_idx + self.n_timesteps - 1
        
        return (anchor, positive, negative,
                self.x[last_frame_idx],
                self.y[last_frame_idx],
                self.theta[last_frame_idx])
    
    @property
    def valid_slice(self):
        """
        Returns a dict with arrays for the valid portion of this dataset.
        Uses the LAST frame of each sequence for position/orientation.
        """
        # Start from start_idx + (n_timesteps - 1) to align with __getitem__
        start = self.start_idx + self.n_timesteps - 1
        end = start + len(self)
        
        s = slice(start, end)
        return {
            'x': self.x[s],
            'y': self.y[s],
            'theta': self.theta[s],
            'lidar': self.lidar[s]
        }


# Example usage and comparison
if __name__ == "__main__":
    print("=== Temporal CNN LiDAR Encoder ===\n")
    
    # Model comparison
    print("Input/Output Shapes:")
    print("-" * 50)
    
    batch_size = 32
    n_timesteps = 5
    num_rays = 100
    embedding_dim = 64
    
    # Create model
    model = TemporalLidarEncoderCNN(
        input_dim=num_rays,
        n_timesteps=n_timesteps,
        embedding_dim=embedding_dim
    )
    
    # Test input
    x = torch.randn(batch_size, n_timesteps, num_rays)
    print(f"Input shape:  {x.shape}")
    print(f"              (batch={batch_size}, timesteps={n_timesteps}, rays={num_rays})")
    
    # Forward pass
    embedding = model(x)
    print(f"\nOutput shape: {embedding.shape}")
    print(f"              (batch={batch_size}, embedding_dim={embedding_dim})")
    
    # Verify normalization
    norms = torch.norm(embedding, dim=1)
    print(f"\nEmbedding norms (should be ~1.0):")
    print(f"  Mean: {norms.mean().item():.4f}")
    print(f"  Std:  {norms.std().item():.4f}")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n" + "="*50)
    print("Benefits of Temporal CNN:")
    print("  ✓ Captures motion patterns across time")
    print("  ✓ Local temporal feature extraction")
    print("  ✓ Translation invariant across rays")
    print("  ✓ Efficient parallel processing")
    print("  ✓ Fewer parameters than flatten approach")