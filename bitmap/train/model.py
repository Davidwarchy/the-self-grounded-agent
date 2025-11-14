# train/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LidarEncoder(nn.Module):
    def __init__(self, input_dim=100, hidden_dims=[256, 128], embedding_dim=64):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

    @property
    def device(self):
        """Return the device of the first parameter (or cpu if no params)."""
        return next(self.parameters()).device