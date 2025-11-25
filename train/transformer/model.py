import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Create constant positional encoding matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        return x + self.pe[:, :x.size(1)]

class TemporalLidarTransformer(nn.Module):
    def __init__(self, num_rays=100, n_timesteps=5, d_model=128, nhead=4, 
                 num_layers=3, dim_feedforward=256, embedding_dim=64, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Input Projection: Map raw 100 rays -> d_model latent size
        self.input_proj = nn.Linear(num_rays, d_model)
        
        # 2. Positional Encoding (Temporal)
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_timesteps + 5)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True, # Critical: inputs are (Batch, Seq, Feature)
            norm_first=True   # Pre-LN usually converges faster
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Projection Head
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embedding_dim)
        )
        
        # Special token for aggregation (optional, but we'll use pooling/last token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: (Batch, n_timesteps, num_rays)
        """
        B, T, R = x.shape
        
        # Project input features
        x = self.input_proj(x) # (B, T, d_model)
        
        # Add CLS token to the beginning (standard BERT-like practice)
        # This allows the model to learn a specific vector for "summary"
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, T+1, d_model)
        
        # Add temporal position encoding
        x = self.pos_encoder(x)
        
        # Transform
        out = self.transformer_encoder(x) # (B, T+1, d_model)
        
        # Extract the CLS token (index 0) as the sequence representation
        # Alternatively, we could take the last token (index -1)
        sequence_embedding = out[:, 0, :] 
        
        # Project to final embedding space
        final_emb = self.head(sequence_embedding)
        
        return F.normalize(final_emb, p=2, dim=1)

    @property
    def device(self):
        return next(self.parameters()).device