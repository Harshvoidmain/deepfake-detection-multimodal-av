"""
Temporal encoder for video sequences
"""

import torch
import torch.nn as nn
import math


class TemporalEncoder(nn.Module):
    """
    Temporal modeling using Transformer encoder
    
    Processes sequence of frame features to capture temporal dynamics
    and motion patterns in videos.
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Positional encoding for temporal information
        self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal sequence
        
        Args:
            x: Frame features [B, T, embed_dim]
            
        Returns:
            features: Temporally encoded features [B, embed_dim]
        """
        # Add positional encoding
        x = self.pos_encoding(x)  # [B, T, embed_dim]
        
        # Apply transformer
        x = self.transformer(x)  # [B, T, embed_dim]
        
        # Temporal pooling to get single representation
        x = x.transpose(1, 2)  # [B, embed_dim, T]
        x = self.temporal_pool(x).squeeze(-1)  # [B, embed_dim]
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding
        
        Args:
            x: Input tensor [B, T, d_model]
            
        Returns:
            x: Tensor with positional encoding added [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
