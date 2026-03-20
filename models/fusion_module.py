"""
Fusion module for combining multimodal features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """
    Multimodal fusion using cross-attention mechanism
    
    Combines visual, audio, and frequency features through cross-attention
    to learn complementary information from different modalities.
    """
    
    def __init__(
        self,
        visual_dim: int = 384,
        audio_dim: int = 768,
        freq_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.15
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project all modalities to common dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)
        
        # Cross-attention between visual and audio
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for all modalities
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Modality-specific attention weights
        self.modality_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        frequency: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multimodal features
        
        Args:
            visual: Visual features [B, visual_dim]
            audio: Audio features [B, audio_dim]
            frequency: Frequency features [B, freq_dim]
            
        Returns:
            fused: Fused features [B, hidden_dim]
        """
        # Project to common dimension
        v = self.visual_proj(visual)  # [B, hidden_dim]
        a = self.audio_proj(audio)    # [B, hidden_dim]
        f = self.freq_proj(frequency)  # [B, hidden_dim]
        
        # Add sequence dimension for attention
        v = v.unsqueeze(1)  # [B, 1, hidden_dim]
        a = a.unsqueeze(1)  # [B, 1, hidden_dim]
        f = f.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Cross-attention: visual attends to audio
        v_attn, _ = self.cross_attention(v, a, a)
        v = self.norm1(v + v_attn)
        
        # Concatenate all modalities
        all_features = torch.cat([v, a, f], dim=1)  # [B, 3, hidden_dim]
        
        # Self-attention across all modalities
        attn_out, _ = self.self_attention(all_features, all_features, all_features)
        all_features = self.norm2(all_features + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(all_features)
        all_features = self.norm3(all_features + ffn_out)
        
        # Compute modality-specific attention weights
        all_features_flat = all_features.view(all_features.size(0), -1)  # [B, 3*hidden_dim]
        modality_weights = self.modality_attention(all_features_flat)  # [B, 3]
        
        # Weighted sum of modalities
        modality_weights = modality_weights.unsqueeze(-1)  # [B, 3, 1]
        fused = (all_features * modality_weights).sum(dim=1)  # [B, hidden_dim]
        
        return fused
