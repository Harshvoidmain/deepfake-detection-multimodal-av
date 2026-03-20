"""
Audio encoder using wav2vec2
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config


class AudioEncoder(nn.Module):
    """
    Audio feature extractor using wav2vec2
    
    wav2vec2 is a self-supervised model for speech representation learning
    that can capture audio artifacts in generated content.
    """
    
    def __init__(
        self,
        backbone: str = 'wav2vec2',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        embed_dim: int = 768
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Load wav2vec2 model
        if pretrained:
            try:
                self.backbone = Wav2Vec2Model.from_pretrained(
                    'facebook/wav2vec2-base-960h'
                )
            except Exception as e:
                print(f"Warning: Could not load pretrained wav2vec2. Error: {e}")
                print("Creating random initialized model instead.")
                config = Wav2Vec2Config()
                self.backbone = Wav2Vec2Model(config)
        else:
            config = Wav2Vec2Config()
            self.backbone = Wav2Vec2Model(config)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output dimension
        backbone_dim = self.backbone.config.hidden_size
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head
        self.projection = nn.Linear(backbone_dim, embed_dim)
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features
        
        Args:
            audio: Audio waveform [B, audio_length]
            
        Returns:
            features: Audio embeddings [B, embed_dim]
        """
        # Extract features from wav2vec2
        with torch.set_grad_enabled(self.training):
            outputs = self.backbone(audio)
            features = outputs.last_hidden_state  # [B, T, hidden_size]
        
        # Temporal pooling: [B, T, hidden_size] -> [B, hidden_size]
        features = features.transpose(1, 2)  # [B, hidden_size, T]
        features = self.temporal_pool(features).squeeze(-1)  # [B, hidden_size]
        
        # Project to target dimension
        features = self.projection(features)  # [B, embed_dim]
        
        return features
