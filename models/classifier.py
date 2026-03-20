"""
Classification head for deepfake detection
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    Multi-layer classification head with dropout
    
    Takes fused multimodal features and predicts real vs fake.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: list = [256, 128],
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify features
        
        Args:
            x: Input features [B, input_dim]
            
        Returns:
            logits: Class logits [B, num_classes]
        """
        return self.classifier(x)
