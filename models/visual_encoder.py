"""
Visual encoder using DINOv2 pretrained models
"""

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """
    Visual feature extractor using DINOv2
    
    DINOv2 is a self-supervised Vision Transformer that provides strong
    visual representations without task-specific fine-tuning.
    """
    
    def __init__(
        self, 
        backbone: str = 'dinov2_vits14',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        embed_dim: int = 384
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        
        # Load DINOv2 model
        if pretrained:
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', backbone)
            except Exception as e:
                print(f"Warning: Could not load pretrained {backbone}. Error: {e}")
                print("Creating random initialized model instead.")
                self.backbone = self._create_random_backbone()
        else:
            self.backbone = self._create_random_backbone()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head to normalize feature dimensions
        self.projection = nn.Linear(self._get_backbone_dim(), embed_dim)
        
    def _create_random_backbone(self):
        """Create a simple ViT as fallback"""
        from timm import create_model
        model = create_model('vit_small_patch14_dinov2', pretrained=False)
        return model
    
    def _get_backbone_dim(self) -> int:
        """Get output dimension of backbone"""
        dim_map = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536
        }
        return dim_map.get(self.backbone_name, 384)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            features: Visual embeddings [B, embed_dim]
        """
        # Extract features from backbone
        with torch.set_grad_enabled(self.training):
            features = self.backbone(x)
        
        # Project to target dimension
        features = self.projection(features)
        
        return features
