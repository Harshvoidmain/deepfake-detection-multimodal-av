"""
Main model architecture for Multimodal Audio-Visual Deepfake Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.visual_encoder import VisualEncoder
from models.audio_encoder import AudioEncoder
from models.temporal_encoder import TemporalEncoder
from models.frequency_analyzer import FrequencyAnalyzer
from models.fusion_module import FusionModule
from models.classifier import Classifier


class MultimodalAVDetector(nn.Module):
    """
    Multimodal Audio-Visual Deepfake Detector
    
    Architecture:
    1. Visual Encoder: DINOv2 for spatial features
    2. Temporal Encoder: Video Swin Transformer for temporal modeling
    3. Audio Encoder: wav2vec2 for audio features
    4. Frequency Analyzer: Wavelet decomposition for forensic artifacts
    5. Fusion Module: Cross-attention fusion of audio-visual features
    6. Classifier: Final classification head
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Visual branch
        self.visual_encoder = VisualEncoder(
            backbone=config['model']['visual']['backbone'],
            pretrained=config['model']['visual']['pretrained'],
            freeze_backbone=config['model']['visual']['freeze_backbone'],
            embed_dim=config['model']['visual']['embed_dim']
        )
        
        # Temporal modeling
        self.temporal_encoder = TemporalEncoder(
            embed_dim=config['model']['visual']['embed_dim'],
            num_layers=config['model']['temporal']['num_layers'],
            num_heads=config['model']['temporal']['num_heads'],
            dropout=config['model']['temporal']['dropout']
        )
        
        # Audio branch
        self.audio_encoder = AudioEncoder(
            backbone=config['model']['audio']['backbone'],
            pretrained=config['model']['audio']['pretrained'],
            freeze_backbone=config['model']['audio']['freeze_backbone'],
            embed_dim=config['model']['audio']['embed_dim']
        )
        
        # Frequency analysis
        self.frequency_analyzer = FrequencyAnalyzer(
            wavelet_type=config['model']['frequency']['wavelet_type'],
            levels=config['model']['frequency']['levels']
        )
        
        # Fusion module
        self.fusion = FusionModule(
            visual_dim=config['model']['visual']['embed_dim'],
            audio_dim=config['model']['audio']['embed_dim'],
            hidden_dim=config['model']['fusion']['hidden_dim'],
            num_heads=config['model']['fusion']['num_heads'],
            dropout=config['model']['fusion']['dropout']
        )
        
        # Classifier
        self.classifier = Classifier(
            input_dim=config['model']['fusion']['hidden_dim'],
            hidden_dims=config['model']['classifier']['hidden_dims'],
            num_classes=config['model']['classifier']['num_classes'],
            dropout=config['model']['classifier']['dropout']
        )
        
    def forward(
        self, 
        video: torch.Tensor,
        audio: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            video: Video frames [B, T, C, H, W]
            audio: Audio waveform [B, audio_length]
            return_features: If True, return intermediate features for XAI
            
        Returns:
            Dictionary containing:
                - logits: Classification logits [B, num_classes]
                - features: Intermediate features (if return_features=True)
        """
        batch_size, num_frames = video.shape[0], video.shape[1]
        
        # Extract visual features
        # Reshape: [B, T, C, H, W] -> [B*T, C, H, W]
        video_flat = video.view(-1, *video.shape[2:])
        visual_features = self.visual_encoder(video_flat)  # [B*T, embed_dim]
        
        # Reshape back: [B*T, embed_dim] -> [B, T, embed_dim]
        visual_features = visual_features.view(batch_size, num_frames, -1)
        
        # Apply temporal encoding
        temporal_features = self.temporal_encoder(visual_features)  # [B, embed_dim]
        
        # Extract audio features
        audio_features = self.audio_encoder(audio)  # [B, embed_dim]
        
        # Frequency analysis on video for forensic artifacts
        freq_features = self.frequency_analyzer(video)  # [B, freq_dim]
        
        # Fuse multimodal features
        fused_features = self.fusion(
            visual=temporal_features,
            audio=audio_features,
            frequency=freq_features
        )  # [B, hidden_dim]
        
        # Classification
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        output = {'logits': logits}
        
        if return_features:
            output['features'] = {
                'visual': visual_features,
                'temporal': temporal_features,
                'audio': audio_features,
                'frequency': freq_features,
                'fused': fused_features
            }
            
        return output
    
    def predict(self, video: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions
        
        Returns:
            predictions: Class predictions [B]
            probabilities: Class probabilities [B, num_classes]
        """
        with torch.no_grad():
            output = self.forward(video, audio)
            logits = output['logits']
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        return predictions, probabilities
