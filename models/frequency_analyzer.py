"""
Frequency analyzer using wavelet decomposition for forensic artifact detection
"""

import torch
import torch.nn as nn
import pywt
import numpy as np


class FrequencyAnalyzer(nn.Module):
    """
    Wavelet-based frequency analysis for detecting forensic artifacts
    
    Based on NVIDIA's approach: uses wavelet decomposition to identify
    frequency-domain artifacts that are characteristic of generative models.
    """
    
    def __init__(
        self,
        wavelet_type: str = 'db8',
        levels: int = 3,
        input_channels: int = 3
    ):
        super().__init__()
        
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.input_channels = input_channels
        
        # Calculate feature dimension from wavelet decomposition
        # Each level produces 3 subbands (LH, HL, HH) + 1 approximation (LL) at last level
        self.feature_dim = input_channels * (3 * levels + 1)
        
        # Learnable weights for different frequency bands
        self.band_weights = nn.Parameter(torch.ones(self.feature_dim))
        
        # Feature processing network
        self.feature_net = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
    def wavelet_decompose(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Perform wavelet decomposition on a single frame
        
        Args:
            frame: Single frame [C, H, W]
            
        Returns:
            coeffs: Wavelet coefficients [feature_dim]
        """
        frame_np = frame.cpu().numpy()
        
        features = []
        for c in range(self.input_channels):
            channel = frame_np[c]
            
            # Perform multi-level wavelet decomposition
            coeffs = pywt.wavedec2(channel, self.wavelet_type, level=self.levels)
            
            # Extract energy from each subband
            # LL (approximation) from last level
            ll_energy = np.sqrt(np.mean(coeffs[0] ** 2))
            features.append(ll_energy)
            
            # Detail coefficients (LH, HL, HH) from each level
            for level_coeffs in coeffs[1:]:
                for subband in level_coeffs:
                    energy = np.sqrt(np.mean(subband ** 2))
                    features.append(energy)
        
        return torch.tensor(features, dtype=frame.dtype, device=frame.device)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Analyze frequency content of video frames
        
        Args:
            video: Video frames [B, T, C, H, W]
            
        Returns:
            features: Frequency-domain features [B, 128]
        """
        batch_size, num_frames = video.shape[0], video.shape[1]
        
        # Process each frame in the video
        all_features = []
        for b in range(batch_size):
            frame_features = []
            # Sample a subset of frames for efficiency
            sample_indices = torch.linspace(
                0, num_frames - 1, min(8, num_frames), dtype=torch.long
            )
            
            for t in sample_indices:
                frame = video[b, t]  # [C, H, W]
                coeffs = self.wavelet_decompose(frame)
                frame_features.append(coeffs)
            
            # Average across sampled frames
            frame_features = torch.stack(frame_features)
            avg_features = frame_features.mean(dim=0)
            all_features.append(avg_features)
        
        features = torch.stack(all_features)  # [B, feature_dim]
        
        # Apply learnable band weights
        features = features * self.band_weights
        
        # Process through feature network
        features = self.feature_net(features)  # [B, 128]
        
        return features
