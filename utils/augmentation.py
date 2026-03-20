"""
Data augmentation transforms for video and audio
"""

import numpy as np
import cv2
import torch
from typing import Optional
import random


class VideoAugmentation:
    """
    Video augmentation pipeline including compression emulation
    """
    
    def __init__(
        self,
        horizontal_flip: bool = True,
        color_jitter: bool = True,
        gaussian_blur: bool = True,
        compression_emulation: bool = True,
        compression_quality_range: tuple = (70, 95),
        mode: str = 'train'
    ):
        self.horizontal_flip = horizontal_flip and (mode == 'train')
        self.color_jitter = color_jitter and (mode == 'train')
        self.gaussian_blur = gaussian_blur and (mode == 'train')
        self.compression_emulation = compression_emulation and (mode == 'train')
        self.compression_quality_range = compression_quality_range
        self.mode = mode
        
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to video frames
        
        Args:
            frames: Video frames [T, C, H, W]
            
        Returns:
            frames: Augmented frames [T, C, H, W]
        """
        if self.mode != 'train':
            return frames
        
        # Horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            frames = frames[:, :, :, ::-1].copy()
        
        # Color jitter
        if self.color_jitter and random.random() > 0.5:
            frames = self._apply_color_jitter(frames)
        
        # Gaussian blur
        if self.gaussian_blur and random.random() > 0.5:
            frames = self._apply_gaussian_blur(frames)
        
        # Compression emulation (social media style)
        if self.compression_emulation and random.random() > 0.5:
            frames = self._apply_compression(frames)
        
        return frames
    
    def _apply_color_jitter(self, frames: np.ndarray) -> np.ndarray:
        """Apply random color jitter"""
        # Random brightness
        brightness_factor = random.uniform(0.8, 1.2)
        frames = np.clip(frames * brightness_factor, 0, 1)
        
        # Random contrast
        contrast_factor = random.uniform(0.8, 1.2)
        mean = frames.mean(axis=(2, 3), keepdims=True)
        frames = np.clip((frames - mean) * contrast_factor + mean, 0, 1)
        
        # Random saturation
        saturation_factor = random.uniform(0.8, 1.2)
        gray = frames.mean(axis=1, keepdims=True)
        frames = np.clip((frames - gray) * saturation_factor + gray, 0, 1)
        
        return frames
    
    def _apply_gaussian_blur(self, frames: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur"""
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.1, 2.0)
        
        blurred_frames = []
        for t in range(frames.shape[0]):
            frame = frames[t].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
            blurred_frames.append(frame.transpose(2, 0, 1))  # [H, W, C] -> [C, H, W]
        
        return np.stack(blurred_frames)
    
    def _apply_compression(self, frames: np.ndarray) -> np.ndarray:
        """
        Emulate social media compression (YouTube, Instagram, TikTok style)
        """
        quality = random.randint(*self.compression_quality_range)
        
        compressed_frames = []
        for t in range(frames.shape[0]):
            frame = frames[t].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            frame = (frame * 255).astype(np.uint8)
            
            # JPEG compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            frame = frame.astype(np.float32) / 255.0
            compressed_frames.append(frame.transpose(2, 0, 1))  # [H, W, C] -> [C, H, W]
        
        return np.stack(compressed_frames)


class AudioAugmentation:
    """
    Audio augmentation pipeline
    """
    
    def __init__(
        self,
        add_noise: bool = True,
        time_stretch: bool = True,
        pitch_shift: bool = False,
        mode: str = 'train'
    ):
        self.add_noise = add_noise and (mode == 'train')
        self.time_stretch = time_stretch and (mode == 'train')
        self.pitch_shift = pitch_shift and (mode == 'train')
        self.mode = mode
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to audio
        
        Args:
            audio: Audio waveform [audio_length]
            
        Returns:
            audio: Augmented audio [audio_length]
        """
        if self.mode != 'train':
            return audio
        
        # Add Gaussian noise
        if self.add_noise and random.random() > 0.5:
            noise_factor = random.uniform(0.001, 0.01)
            noise = np.random.randn(len(audio)) * noise_factor
            audio = audio + noise
            audio = np.clip(audio, -1.0, 1.0)
        
        # Time stretching (not implemented to avoid librosa dependency in augmentation)
        # Can be added if needed
        
        return audio


def get_transforms(config: dict, mode: str = 'train'):
    """
    Create augmentation transforms based on config
    
    Args:
        config: Configuration dictionary
        mode: 'train', 'val', or 'test'
        
    Returns:
        video_transform, audio_transform
    """
    aug_config = config['training']['augmentation']
    
    video_transform = VideoAugmentation(
        horizontal_flip=aug_config['horizontal_flip'],
        color_jitter=aug_config['color_jitter'],
        gaussian_blur=aug_config['gaussian_blur'],
        compression_emulation=aug_config['compression_emulation'],
        compression_quality_range=tuple(aug_config['compression_quality']),
        mode=mode
    )
    
    audio_transform = AudioAugmentation(
        add_noise=True,
        time_stretch=False,
        pitch_shift=False,
        mode=mode
    )
    
    return video_transform, audio_transform
