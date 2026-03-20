"""
Dataset class for deepfake video detection
"""

import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import librosa
from typing import Dict, Tuple, Optional
import json


class DeepfakeDataset(Dataset):
    """
    Dataset for loading video and audio data for deepfake detection
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        num_frames: int = 16,
        frame_size: int = 224,
        audio_sample_rate: int = 16000,
        clip_duration: float = 2.0,
        transform=None,
        audio_transform=None,
        mode: str = 'train'
    ):
        """
        Args:
            data_dir: Root directory containing videos
            metadata_file: JSON file with video paths and labels
            num_frames: Number of frames to sample from each video
            frame_size: Size to resize frames to
            audio_sample_rate: Sample rate for audio
            clip_duration: Duration of audio clip in seconds
            transform: Video augmentation pipeline
            audio_transform: Audio augmentation pipeline
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.audio_sample_rate = audio_sample_rate
        self.clip_duration = clip_duration
        self.transform = transform
        self.audio_transform = audio_transform
        self.mode = mode
        self._audio_load_warning_count = 0
        self._audio_load_warning_limit = 3
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata[mode]
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video sample
        
        Returns:
            Dictionary containing:
                - video: Video frames [T, C, H, W]
                - audio: Audio waveform [audio_length]
                - label: Binary label (0=real, 1=fake)
                - video_path: Path to video file
        """
        sample = self.samples[idx]
        video_path = os.path.join(self.data_dir, sample['path'])
        label = sample['label']  # 0=real, 1=fake
        
        # Load video and audio
        frames = self._load_video(video_path)
        audio = self._load_audio(video_path)
        
        # Apply transformations
        if self.transform is not None:
            frames = self.transform(frames)
        
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
        
        # Convert to tensors
        frames = torch.from_numpy(frames).float()
        audio = torch.from_numpy(audio).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'video': frames,
            'audio': audio,
            'label': label,
            'video_path': video_path
        }
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """
        Load and preprocess video frames
        
        Returns:
            frames: Array of shape [T, H, W, C]
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frame indices uniformly
        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
            # Pad if necessary
            while len(frame_indices) < self.num_frames:
                frame_indices.append(frame_indices[-1])
        else:
            frame_indices = np.linspace(
                0, total_frames - 1, self.num_frames, dtype=int
            )
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                # Use last valid frame if read fails
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.float32))
        
        cap.release()
        
        frames = np.stack(frames)  # [T, H, W, C]
        frames = frames.transpose(0, 3, 1, 2)  # [T, C, H, W]
        
        return frames
    
    def _load_audio(self, video_path: str) -> np.ndarray:
        """
        Load and preprocess audio
        
        Returns:
            audio: Waveform array [audio_length]
        """
        try:
            # Load audio using librosa
            audio, sr = librosa.load(
                video_path,
                sr=self.audio_sample_rate,
                duration=self.clip_duration,
                mono=True
            )
            
            # Pad or trim to exact length
            target_length = int(self.audio_sample_rate * self.clip_duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
                
        except Exception:
            if self._audio_load_warning_count < self._audio_load_warning_limit:
                print(f"Warning: Audio track unavailable/unsupported for some videos; using silent fallback. mode={self.mode}")
                self._audio_load_warning_count += 1
            # Return silent audio
            target_length = int(self.audio_sample_rate * self.clip_duration)
            audio = np.zeros(target_length, dtype=np.float32)
        
        return audio


def create_metadata_json(data_dir: str, output_file: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Helper function to create metadata JSON from directory structure
    
    Expected structure:
        data_dir/
            real/
                video1.mp4
                video2.mp4
                ...
            fake/
                video1.mp4
                video2.mp4
                ...
    """
    import random
    
    metadata = {'train': [], 'val': [], 'test': []}
    
    # Collect real videos
    real_dir = os.path.join(data_dir, 'real')
    if os.path.exists(real_dir):
        for video_file in os.listdir(real_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                metadata['train'].append({
                    'path': os.path.join('real', video_file),
                    'label': 0
                })
    
    # Collect fake videos
    fake_dir = os.path.join(data_dir, 'fake')
    if os.path.exists(fake_dir):
        for video_file in os.listdir(fake_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                metadata['train'].append({
                    'path': os.path.join('fake', video_file),
                    'label': 1
                })
    
    # Shuffle and split
    random.shuffle(metadata['train'])
    total = len(metadata['train'])
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    metadata['val'] = metadata['train'][train_end:val_end]
    metadata['test'] = metadata['train'][val_end:]
    metadata['train'] = metadata['train'][:train_end]
    
    # Save metadata
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata: {len(metadata['train'])} train, {len(metadata['val'])} val, {len(metadata['test'])} test")
