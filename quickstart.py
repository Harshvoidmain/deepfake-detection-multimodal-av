"""
Quick start script to verify installation and test the model
"""

import torch
import sys

print("="*60)
print("DEEPFAKE DETECTION - INSTALLATION CHECK")
print("="*60)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check PyTorch
try:
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available(): 
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")

# Check other packages
packages_to_check = [
    'cv2',
    'numpy',
    'librosa',
    'transformers',
    'pywt',
    'yaml',
    'timm'
]

print(f"\n{'Package':<20} {'Status':<10}")
print("-"*30)

for package in packages_to_check:
    try:
        __import__(package)
        print(f"{package:<20} ✓")
    except ImportError:
        print(f"{package:<20} ✗ MISSING")

# Check FFmpeg
import subprocess
try:
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, 
                          text=True,
                          timeout=5)
    if result.returncode == 0:
        print(f"{'ffmpeg':<20} ✓")
    else:
        print(f"{'ffmpeg':<20} ✗ NOT WORKING")
except Exception as e:
    print(f"{'ffmpeg':<20} ✗ NOT FOUND")

# Test model initialization
print("\n" + "="*60)
print("TESTING MODEL INITIALIZATION")
print("="*60)

try:
    from models import MultimodalAVDetector
    from utils import load_config
    
    print("\n✓ Importing modules successful")
    
    # Load config
    config = load_config('configs/config.yaml')
    print("✓ Configuration loaded")
    
    # Initialize model
    model = MultimodalAVDetector(config)
    print("✓ Model initialized")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with dummy data
    print("\n✓ Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy inputs
    batch_size = 2
    num_frames = config['data']['num_frames']
    frame_size = config['data']['frame_size']
    audio_length = int(config['data']['audio_sample_rate'] * config['data']['clip_duration'])
    
    dummy_video = torch.randn(batch_size, num_frames, 3, frame_size, frame_size).to(device)
    dummy_audio = torch.randn(batch_size, audio_length).to(device)
    
    with torch.no_grad():
        output = model(dummy_video, dummy_audio)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output['logits'].shape}")
    print(f"  Output device: {output['logits'].device}")
    
    print("\n" + "="*60)
    print("✓ ALL CHECKS PASSED!")
    print("="*60)
    print("\nYour environment is ready for training and inference!")
    print("\nNext steps:")
    print("1. Prepare your dataset in data/raw/")
    print("2. Run: python -c \"from utils import create_metadata_json; create_metadata_json('data/raw', 'data/processed/metadata.json')\"")
    print("3. Start training: python train.py --config configs/config.yaml")
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    print("\nPlease check the error message and ensure all dependencies are installed.")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
