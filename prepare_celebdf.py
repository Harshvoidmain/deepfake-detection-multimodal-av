"""
Helper script to organize Celeb-DF v2 dataset for training

Celeb-DF v2 structure:
    Celeb-DF-v2/
    ├── Celeb-real/       (590 real celebrity videos)
    ├── YouTube-real/     (300 real YouTube videos)  
    ├── Celeb-synthesis/  (5,639 deepfake videos)
    └── List_of_testing_videos.txt

This script will:
1. Copy/symlink videos to data/raw/real/ and data/raw/fake/
2. Create metadata.json for training
3. Perform train/val/test split
"""

import os
import shutil
import json
import random
from pathlib import Path
from tqdm import tqdm


def organize_celebdf(
    celebdf_root: str,
    output_dir: str = "data/raw",
    use_symlinks: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    Organize Celeb-DF v2 dataset into training structure
    
    Args:
        celebdf_root: Path to downloaded Celeb-DF-v2 directory
        output_dir: Output directory (default: data/raw)
        use_symlinks: Use symlinks instead of copying (saves disk space)
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
    """
    celebdf_root = Path(celebdf_root)
    output_dir = Path(output_dir)
    
    # Create output directories
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("ORGANIZING CELEB-DF V2 DATASET")
    print("="*60)
    
    # Process real videos
    print("\n📁 Processing real videos...")
    real_sources = [
        celebdf_root / "Celeb-real",
        celebdf_root / "YouTube-real"
    ]
    
    real_count = 0
    for source_dir in real_sources:
        if not source_dir.exists():
            print(f"Warning: {source_dir} not found, skipping...")
            continue
            
        videos = list(source_dir.glob("*.mp4"))
        print(f"  Found {len(videos)} videos in {source_dir.name}")
        
        for video in tqdm(videos, desc=f"  Copying {source_dir.name}"):
            dest = real_dir / f"{source_dir.name}_{video.name}"
            
            if use_symlinks:
                if dest.exists():
                    dest.unlink()
                dest.symlink_to(video.absolute())
            else:
                shutil.copy2(video, dest)
            
            real_count += 1
    
    print(f"✓ Processed {real_count} real videos")
    
    # Process fake videos
    print("\n📁 Processing fake videos...")
    fake_source = celebdf_root / "Celeb-synthesis"
    
    if not fake_source.exists():
        print(f"Error: {fake_source} not found!")
        return
    
    fake_videos = list(fake_source.glob("*.mp4"))
    print(f"  Found {len(fake_videos)} fake videos")
    
    fake_count = 0
    for video in tqdm(fake_videos, desc="  Copying fake videos"):
        dest = fake_dir / video.name
        
        if use_symlinks:
            if dest.exists():
                dest.unlink()
            dest.symlink_to(video.absolute())
        else:
            shutil.copy2(video, dest)
        
        fake_count += 1
    
    print(f"✓ Processed {fake_count} fake videos")
    
    # Create metadata
    print("\n📊 Creating metadata...")
    metadata = create_metadata(real_dir, fake_dir, train_ratio, val_ratio)
    
    # Save metadata
    metadata_dir = Path("data/processed")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Real videos:  {real_count}")
    print(f"Fake videos:  {fake_count}")
    print(f"Total videos: {real_count + fake_count}")
    print(f"\nSplit:")
    print(f"  Train: {len(metadata['train'])} ({len(metadata['train'])/(real_count+fake_count)*100:.1f}%)")
    print(f"  Val:   {len(metadata['val'])} ({len(metadata['val'])/(real_count+fake_count)*100:.1f}%)")
    print(f"  Test:  {len(metadata['test'])} ({len(metadata['test'])/(real_count+fake_count)*100:.1f}%)")
    print("="*60)
    
    print(f"\n✅ Dataset ready! You can now run:")
    print(f"   python train.py --config configs/config.yaml")


def create_metadata(real_dir, fake_dir, train_ratio=0.7, val_ratio=0.15):
    """Create metadata JSON with train/val/test split"""
    
    samples = []
    
    # Add real videos
    for video in real_dir.glob("*.mp4"):
        samples.append({
            'path': str(Path('real') / video.name),
            'label': 0
        })
    
    # Add fake videos
    for video in fake_dir.glob("*.mp4"):
        samples.append({
            'path': str(Path('fake') / video.name),
            'label': 1
        })
    
    # Shuffle
    random.shuffle(samples)
    
    # Split
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    metadata = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }
    
    return metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize Celeb-DF v2 dataset')
    parser.add_argument('celebdf_path', type=str,
                       help='Path to Celeb-DF-v2 directory')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory (default: data/raw)')
    parser.add_argument('--symlink', action='store_true',
                       help='Use symlinks instead of copying (saves disk space)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    organize_celebdf(
        args.celebdf_path,
        args.output,
        args.symlink,
        args.train_ratio,
        args.val_ratio
    )
