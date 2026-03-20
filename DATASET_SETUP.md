# Celeb-DF v2 Dataset Preparation

## After Download Completes:

### 1. Extract the Dataset
```bash
# Extract your downloaded file to a location, e.g.:
# C:/datasets/Celeb-DF-v2/
```

Expected structure after extraction:
```
Celeb-DF-v2/
├── Celeb-real/          # 590 real celebrity videos
├── YouTube-real/        # 300 real YouTube videos
├── Celeb-synthesis/     # 5,639 deepfake videos
└── List_of_testing_videos.txt
```

### 2. Organize for Training

**Option A: Copy files (uses more disk space but safer)**
```bash
python prepare_celebdf.py "C:/path/to/Celeb-DF-v2"
```

**Option B: Symlink files (saves disk space)**
```bash
python prepare_celebdf.py "C:/path/to/Celeb-DF-v2" --symlink
```

This will:
- Organize videos into `data/raw/real/` and `data/raw/fake/`
- Create train/val/test splits (70%/15%/15%)
- Generate `data/processed/metadata.json`

### 3. Verify Dataset
```bash
python -c "from utils import load_config; import json; metadata = json.load(open('data/processed/metadata.json')); print(f'Train: {len(metadata[\"train\"])}, Val: {len(metadata[\"val\"])}, Test: {len(metadata[\"test\"])}')"
```

### 4. Start Training
```bash
python train.py --config configs/config.yaml
```

## Dataset Statistics

- **Real Videos:** 890
  - Celeb-real: 590
  - YouTube-real: 300
- **Fake Videos:** 5,639
- **Total:** 6,529 videos

## Training Recommendations

### For Full Dataset:
```yaml
# In configs/config.yaml
training:
  batch_size: 4  # Reduce if OOM errors
  num_epochs: 30
  learning_rate: 0.0001
```

### For Faster Experimentation (Small Subset):
```bash
# Use only 10% of data for quick testing
python prepare_celebdf.py "C:/path/to/Celeb-DF-v2" --train-ratio 0.07 --val-ratio 0.02
```

## Expected Training Time

**On RTX 4070 Laptop GPU:**
- Full dataset (6,500 videos): ~12-24 hours per epoch
- Subset (650 videos): ~1-2 hours per epoch
- Recommended: 20-30 epochs for convergence

## Tips

1. **Start small:** Test with a subset first to ensure everything works
2. **Monitor GPU:** Use `nvidia-smi` to watch memory usage
3. **Checkpoints:** Model saves every 5 epochs automatically
4. **Early stopping:** Monitor validation accuracy

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size` in config (try 2 or 1)
- Reduce `num_frames` (try 8 instead of 16)
- Reduce `frame_size` (try 112 instead of 224)

**Slow training?**
- Ensure CUDA is being used: check `nvidia-smi`
- Enable mixed precision: `mixed_precision: true` in config
- Increase `num_workers` for faster data loading
