"""
Training script for deepfake video detection model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
import time
from tqdm import tqdm

from models import MultimodalAVDetector
from utils import (
    DeepfakeDataset,
    get_transforms,
    load_config,
    save_config,
    set_seed,
    get_device,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    accuracy
)


class Trainer:
    """
    Trainer class for deepfake detection model
    """
    
    def __init__(self, config_path: str, resume_path: str = None, num_epochs: int = None):
        # Load configuration
        self.config = load_config(config_path)

        # Optional epoch override (useful for extending runs)
        if num_epochs is not None:
            self.config['training']['num_epochs'] = int(num_epochs)
        
        # Set random seed
        set_seed(self.config['seed'])
        
        # Get device
        self.device = get_device(self.config)
        
        # Create directories
        os.makedirs(self.config['logging']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logs'], exist_ok=True)
        
        # Initialize model
        self.model = MultimodalAVDetector(self.config).to(self.device)
        print(f"Model initialized with {count_parameters(self.model):,} trainable parameters")
        
        # Initialize datasets and dataloaders
        self._init_datasets()
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Initialize loss function
        self._init_loss()
        
        # Mixed precision training
        self.use_amp = self.config['hardware']['mixed_precision'] and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self.start_epoch = 0
        self.best_val_acc = 0.0

        # Optional resume
        if resume_path:
            checkpoint = load_checkpoint(
                self.model,
                resume_path,
                optimizer=self.optimizer,
                device=self.device
            )
            self.start_epoch = checkpoint.get('epoch', -1) + 1
            self.best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"Resuming from epoch {self.start_epoch + 1}")
            print(f"Best validation accuracy so far: {self.best_val_acc:.4f}")
        
    def _init_datasets(self):
        """Initialize datasets and dataloaders"""
        print("Initializing datasets...")
        
        # Get transforms
        train_transforms = get_transforms(self.config, mode='train')
        val_transforms = get_transforms(self.config, mode='val')
        
        # Create datasets
        self.train_dataset = DeepfakeDataset(
            data_dir=self.config['data']['raw_dir'],
            metadata_file=os.path.join(self.config['data']['processed_dir'], 'metadata.json'),
            num_frames=self.config['data']['num_frames'],
            frame_size=self.config['data']['frame_size'],
            audio_sample_rate=self.config['data']['audio_sample_rate'],
            clip_duration=self.config['data']['clip_duration'],
            transform=train_transforms[0],
            audio_transform=train_transforms[1],
            mode='train'
        )
        
        self.val_dataset = DeepfakeDataset(
            data_dir=self.config['data']['raw_dir'],
            metadata_file=os.path.join(self.config['data']['processed_dir'], 'metadata.json'),
            num_frames=self.config['data']['num_frames'],
            frame_size=self.config['data']['frame_size'],
            audio_sample_rate=self.config['data']['audio_sample_rate'],
            clip_duration=self.config['data']['clip_duration'],
            transform=val_transforms[0],
            audio_transform=val_transforms[1],
            mode='val'
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['validation']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Validation dataset: {len(self.val_dataset)} samples")
        
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_name = self.config['training']['scheduler'].lower()
        num_epochs = self.config['training']['num_epochs']
        
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=num_epochs // 3,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
    def _init_loss(self):
        """Initialize loss function"""
        loss_type = self.config['training']['loss']['type']
        label_smoothing = self.config['training']['loss'].get('label_smoothing', 0.0)
        
        if loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            
            batch_size = video.size(0)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda'):
                    output = self.model(video, audio)
                    loss = self.criterion(output['logits'], labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(video, audio)
                loss = self.criterion(output['logits'], labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Calculate accuracy
            acc = accuracy(output['logits'], labels)
            
            # Update meters
            losses.update(loss.item(), batch_size)
            accs.update(acc, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accs.avg:.4f}'
            })
        
        return losses.avg, accs.avg
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            
            batch_size = video.size(0)
            
            # Forward pass
            output = self.model(video, audio)
            loss = self.criterion(output['logits'], labels)
            
            # Calculate accuracy
            acc = accuracy(output['logits'], labels)
            
            # Update meters
            losses.update(loss.item(), batch_size)
            accs.update(acc, batch_size)
        
        return losses.avg, accs.avg
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['training']['num_epochs']} epochs")
        print(f"{'='*60}")
        
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.config['validation']['eval_frequency'] == 0:
                val_loss, val_acc = self.validate()
                
                print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print(f"Time: {time.time() - epoch_start:.2f}s")
                
                # Update scheduler
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
                
                # Save checkpoint
                if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                    save_path = os.path.join(
                        self.config['logging']['checkpoint_dir'],
                        f'checkpoint_epoch_{epoch+1}.pth'
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        save_path,
                        val_acc=val_acc,
                        train_acc=train_acc
                    )
                
                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    best_path = os.path.join(
                        self.config['logging']['checkpoint_dir'],
                        'best_model.pth'
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        best_path,
                        val_acc=val_acc,
                        train_acc=train_acc
                    )
                    print(f"[OK] New best model saved! Val Acc: {val_acc:.4f}")
                
                print(f"{'='*60}\n")
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Override total number of epochs in config')
    args = parser.parse_args()
    
    trainer = Trainer(args.config, resume_path=args.resume, num_epochs=args.num_epochs)
    trainer.train()
