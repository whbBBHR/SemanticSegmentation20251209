"""
Training Pipeline for Real-Time Segmentation Models
===================================================

Complete training setup with:
- Data augmentation for robustness
- Mixed precision training for speed
- Learning rate scheduling
- Multi-scale training
- Online hard example mining
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining Loss
    Focuses on hardest examples during training
    """
    def __init__(self, thresh=0.7, min_kept=100000):
        super().__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred, target):
        # Calculate per-pixel loss
        pixel_losses = self.criterion(pred, target).view(-1)
        
        # Sort losses
        sorted_losses, _ = torch.sort(pixel_losses, descending=True)
        
        # Keep hard examples
        if sorted_losses.numel() > self.min_kept:
            threshold = sorted_losses[self.min_kept]
            hard_mask = pixel_losses >= threshold
            hard_losses = pixel_losses[hard_mask]
        else:
            hard_losses = sorted_losses
        
        return hard_losses.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: Cross Entropy + Focal Loss + Dice Loss
    """
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss()
    
    def dice_loss(self, pred, target):
        """Dice Loss for better boundary prediction"""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Avoid ignore_index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        return 1.0 - dice.mean()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return ce + 0.5 * focal + 0.3 * dice


class SegmentationTrainer:
    """
    Complete training pipeline for segmentation models
    """
    def __init__(self, model, train_loader, val_loader, num_classes,
                 device='cuda', learning_rate=1e-3, num_epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.num_epochs = num_epochs
        
        # Loss function
        self.criterion = CombinedLoss(num_classes)
        
        # Optimizer with different LR for backbone and head
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'encoder' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},
            {'params': head_params, 'lr': learning_rate}
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics
        self.best_miou = 0.0
        self.train_losses = []
        self.val_mious = []
    
    def calculate_miou(self, pred, target, num_classes):
        """Calculate mean Intersection over Union"""
        pred = pred.argmax(dim=1)
        
        ious = []
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou.item())
        
        return np.mean(ious) if ious else 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Mixed precision training
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_miou = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                miou = self.calculate_miou(outputs, targets, self.num_classes)
                total_miou += miou
        
        avg_miou = total_miou / len(self.val_loader)
        self.val_mious.append(avg_miou)
        return avg_miou
    
    def train(self):
        """Complete training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {self.num_epochs}")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_miou = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val mIoU: {val_miou:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint('best_model.pth', epoch, val_miou)
                print(f"✓ New best model saved! mIoU: {val_miou:.4f}")
            
            print("=" * 60)
        
        print(f"\nTraining completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")
    
    def save_checkpoint(self, filename, epoch, miou):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'miou': miou,
            'train_losses': self.train_losses,
            'val_mious': self.val_mious
        }
        torch.save(checkpoint, filename)


# Data augmentation pipeline
class SegmentationAugmentation:
    """
    Data augmentation for segmentation
    """
    def __init__(self, size=(512, 1024), scale_range=(0.5, 2.0)):
        self.size = size
        self.scale_range = scale_range
    
    def __call__(self, image, mask):
        # Random scale
        scale = np.random.uniform(*self.scale_range)
        new_h = int(image.shape[0] * scale)
        new_w = int(image.shape[1] * scale)
        
        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode='nearest'
        ).squeeze(0).squeeze(0).long()
        
        # Random crop
        if new_h > self.size[0] and new_w > self.size[1]:
            y = np.random.randint(0, new_h - self.size[0])
            x = np.random.randint(0, new_w - self.size[1])
            image = image[:, y:y+self.size[0], x:x+self.size[1]]
            mask = mask[y:y+self.size[0], x:x+self.size[1]]
        else:
            # Pad if needed
            image = F.pad(image, (0, max(0, self.size[1]-new_w),
                                  0, max(0, self.size[0]-new_h)))
            mask = F.pad(mask, (0, max(0, self.size[1]-new_w),
                                0, max(0, self.size[0]-new_h)))
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        
        return image, mask


if __name__ == "__main__":
    print("Training Pipeline Setup Complete!")
    print("\nKey Components:")
    print("- Combined Loss (CE + Focal + Dice)")
    print("- Mixed Precision Training")
    print("- Cosine Annealing LR Schedule")
    print("- Data Augmentation")
    print("- mIoU Metric Tracking")
