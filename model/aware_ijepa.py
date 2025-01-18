import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import albumentations as A


from DRijepa import DRIjepa
from PIL import Image


class DRSpecificAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.5
            ),
            A.OneOf([
                # lightning
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5),
            ], p=0.3),
            A.OneOf([

                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.2),
            # Lesion-focused augmentations
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                A.UnsharpMask(blur_limit=(3, 7), p=0.5),
            ], p=0.3),
            # Color augmentations for different camera settings
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=5,
                p=0.3
            ),
            # Preserve circular FOV
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                mask_fill_value=0,
                p=0.2
            ),
        ])

    def __call__(self, image):
        image = np.array(image)  # Ensure input is numpy array
        augmented = self.transform(image=image)
        return Image.fromarray(augmented['image'])

class MemoryEfficientBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gradient_accumulation_steps = 4
        
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))
        else:
            indices = torch.arange(len(self.dataset))
            
        mini_batch_size = self.batch_size // self.gradient_accumulation_steps
        for i in range(0, len(indices), mini_batch_size):
            yield indices[i:i + mini_batch_size]

class DRLesionAttention(nn.Module):
    """Attention module specifically designed for DR lesions"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.lesion_queries = nn.Parameter(torch.randn(1, 5, dim)) 
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n d -> b n d'), qkv)
        
        # Add lesion-specific queries
        lesion_queries = self.lesion_queries.repeat(b, 1, 1)
        q = torch.cat([q, lesion_queries], dim=1)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        return out

class DRSpecificIJEPA(DRIjepa):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add DR-specific components
        self.lesion_attention = DRLesionAttention(kwargs.get('embed_dim', 1024))
        
        # Add lesion-specific prediction heads
        self.lesion_heads = nn.ModuleDict({
            'microaneurysms': nn.Linear(kwargs.get('embed_dim', 1024), 1),
            'hemorrhages': nn.Linear(kwargs.get('embed_dim', 1024), 1),
            'hard_exudates': nn.Linear(kwargs.get('embed_dim', 1024), 1),
            'cotton_wool_spots': nn.Linear(kwargs.get('embed_dim', 1024), 1),
            'neovascularization': nn.Linear(kwargs.get('embed_dim', 1024), 1)
        })

    def forward(self, images, boxes=None):
        features, target_features = super().forward(images, boxes)
        
        # Apply lesion attention
        features_with_attention = self.lesion_attention(features)
        
        # Get lesion-specific predictions
        lesion_predictions = {
            name: head(features_with_attention.mean(1))
            for name, head in self.lesion_heads.items()
        }
        
        return features, target_features, lesion_predictions

class DRTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        grad_accumulation_steps=4,
        mixed_precision=True
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.grad_accumulation_steps = grad_accumulation_steps
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
    def train_step(self, batch, lesion_labels=None):
        self.model.train()
        total_loss = 0
        
        # Split batch for gradient accumulation
        micro_batches = torch.chunk(batch, self.grad_accumulation_steps)
        
        for i, micro_batch in enumerate(micro_batches):
            with autocast(enabled=self.mixed_precision):
                features, target_features, lesion_preds = self.model(micro_batch)
                
                # Compute main I-JEPA loss
                main_loss = self.criterion(features, target_features)
                
                # Compute lesion prediction loss if labels are provided
                lesion_loss = 0
                if lesion_labels is not None:
                    for name, pred in lesion_preds.items():
                        lesion_loss += F.binary_cross_entropy_with_logits(
                            pred,
                            lesion_labels[name][i * len(micro_batch):(i + 1) * len(micro_batch)]
                        )
                
                loss = main_loss + 0.5 * lesion_loss
                scaled_loss = loss / self.grad_accumulation_steps
                
            if self.mixed_precision:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            total_loss += loss.item()
        
        # Update weights
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.optimizer.zero_grad()
        
        return total_loss / self.grad_accumulation_steps

def create_dr_specific_transforms(img_size=2048):
    """Create DR-specific image transformations"""
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        DRSpecificAugmentation(),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def train_dr_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    learning_rate=1e-4,
    weight_decay=1e-4
):
    """Training loop with DR-specific optimizations"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    trainer = DRTrainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        device=device,
        mixed_precision=True
    )
    
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        
        # Training
        model.train()
        for batch, lesion_labels in train_loader:
            batch = batch.to(device)
            loss = trainer.train_step(batch, lesion_labels)
            train_loss += loss
            
        # Validation
        model.eval()
        with torch.no_grad():
            for batch, lesion_labels in val_loader:
                batch = batch.to(device)
                features, target_features, lesion_preds = model(batch)
                loss = trainer.criterion(features, target_features)
                val_loss += loss.item()
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")




