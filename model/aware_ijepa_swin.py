import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformer
from einops import rearrange

from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torch.cuda.amp import autocast



class SwinIJepa(nn.Module):
    """Swin Transformer-based IJEPA implementation for medical images"""
    def __init__(
        self,
        img_size=2048,
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4.0,
        drop_rate=0.0,
        **kwargs
    ):
        super().__init__()
        
        # Swin Transformer as context encoder
        self.context_encoder = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        
        # Clone as target encoder (no gradient)
        self.target_encoder = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        # Prediction head for masked patches
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim * 8),
            nn.Linear(embed_dim * 8, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
        # Mask generator parameters
        self.num_masks = 4
        self.mask_ratio = 0.6
        
    def get_random_boxes(self, batch_size):
        """Generate random mask boxes for Swin's windowed architecture"""
        # Implementation depends on your specific masking strategy
        # This is a simplified version for demonstration
        return torch.rand(batch_size, self.num_masks, 4)  # (x1,y1,x2,y2)
        
    def forward(self, x, boxes=None):
        # Extract features from both encoders
        context_features = self.context_encoder(x)
        with torch.no_grad():
            target_features = self.target_encoder(x)
            
        # Get final layer features and reshape for prediction
        B, _, _, C = context_features.shape
        context_features = context_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply predictor to context features
        pred_features = self.predictor(context_features)
        pred_features = pred_features.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        return pred_features, target_features

class DRLesionAttention(nn.Module):
    """Modified attention for Swin features"""
    def __init__(self, dim, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.scale = dim ** -0.5
        self.lesion_queries = nn.Parameter(torch.randn(1, 5, dim))
        
        # Swin-style relative position bias
        self.rel_pos_bias = nn.Parameter(torch.randn((2 * window_size - 1) ** 2, num_heads))
        
    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size, self.window_size, C)
        return windows
        
    def forward(self, x):
        # x: [B, H, W, C] from Swin output
        B, H, W, C = x.shape
        windows = self.window_partition(x)
        num_windows = windows.shape[0]
        
        # Add learnable lesion queries
        lesion_queries = self.lesion_queries.repeat(num_windows, 1, 1)
        window_queries = torch.cat([windows.mean([1,2]), lesion_queries], dim=1)
        
        # Window-based attention
        k = v = windows.flatten(1,2)
        q = window_queries
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.rel_pos_bias
        dots += relative_position_bias
        
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        return out.reshape(B, H, W, C)

class DRSpecificIJEPA(SwinIJepa):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Lesion attention module
        self.lesion_attention = DRLesionAttention(
            dim=kwargs.get('embed_dim', 192),
            window_size=kwargs.get('window_size', 7)
        )
        
        # Lesion prediction heads
        self.lesion_heads = nn.ModuleDict({
            'microaneurysms': nn.Linear(kwargs['embed_dim'], 1),
            'hemorrhages': nn.Linear(kwargs['embed_dim'], 1),
            'hard_exudates': nn.Linear(kwargs['embed_dim'], 1),
            'cotton_wool_spots': nn.Linear(kwargs['embed_dim'], 1),
            'neovascularization': nn.Linear(kwargs['embed_dim'], 1)
        })

    def forward(self, images, boxes=None):
        features, target_features = super().forward(images, boxes)
        
        # Apply lesion attention (features shape: [B, H, W, C])
        features = self.lesion_attention(features)
        
        # Global average pooling
        pooled_features = features.mean(dim=[1, 2])
        
        # Lesion predictions
        lesion_predictions = {
            name: head(pooled_features)
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
        
        micro_batches = torch.chunk(batch, self.grad_accumulation_steps)
        
        for i, micro_batch in enumerate(micro_batches):
            with autocast(enabled=self.mixed_precision):
                features, target_features, lesion_preds = self.model(micro_batch)
                
                main_loss = self.criterion(features, target_features)
                
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
        
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss / self.grad_accumulation_steps



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


