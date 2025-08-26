import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange
from torch.cuda.amp import autocast, GradScaler


from train.DRijepa import DRIjepa
from PIL import Image

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torch.cuda.amp import autocast




class DRLesionAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        #custom queries -> 5,dim
        self.lesion_queries = nn.Parameter(torch.randn(1, 5, dim)) 
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # no op
        q, k, v = map(lambda t: rearrange(t, 'b n d -> b n d'), qkv)
        
        
        lesion_queries = self.lesion_queries.repeat(b, 1, 1)
        q = torch.cat([q, lesion_queries], dim=1)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        return out

class DRSpecificIJEPA(DRIjepa):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        self.lesion_attention = DRLesionAttention(kwargs.get('embed_dim', 1024))
        
        
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
            #means along the dim=1
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




class DRIJEPALightning(pl.LightningModule):
    def __init__(
        self,
        img_size=2048,
        patch_size=32,
        in_chan=3,
        embed_dim=1024,
        encoder_depth=12,
        pred_depth=4,
        num_heads=16,
        mlp_ratio=4,
        dropout=0.1,
        learning_rate=1.5e-4,
        weight_decay=0.05,
        warmup_epochs=5,
        max_epochs=50,
        batch_size=32,
        lesion_loss_weight=0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize DR-specific IJEPA model
        self.model = DRSpecificIJEPA(
            img_size=img_size,
            patch_size=patch_size,
            in_chan=in_chan,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            pred_depth=pred_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=dropout
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lesion_loss_weight = lesion_loss_weight

    def training_step(self, batch, batch_idx):
        images, lesion_labels = batch
        boxes = self.model.get_random_boxes(images.shape[0])
        
        # Forward pass with mixed precision
        with autocast():
            features, target_features, lesion_preds = self.model(images, boxes)
            
     
            main_loss = F.mse_loss(features, target_features)
            
            # Lesion detection loss
            lesion_loss = 0
            if lesion_labels is not None:
                for name, pred in lesion_preds.items():
                    if name in lesion_labels:
                        lesion_loss += F.binary_cross_entropy_with_logits(
                            pred,
                            lesion_labels[name]
                        )
            
            # Combined loss
            total_loss = main_loss + self.lesion_loss_weight * lesion_loss
        
        # Log losses
        self.log('train_main_loss', main_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_lesion_loss', lesion_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Update target encoder with momentum
        self.model.momentum_update(
            self.model.target_encoder,
            self.model.context_encoder,
            momentum=0.9996
        )
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, lesion_labels = batch
        boxes = self.model.get_random_boxes(images.shape[0])
        
        features, target_features, lesion_preds = self.model(images, boxes)
        
        # Calculate losses
        main_loss = F.mse_loss(features, target_features)
        
        lesion_loss = 0
        if lesion_labels is not None:
            for name, pred in lesion_preds.items():
                if name in lesion_labels:
                    lesion_loss += F.binary_cross_entropy_with_logits(
                        pred,
                        lesion_labels[name]
                    )
        
        total_loss = main_loss + self.lesion_loss_weight * lesion_loss
        
        # Log validation metrics
        self.log('val_main_loss', main_loss, on_epoch=True, sync_dist=True)
        self.log('val_lesion_loss', lesion_loss, on_epoch=True, sync_dist=True)
        self.log('val_total_loss', total_loss, on_epoch=True, sync_dist=True)
        
        return total_loss

    def configure_optimizers(self):
        # Separate parameters for weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optim_groups,
            lr=self.learning_rate * (self.batch_size / 256),
            betas=(0.9, 0.95)
        )

        # Cosine learning rate schedule with warmup
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=1e-6
        )
        
        if self.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.warmup_epochs]
                ),
                'interval': 'epoch',
                'frequency': 1
            }
        else:
            scheduler = {
                'scheduler': main_scheduler,
                'interval': 'epoch',
                'frequency': 1
            }

        return [optimizer], [scheduler]

def main():
    # Dataset setup with DR-specific augmentations
    train_transform, val_transform = create_dr_specific_transforms()
    
    from data_pipeline import data_set , data_aug
    dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid"]
    uniform_training_data_ld = data_set.SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=data_aug.IJEPAAugmentation(),
        batch_size=32,
        num_work=4,
    )

    train_data_ld = uniform_training_data_ld.get_loader()
    
    uniform_validation_data_ld = data_set.SSLValidLoader(
        dataset_names=dataset_names,
        transformation=None,
        batch_size=32,
        num_work=4,
    )

    valid_data_ld = uniform_validation_data_ld.get_loader()


    # Initialize model
    model = DRIJEPALightning(
        img_size=2048,
        batch_size=32,
        max_epochs=50
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='dr-ijepa-{epoch:02d}-{val_total_loss:.3f}',
            save_top_k=3,
            monitor='val_total_loss',
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Trainer setup
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,  # Adjust based on available GPUs
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=50,
        precision=16,  # Mixed precision training
        callbacks=callbacks,
        gradient_clip_val=3.0,
        accumulate_grad_batches=4,  # Gradient accumulation
        log_every_n_steps=50,
        sync_batchnorm=True
    )

    # Train
    trainer.fit(model, train_data_ld, valid_data_ld)

if __name__ == "__main__":
    main()
