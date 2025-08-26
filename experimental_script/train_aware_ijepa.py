import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from typing import Optional, Dict, Any
import torchmetrics

from train.aware_ijepa import DRSpecificIJEPA , MemoryEfficientBatchSampler , DRSpecificAugmentation
from data_pipeline.data_set import UnitedTrainingDataset , UnitedValidationDataset

class DRLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: DRSpecificIJEPA,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        grad_accumulation_steps: int = 4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        
        # Initialize metrics
        self.train_lesion_auroc = nn.ModuleDict({
            name: torchmetrics.AUROC(task='binary')
            for name in self.model.lesion_heads.keys()
        })
        self.val_lesion_auroc = nn.ModuleDict({
            name: torchmetrics.AUROC(task='binary')
            for name in self.model.lesion_heads.keys()
        })
        
        # Loss weights
        self.feature_loss_weight = 1.0
        self.lesion_loss_weight = 0.5

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, lesion_labels = batch
        features, target_features, lesion_preds = self.model(images)
        
        # Compute main I-JEPA loss
        feature_loss = F.mse_loss(features, target_features)
        
        # Compute lesion prediction loss
        lesion_losses = {}
        for name, pred in lesion_preds.items():
            lesion_losses[name] = F.binary_cross_entropy_with_logits(
                pred, lesion_labels[name]
            )
            self.train_lesion_auroc[name](pred.sigmoid(), lesion_labels[name])
            
        total_lesion_loss = sum(lesion_losses.values())
        
        # Combine losses
        total_loss = (
            self.feature_loss_weight * feature_loss +
            self.lesion_loss_weight * total_lesion_loss
        )
        
        # Log metrics
        self.log('train/feature_loss', feature_loss, sync_dist=True)
        for name, loss in lesion_losses.items():
            self.log(f'train/lesion_loss_{name}', loss, sync_dist=True)
        self.log('train/total_loss', total_loss, sync_dist=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, lesion_labels = batch
        features, target_features, lesion_preds = self.model(images)
        
        # Compute losses
        feature_loss = F.mse_loss(features, target_features)
        
        lesion_losses = {}
        for name, pred in lesion_preds.items():
            lesion_losses[name] = F.binary_cross_entropy_with_logits(
                pred, lesion_labels[name]
            )
            self.val_lesion_auroc[name](pred.sigmoid(), lesion_labels[name])
            
        total_lesion_loss = sum(lesion_losses.values())
        total_loss = (
            self.feature_loss_weight * feature_loss +
            self.lesion_loss_weight * total_lesion_loss
        )
        
        # Log metrics
        self.log('val/feature_loss', feature_loss, sync_dist=True)
        for name, loss in lesion_losses.items():
            self.log(f'val/lesion_loss_{name}', loss, sync_dist=True)
        self.log('val/total_loss', total_loss, sync_dist=True)
        
        return total_loss

    def on_validation_epoch_end(self):
        # Log AUROC metrics
        for name in self.model.lesion_heads.keys():
            train_auroc = self.train_lesion_auroc[name].compute()
            val_auroc = self.val_lesion_auroc[name].compute()
            
            self.log(f'train/auroc_{name}', train_auroc, sync_dist=True)
            self.log(f'val/auroc_{name}', val_auroc, sync_dist=True)
            
            # Reset metrics
            self.train_lesion_auroc[name].reset()
            self.val_lesion_auroc[name].reset()

    def configure_optimizers(self):
        # Set up optimizer with weight decay
        param_groups = [
            {
                'params': [p for n, p in self.named_parameters() if 'bias' not in n],
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters() if 'bias' in n],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.learning_rate
        )
        
        # Learning rate scheduler with warmup
        scheduler = {
            'scheduler': self.get_scheduler(optimizer),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]
    
    def get_scheduler(self, optimizer):
        warmup_steps = self.hparams.warmup_epochs * self.trainer.estimated_stepping_batches
        total_steps = self.hparams.max_epochs * self.trainer.estimated_stepping_batches
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_distributed_dr_model(
    model: DRSpecificIJEPA,
    train_dataset,
    val_dataset,
    batch_size: int = 32,
    num_gpus: int = -1,  # -1 means use all available GPUs
    max_epochs: int = 100,
    accumulate_grad_batches: int = 4,
    precision: str = '16-mixed',
    project_name: str = 'DR-IJEPA',
    experiment_name: str = 'distributed-training'
):
    """
    Set up and run distributed training using PyTorch Lightning
    """
    # Initialize Lightning module
    model = DRLightningModule(
        model=model,
        max_epochs=max_epochs,
        grad_accumulation_steps=accumulate_grad_batches
    )
    
    # Set up data loaders with memory efficient sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=MemoryEfficientBatchSampler(train_dataset, batch_size),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        log_model=True
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='{epoch}-{val_loss:.2f}',
            monitor='val/total_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val/total_loss',
            patience=10,
            mode='min'
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=max_epochs,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        deterministic=True,
        log_every_n_steps=10
    )
    
    # Start training
    trainer.fit(model, train_loader, val_loader)
    
    return model


model = DRSpecificIJEPA()

dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid"]
train_dataset = UnitedTrainingDataset(*dataset_names , transformation=DRSpecificAugmentation())
valid_dataset = UnitedValidationDataset(*dataset_names , transformation=DRSpecificAugmentation())

trained_model = train_distributed_dr_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=valid_dataset,
    batch_size=64,
    num_gpus=-1,  # Use all available GPUs
    max_epochs=100,
    accumulate_grad_batches=4,
    precision='16-mixed',
    project_name='DR-IJEPA',
    experiment_name='distributed-training'
)