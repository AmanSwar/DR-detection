import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import LinearLR, ChainedScheduler
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from model.ijepa import IJEPA 
from model.DRijepa import DRIjepa , IJEPALoss
from data_pipeline import data_aug , data_set
class IJEPA_Lightning(pl.LightningModule):
    def __init__(
        self,
        img_size=512,  # Increased for retinal images
        patch_size=32,  # Larger patches
        in_chans=3,
        embed_dim=1024,  # Increased embedding dimension
        encoder_depth=12,
        predictor_depth=4,
        num_heads=16,  # Increased heads
        mlp_ratio=4,
        dropout=0.1,
        learning_rate=1.5e-4,
        weight_decay=0.05,
        warmup_epochs=5,
        max_epochs=50,
        batch_size=32
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize base IJEPA model
        self.model = DRIjepa(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            predictor_depth=predictor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.loss_fn = IJEPALoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size

    def get_random_boxes(self, batch_size, n_boxes=6):  # Increased number of boxes
        """Generate random target boxes for masked prediction"""
        boxes = []
        for _ in range(batch_size):
            batch_boxes = []
            for _ in range(n_boxes):
                # Adjusted for 512/32 = 16 grid
                x1 = torch.randint(0, 16, (1,)).item()
                y1 = torch.randint(0, 16, (1,)).item()
                w = torch.randint(3, 7, (1,)).item()  # Larger boxes
                h = torch.randint(3, 7, (1,)).item()
                batch_boxes.append([x1, y1, x1 + w, y1 + h])
            boxes.append(batch_boxes)
        return torch.tensor(boxes)

    def training_step(self, batch, batch_idx):
        images = batch
        boxes = self.get_random_boxes(images.shape[0])
        
        pred_features, target_features = self.model(images, boxes)
        loss = self.loss_fn(pred_features, target_features)
        
        # Update teacher encoder
        self.model.momentum_update(
            self.model.target_encoder,
            self.model.context_encoder,
            momentum=0.9996  # Increased momentum for stability
        )
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Separate parameters for weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d)
        
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

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.learning_rate * (self.batch_size / 256),
            betas=(0.9, 0.95)
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - self.warmup_epochs,
                eta_min=1e-6
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler['scheduler']],
                    milestones=[self.warmup_epochs]
                ),
                'interval': 'epoch',
                'frequency': 1
            }

        return [optimizer], [scheduler]

def main():
    # Dataset setup (using your existing data pipeline)
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    uniform_data_ld = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=data_aug.IJEPAAugmentation(),
        batch_size=32,  # Per GPU batch size
        num_workers=8,
        sampler=True
    )
    train_loader = uniform_data_ld.get_loader()

    # Initialize model
    model = IJEPA_Lightning(
        img_size=512,
        batch_size=32,
        max_epochs=50
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='ijepa-retina-{epoch:02d}-{train_loss:.3f}',
            save_top_k=3,
            monitor='train_loss',
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Logger
    wandb_logger = WandbLogger(project="ijepa-retina", log_model="all")

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,  # Your 2 A100s
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=50,
        precision=16,  # Mixed precision
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=3.0,
        accumulate_grad_batches=1,
        log_every_n_steps=50,
        sync_batchnorm=True
    )

    # Train
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()