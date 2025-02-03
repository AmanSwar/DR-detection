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

from model.ijepa import IJEPA , IJEPALoss
from data_pipeline import data_aug , data_set

class IjepaLightning(pl.LightningModule):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            encoder_depth=12,
            predictor_depth=4,
            num_heads=12,
            mlp_ratio=4,
            dropout=0.1,
            momentum=0.999,
            base_lr=1.5e-4,
            weight_decay=0.05,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.ijepa = IJEPA(
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

        self.criterion = IJEPALoss()
        self.momentum = momentum

    def training_step(self, batch , batch_idx):
        images = batch
        pred_feat, target_feat = self.ijepa(images)
        loss = self.criterion(pred_feat, target_feat)
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        self.log('grad_norm', self.get_grad_norm(), prog_bar=False)
        return loss
    
    def on_train_batch_end(self , outputs , batch , batch_idx):
        with torch.no_grad():
            for t_param, c_param in zip(self.ijepa.target_encoder.parameters(),self.ijepa.context_encoder.parameters()):
                t_param.data = t_param.data * self.momentum + c_param.data * (1. - self.momentum)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.ijepa.context_encoder.parameters(),
            lr=self.hparams.base_lr,
            betas=(0.9 , 0.95),
            weight_decay=self.hparams.weight_decay
        )


        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.trainer.max_epochs,
        #     eta_min=1e-6
        # )
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=290, eta_min=1e-6)
        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        return [optimizer] , [scheduler]
    
class IJEPADataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset_names=("eyepacs", "aptos", "ddr", "idrid"),
            batch_size=64,
            num_workers=8,
            img_size=224
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_names = dataset_names
        self.transform = data_aug.IJEPAAugmentation()

    def setup(self , stage=None):
        self.full_dataset = data_set.UnitedTrainingDataset(
            *self.dataset_names,
            transformation=self.transform
        )

    def train_dataloader(self):

        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
def train_ijepa():

    model = IjepaLightning(
        embed_dim=1024,
        num_heads=16,
        encoder_depth=12,
        predictor_depth=4,
        base_lr=1.5e-4,
        weight_decay=0.05
    )

    datamodule = IJEPADataModule(
        batch_size=64,
        num_workers=8
    )

    wandb_logger = WandbLogger(project="ijepa_training_lightning" , log_model=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=300,
        precision='16-mixed',
        logger=wandb_logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=None,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                dirpath='ijepa_checkpts',
                save_top_k=2,
                monitor='train_loss',
                every_n_epochs=5
            )
            
        ]
        
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train_ijepa()
        
        