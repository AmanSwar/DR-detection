import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


from model.iBOT import  MaskedViT , CustomiBOT , RetAug
from data_pipeline.data_set import UnitedTrainingDataset , UnitedValidationDataset

class IbotlitModule(pl.LightningModule):
    
    def __init__(
            self,
            img_size = 512,
            patch_size = 16,
            embed_dim = 768,
            mask_ratio = 0.4,
            momentum = 0.996,
            lr = 3e-4,
            warmpup_epochs = 10
    ):
        
        super().__init__()

        self.save_hyperparameters()


        student = MaskedViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            mask_ratio=mask_ratio
        )
        teacher = MaskedViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            mask_ratio=mask_ratio

        )

        self.model = CustomiBOT(
            student=student,
            teacher=teacher,
            embed_dim=embed_dim,
            momentum=momentum,
            mask_ratio=mask_ratio
        )

        self.lr = lr
        self.warmup_ep = warmpup_epochs

    def forward(self , x1 , x2):
        return self.model(x1 , x2)
    
    def training_step(self , batch , batch_idx):

        (x1 , x2) , _ = batch

        loss = self.model(x1 , x2)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):

        optim = torch.optim.AdamW(
            self.model.student.parameters(),
            lr=self.lr,
            weight_decay=0.04
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda= lambda epoch: min(epoch / self.warmup_ep ,1.0)
        )

        return optim , scheduler

    def on_after_batch(self):

        self.model.momentum_update()


class iBOTDataM(pl.LightningDataModule):

    def __init__(
            self,
             train_ds,
             valid_ds,
             img_size=512
             batch_size=64,
             num_worker=8
    ):
        
        super().__init__()

        self.train_dataset = train_ds
        self.val_dataset = valid_ds
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_worker

        self.train_aug = RetAug(img_size=img_size)

        self.val_aug = A.Compose(
            [
                A.Resize(img_size , img_size),
                ToTensorV2()
            ]

        )


    def setup(self , stage=None):

        self.train_ds = UnitedTrainingDataset(
            *self.train_dataset,
            transformation= lambda x : x
        )

        self.val_ds = UnitedValidationDataset(
            *self.val_dataset,
            transformation=self.val_aug
        )


    def train_dataloader(self):

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.ssl_collate,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    

    def ssl_collate(self , batch):

        images , _ = zip(*batch)
        view1 , view2 = [] , []

        for img in images:

            v1 , v2 = self.train_aug(img)

            view1.append(v1)
            view2.append(v2)

        return (torch.stack(view1) , torch.stack(view2)) , torch.zeros(len(images))
    

config = {
    'train_datasets': ["eyepacs", "aptos", "ddr", "idrid"],
    'val_datasets': ["eyepacs", "aptos" , "ddr" , "idrid"],
    'img_size': 512,
    'batch_size': 64,
    'num_workers': 12,
    'embed_dim': 768,
    'mask_ratio': 0.4,
    'momentum': 0.996,
    'lr': 3e-4,
    'warmup_epochs': 10,
    'max_epochs': 300
}


def train_ibot():

    model = IbotlitModule()
    