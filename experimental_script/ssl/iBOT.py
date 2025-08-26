import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import RandomApply
import pytorch_lightning as pl

from train.utils import vit_config
from data_pipeline.data_aug import IbotRetAug



class MaskedViT(nn.Module):

    def __init__(
        self , 
        img_size = vit_config["img_size"],
        patch_size = vit_config["patch_size"],
        in_chan = vit_config["in_chans"], 
        embed_dim = vit_config["embed_dim"],
        depth=vit_config["depth"], 
        num_heads=vit_config["num_heads"],
        mlp_ratio = vit_config["mlp_ratio"],
        mask_ratio=vit_config["mask_ratio"],
        lesion_mask_prob=0.7
    ):

        super().__init__()
        self.path_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.lesion_mask_prob = lesion_mask_prob

        self.patch_embed = nn.Conv2d(in_chan , embed_dim , kernel_size=patch_size , stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1 , self.num_patches + 1 , embed_dim))

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_ratio * embed_dim
                )
                for _ in range(depth)
            ]
        )

        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1 , 1 , embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token , std=0.02)

        self.global_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim)
        )

        self.local_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim)
        )


    def lesion_aware_masking(self , x , attention_maps=None):
        B,N,D = x.shape
        num_mask = int(N * self.mask_ratio)
    
        if attention_maps is None:
            rand_mask = torch.rand(B , N , device=x.device)

        else:
            rand_mask = 0.3 * torch.rand(B , N , device=x.device) + 0.7 * attention_maps

        mask_idx = rand_mask.topk(num_mask , dim=1).indices

        mask = torch.zeros(B , N , device=x.device).scatter_(1 , mask_idx , 1)

        return mask.bool()
    
    def forward(self, x, mask_ratio=0.4):
    # Get batch size from the input image.
        B = x.size(0)
        
        # Patch embedding: from [B, 3, 512, 512] to [B, embed_dim, 32, 32]
        x = self.patch_embed(x)
        
        # Flatten and transpose: now x is [B, num_patches, embed_dim] where num_patches = 1024.
        x = x.flatten(2).transpose(1, 2)
        
        # Now extract the correct number of patches.
        N = x.size(1)  # should be 1024 for a 512x512 image with 16x16 patches.
        
        # Add positional embedding (skip the class token position).
        x = x + self.pos_embed[:, 1:, :]
        
        # Compute the mask (mask shape: [B, N]).
        mask = self.lesion_aware_masking(x)
        
        # Expand the mask token to match the number of patches.
        mask_token = self.mask_token.expand(B, N, -1)
        
        # Use the mask (converted to float) to combine x and mask_token.
        x = x * ((~mask).float().unsqueeze(-1)) + mask_token * mask.float().unsqueeze(-1)
        
        # If you have a class token (make sure self.cls_token is defined):
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)
        
        # Process through transformer blocks.
        for blk in self.blocks:
            x = blk(x)
        
        # Project global and local features.
        global_feat = self.global_proj(x[:, 0])
        local_feat = self.local_proj(x[:, 1:])
        
        return global_feat, local_feat, mask
    
class CustomiBOT(nn.Module):

    def __init__(
            self,  
            student=None,
            teacher=None,
            embed_dim=vit_config["embed_dim"],
            temp=0.1,
            mask_ratio=vit_config["mask_ratio"],
            momentum=0.996
    ):
        super().__init__()

        self.student = student if student is not None else MaskedViT()
        self.teacher = teacher if student is not None else MaskedViT()
        self.momentum = momentum
        self.temp = temp
        self.mask_ratio = mask_ratio

        for t_param , s_param in zip(self.teacher.parameters() , self.student.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

        self.global_head = nn.Sequential(
            nn.Linear(embed_dim , 512),
            nn.GELU(),
            nn.Linear(512 , embed_dim)
        )

        self.local_head = nn.Sequential(
            nn.Linear(embed_dim , 512),
            nn.GELU(),
            nn.Linear(512 , embed_dim)
        )

    @torch.no_grad()
    def momentum_update(self):

        for t_param , s_param in zip(self.teacher.parameters() , self.student.parameters()):
            t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

    def compute_loss(self , student_global , student_local , teacher_global , teacher_local , mask):
        
        global_loss = F.cosine_embedding_loss(
            self.global_head(student_global),
            teacher_global.detach(),
            torch.ones(student_global.size(0).to(student_global.device))
        )

        masked_student = student_local[mask].view(-1 , student_local.size(-1))
        masked_teacher = teacher_local[mask].view(-1 , teacher_local.size(-1))
        local_loss = F.cosine_embedding_loss(
            self.local_head(masked_student),
            masked_teacher.detach(),
            torch.ones(masked_student.size(0).to(student_local.device))
        )

        return global_loss + 0.5 * local_loss
    
    def forward(self , x1 , x2):

        s_global1, s_local1, mask1 = self.student(x1)
        s_global2, s_local2, mask2 = self.student(x2)
        
        with torch.no_grad():
            t_global1, t_local1, _ = self.teacher(x1)
            t_global2, t_local2, _ = self.teacher(x2)
        
        loss = 0.5 * (self.compute_loss(s_global1, s_local1, t_global2, t_local2, mask1) +
                      self.compute_loss(s_global2, s_local2, t_global1, t_local1, mask2))
        
        # Update teacher
        self.momentum_update()
        return loss


class iBOTLightningModule(pl.LightningModule):
    def __init__(self, 
                 model: CustomiBOT,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.05,
                 max_epochs: int = 300,
                 warmup_epochs: int = 10):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        loss = self.model(x1, x2, update_teacher=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        # Set update_teacher to False during validation so the teacher remains fixed.
        loss = self.model(x1, x2, update_teacher=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6),
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]


def main():
    # Dataset hyperparameters:
    num_train_samples = 103284  # full training set size
    num_val_samples = 5000      # or any validation subset size you prefer
    batch_size = 128

    # Data augmentation (two views per sample are generated for self-supervision)
    transform = IbotRetAug()

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

    # Initialize iBOT model
    ibot_model = CustomiBOT()

    # Wrap the model in our Lightning module
    lightning_model = iBOTLightningModule(
        model=ibot_model,
        learning_rate=1e-3,
        weight_decay=0.05,
        max_epochs=300,
        warmup_epochs=10
    )

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=300,
        gpus=1,             # Adjust based on your available GPUs
        precision=16,       # Mixed precision training
        accumulate_grad_batches=1,
        log_every_n_steps=50
    )

    # Start training (validation is run at the end of each epoch)
    trainer.fit(lightning_model, train_data_ld, valid_data_ld)

if __name__ == "__main__":
    main()