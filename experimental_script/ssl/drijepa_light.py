import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from model.utils import RearrangeAndLayerNorm, vit_config


#####################
# MODEL DEFINITIONS #
#####################

class Patchify(nn.Module):
    """
    Convert images into patches, with a hierarchical projection to support large image sizes.
    """
    def __init__(self, img_size=1024, patch_size=16, in_chan=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.hierarch_proj = nn.Sequential(
            nn.Conv2d(in_chan, embed_dim // 4, kernel_size=7, stride=2, padding=3),
            RearrangeAndLayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            RearrangeAndLayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size // 4, stride=patch_size // 4)
        )

    def forward(self, x):
        x = self.hierarch_proj(x)  # (B, C, H, W)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            # Create a transformer block
            block = nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ])
            self.layers.append(block)

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            x_norm = norm1(x)
            x_attn = attn(x_norm, x_norm, x_norm)
            # attn returns a tuple (output, weights)
            x = x + x_attn[0]
            x_norm_2 = norm2(x)
            x_mlp = mlp(x_norm_2)
            x = x + x_mlp
        return x


class DRIjepa(nn.Module):
    def __init__(
        self,
        img_size=vit_config["img_size"],
        patch_size=vit_config["patch_size"],
        in_chan=vit_config["in_chans"],
        embed_dim=vit_config["embed_dim"],
        encoder_depth=vit_config["depth"],
        pred_depth=vit_config["pred_depth"],
        n_heads=vit_config["num_heads"],
        mlp_ratio=vit_config["mlp_ratio"],
        drop=0.1
    ):
        super().__init__()

        self.patch_embed = Patchify(img_size=img_size, patch_size=patch_size, in_chan=in_chan, embed_dim=embed_dim)

        self.target_encoder = TransformerEncoder(
            dim=embed_dim,
            depth=encoder_depth,
            heads=n_heads,
            mlp_dim=embed_dim * mlp_ratio,
            dropout=drop
        )

        self.context_encoder = TransformerEncoder(
            dim=embed_dim,
            depth=encoder_depth,
            heads=n_heads,
            mlp_dim=embed_dim * mlp_ratio,
            dropout=drop
        )

        self.predictor = nn.Sequential(
            TransformerEncoder(
                dim=embed_dim,
                depth=pred_depth,
                heads=n_heads,
                mlp_dim=embed_dim * mlp_ratio,
                dropout=drop
            ),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.apply(self._init_weights)
        self.grid_size = img_size // patch_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def get_random_boxes(self, batch_size, n_box=6):
        boxes = []
        for b in range(batch_size):
            batch_boxes = []
            for _ in range(n_box):
                center_bias = torch.randn(2) * 0.5
                x_center = self.grid_size // 2 + int(center_bias[0] * self.grid_size // 4)
                y_center = self.grid_size // 2 + int(center_bias[1] * self.grid_size // 4)
                w = torch.randint(4, 8, (1,)).item()
                h = torch.randint(4, 8, (1,)).item()
                x1 = torch.randint(0, self.grid_size - w, (1,)).item()
                y1 = torch.randint(0, self.grid_size - h, (1,)).item()
                x2 = x1 + w
                y2 = y1 + h
                batch_boxes.append([x1, y1, x2, y2])
            boxes.append(batch_boxes)
        return torch.tensor(boxes)

    def extract_target(self, feature, boxes):
        B, N, D = feature.shape
        H = W = int(N ** 0.5)
        features = rearrange(feature, 'b (h w) d -> b d h w', h=H)
        
        target_features = []
        for b in range(B):
            batch_targets = []
            for box in boxes[b]:
                x1, y1, x2, y2 = box
                target = features[b:b+1, :, y1:y2, x1:x2]
                target = F.adaptive_avg_pool2d(target, (min(H, 2), min(W, 2)))
                target = rearrange(target, 'b c h w -> b (h w) c')
                batch_targets.append(target)
            target_features.append(torch.cat(batch_targets, dim=1))
        return torch.stack(target_features)

    @torch.no_grad()
    def momentum_update(self, target_encoder: nn.Module, context_encoder: nn.Module, momentum=0.999):
        for target_param, context_param in zip(target_encoder.parameters(), context_encoder.parameters()):
            target_param.data.mul_(momentum).add_((1 - momentum) * context_param.data)

    def forward(self, images, boxes=None):
        B = images.shape[0]
        if boxes is None:
            boxes = self.get_random_boxes(B)

        x = self.patch_embed(images)
        context_features = self.context_encoder(x)

        with torch.no_grad():
            target_feat = self.target_encoder(x)
            target_feat = self.extract_target(target_feat, boxes)

        predicted_feature = self.predictor(context_features)
        predicted_feature = self.extract_target(predicted_feature, boxes)

        return predicted_feature, target_feat


class IJEPALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_features, target_features):
        predicted_features = F.normalize(predicted_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        sim = torch.einsum('bnd, bnd -> bn', predicted_features, target_features)
        loss = -sim.mean()
        return loss


def create_DRijepa(
    
):
    model = DRIjepa()
    loss = IJEPALoss()
    return model, loss


#############################
# LIGHTNING MODULE WRAPPER  #
#############################

class LightningDRIjepa(pl.LightningModule):
    def __init__(self, model, loss_fn, lr=1.5e-4, T_max=300, eta_min=1e-6):
        """
        Args:
            model: the DRijepa model instance.
            loss_fn: the IJEPALoss instance.
            lr: learning rate.
            T_max: maximum number of iterations for the cosine annealing scheduler.
            eta_min: minimum learning rate.
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.T_max = T_max
        self.eta_min = eta_min

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Assuming the batch is a tuple where the first element is the image tensor.
        images = batch[0]
        pred_feat, target_feat = self.model(images)
        loss = self.loss_fn(pred_feat, target_feat)
        # Log the training loss (both step and epoch level logging)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=self.eta_min
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # Standard optimizer step
        optimizer.step(closure=closure)
        # Update momentum encoder after optimizer step
        momentum = 0.99 + (0.999 - 0.99) * (self.current_epoch / self.trainer.max_epochs)
        self.model.momentum_update(self.model.target_encoder, self.model.context_encoder)
        optimizer.zero_grad()


#########################
# MAIN TRAINING SCRIPT  #
#########################

if __name__ == "__main__":
    print("Starting training using PyTorch Lightning...")
    BATCH_SIZE = 32

    # Import your dataloader components (adjust the import as needed).
    from data_pipeline import data_set, data_aug

    dataset_names = ["eyepacs", "aptos", "ddr", "idrid"]
    uniform_data_ld = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=data_aug.IJEPAAugmentation(),
        batch_size=BATCH_SIZE,
        num_workers=4,
        sampler=True
    )
    train_loader = uniform_data_ld.get_loader()

    # Create the model and loss
    model, loss_fn = create_DRijepa()

    # Wrap the model into our LightningModule
    lightning_model = LightningDRIjepa(model=model, loss_fn=loss_fn)

    # Set up a ModelCheckpoint callback (optional; Lightning will save checkpoints automatically)
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoint',
        filename='DRijepa-{epoch:02d}-{train_loss:.4f}',
        save_top_k=3,
        mode='min'
    )

    # Create the Lightning Trainer.
    # Here we set accelerator="gpu", devices=2 and strategy="ddp" for distributed data parallel training across 2 A100 GPUs.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=300,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50
    )

    # Optionally, initialize wandb logger (or use any other logger supported by Lightning)
    wandb_logger = pl.loggers.WandbLogger(project="DRIjepa")
    trainer.logger = wandb_logger

    # Start training!
    trainer.fit(lightning_model, train_dataloaders=train_loader)
