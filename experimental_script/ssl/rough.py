import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
import wandb

# Import your own modules (adjust these as needed)
from model.utils import RearrangeAndLayerNorm, vit_config
from data_pipeline import data_set, data_aug 




class Patchify(nn.Module):
    """
    Convert images into patches with a hierarchical projection to support large image sizes.
    """
    def __init__(self, img_size=1024, patch_size=16, in_chan=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # A hierarchical projection can help in learning representations (as suggested by Claude)
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
        x = self.hierarch_proj(x)  # shape: (B, C, H, W)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
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
            # nn.MultiheadAttention returns a tuple; we take the first element (the attention output)
            x_attn = attn(x_norm, x_norm, x_norm)[0]
            x = x + x_attn
            x_norm2 = norm2(x)
            x_mlp = mlp(x_norm2)
            x = x + x_mlp
        return x
    

class DRIjepa(nn.Module):
    def __init__(self,
                 img_size=vit_config["img_size"],
                 patch_size=vit_config["patch_size"],
                 in_chan=vit_config["in_chans"],
                 embed_dim=vit_config["embed_dim"],
                 encoder_depth=vit_config["depth"],
                 pred_depth=vit_config["pred_depth"],
                 n_heads=vit_config["num_heads"],
                 mlp_ratio=vit_config["mlp_ratio"],
                 drop=0.1):
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
                x1 = max(0, x_center - w // 2)
                y1 = max(0, y_center - h // 2)
                x2 = min(self.grid_size, x1 + w)
                y2 = min(self.grid_size, y1 + h)
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
                # Resize target to a fixed small size (using adaptive average pooling)
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
        print(predicted_features.shape)
        print(predicted_features.shape)
        predicted_features = F.normalize(predicted_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        # Compute cosine similarity for each patch feature and then average
        # sim = torch.einsum('bnd,bnd->bn', predicted_features, target_features)
        sim = (predicted_features * target_features).sum(dim=-1)
        loss = -sim.mean()
        return loss
    

def create_DRijepa():
    model = DRIjepa()
    loss = IJEPALoss()
    return model, loss

class Trainer:
    def __init__(self,
                 model,
                 loss_fn,
                 train_loader,
                 optim,
                 scheduler=None,
                 max_ep=300,
                 save_dir="checkpoint",
                 log_interval=100,
                 save_interval=10,
                 device="cuda"):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.optim = optim
        self.scheduler = scheduler
        self.max_ep = max_ep
        self.save_dir = save_dir if isinstance(save_dir, (str, os.PathLike)) else save_dir
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.device = device

    def save_checkpoint(self, epoch, loss):
        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optim_state_dict': self.optim.state_dict(),
            'loss': loss,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_ep_{epoch}.pt"))

    def train_epoch(self, epoch):
        

        self.model.train()
        n_batch = len(self.train_loader)
        total_loss = 0
        pbar = tqdm(total=n_batch, desc=f"Epoch: {epoch}")
        for batch_idx, batch in enumerate(self.train_loader):
            # Assuming your dataloader returns a tuple with images as the first element
            img = batch.to(self.device)
            self.optim.zero_grad()

            pred_feat, target_feat = self.model(img)
            loss = self.loss_fn(pred_feat, target_feat)
            loss.backward()
            self.optim.step()

            # Call momentum update on the underlying model (DDP or not)
            if hasattr(self.model, "module"):
                self.model.module.momentum_update(
                    self.model.module.target_encoder,
                    self.model.module.context_encoder
                )
            else:
                self.model.momentum_update(
                    self.model.target_encoder,
                    self.model.context_encoder
                )

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })
            pbar.update()
        pbar.close()
        avg_loss = total_loss / n_batch
        return avg_loss

    def train(self):
        best_loss = float("inf")
        self.model.to(self.device)
        for ep in range(self.max_ep):
            epoch_start_time = time.time()
            loss = self.train_epoch(ep)
            ep_dur = time.time() - epoch_start_time

            if self.scheduler is not None:
                self.scheduler.step()

            wandb.log({
                'epoch_loss': loss,
                'epoch': ep,
                'learning_rate': self.optim.param_groups[0]['lr'],
                'epoch_duration': ep_dur
            })
            print(f"Epoch {ep} : Loss = {loss:.4f} , Time={ep_dur:.2f}s")
            if ep % self.save_interval == 0 or loss < best_loss:
                self.save_checkpoint(ep, loss)
                if loss < best_loss:
                    best_loss = loss


def train_single_gpu(data_ld , batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting single-GPU training on device:", device)

    # Initialize wandb if desired
    wandb.init(project="dri_jepa_single_gpu")

    # BATCH_SIZE = 64
    # # Create your dataloader using your data pipeline
    # dataset_names = ["eyepacs", "aptos", "ddr", "idrid"]
    # uniform_data_ld = data_set.UniformTrainDataloader(
    #     dataset_names=dataset_names,
    #     transformation=data_aug.IJEPAAugmentation(),
    #     batch_size=BATCH_SIZE,
    #     num_workers=4,
    #     sampler=False  # For single GPU, sampler can be False
    # )
    train_loader = data_ld
    BATCH_SIZE = batch_size
    model, loss_fn = create_DRijepa()
    optim_obj = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_obj,
        T_max=300,
        eta_min=1e-6
    )
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        optim=optim_obj,
        scheduler=scheduler,
        device=device,
        max_ep=300,
        save_dir="checkpoint_single"
    )
    trainer.train()

def ddp_main_worker(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank} starting DDP training on {device}")

    # Initialize wandb only on rank 0 to avoid duplicate logs (or use rank-specific logging)
    if rank == 0:
        wandb.init(project="dri_jepa_ddp")

    BATCH_SIZE = 64  # per GPU batch size
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid"]
    uniform_data_ld = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=data_aug.IJEPAAugmentation(),
        batch_size=BATCH_SIZE,
        num_workers=4,
        sampler=False  # We'll override the sampler below
    )
    # Get the dataset from your data loader object if possible.
    # (Assuming uniform_data_ld has an attribute 'dataset')
    dataset = uniform_data_ld.dataset  
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=sampler)

    model, loss_fn = create_DRijepa()
    model = model.to(device)
    # Wrap the model in DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optim_obj = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_obj,
        T_max=300,
        eta_min=1e-6
    )
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        optim=optim_obj,
        scheduler=scheduler,
        device=device,
        max_ep=300,
        save_dir="checkpoint_ddp"
    )
    trainer.train()

    dist.destroy_process_group()


def train_ddp(args):
    world_size = 2  # Use 2 GPUs
    mp.spawn(ddp_main_worker, args=(world_size, args), nprocs=world_size, join=True)