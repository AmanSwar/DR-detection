import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.swin_transformer import SwinTransformer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb  # <-- Import wandb

from data_pipeline.data_aug import DinowregAug 
from data_pipeline.data_set import SSLTrainLoader, SSLValidLoader , DistSSLTrainLoader , DistSSLValidLoader
from model.utils import vit_config, vit_test_config, swin_test_config, swin_config

def get_mem_info():
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    print("\n")
    print(f"Total GPU memory: {total} bytes")
    print(f"Free GPU memory: {free} bytes")
    print("\n")

get_mem_info()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def validate_model(model, valid_loader, device):
    """Run a validation loop and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in valid_loader:
            x1, x2 = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            # Forward passes for both student and teacher
            s_cls1, s_patch1, s_reg1 = model.student(x1)
            s_cls2, s_patch2, s_reg2 = model.student(x2)
            t_cls1, t_patch1, t_reg1 = model.teacher(x1)
            t_cls2, t_patch2, t_reg2 = model.teacher(x2)
            loss1 = model.dino_loss(s_cls1, t_cls2, (s_reg1, t_reg2))
            loss2 = model.dino_loss(s_cls2, t_cls1, (s_reg2, t_reg1))
            loss = (loss1 + loss2) / 2
            total_loss += loss.item()
            count += 1
    avg_loss = total_loss / count if count > 0 else 0.0
    return avg_loss

def save_checkpoint(model, optimizer, epoch, filename):
    """Save a checkpoint of the current model and optimizer state."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------
class SwinRegs(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.0,
        num_regs=8,
        num_transformer_layers=4,
        transformer_embed_dim=1024,
        transformer_heads=16,
    ):
        super().__init__()
        # Swin backbone configuration
        self.swin = SwinTransformer(
            patch_size=(patch_size, patch_size),
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,    
            window_size=(window_size, window_size),
            mlp_ratio=mlp_ratio,
        )
        self.swin.head = nn.Identity()  # Remove classification head
        
        # Calculate final feature dimensions
        num_stages = len(depths)
        self.final_res = img_size // (patch_size * (2 ** (num_stages - 1)))
        self.num_patches = (self.final_res) ** 2
        self.final_dim = embed_dim * (2 ** (num_stages - 1))
        
        # Projection to transformer dimension
        self.proj = nn.Linear(self.final_dim, transformer_embed_dim)
        
        # Tokens and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_embed_dim))
        self.reg_tokens = nn.Parameter(torch.zeros(1, num_regs, transformer_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1 + num_regs, transformer_embed_dim) * 0.02
        )
        
        # Transformer layers
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=transformer_embed_dim,
                nhead=transformer_heads,
                dim_feedforward=int(transformer_embed_dim * mlp_ratio),
                activation="gelu",
                batch_first=True
            ) for _ in range(num_transformer_layers)
        ])

        # Projection head
        self.proj_head = nn.Sequential(
            nn.LayerNorm(transformer_embed_dim),
            nn.Linear(transformer_embed_dim, transformer_embed_dim * 2),
            nn.GELU(),
            nn.Linear(transformer_embed_dim * 2, transformer_embed_dim)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.reg_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        
        # Swin backbone features
        features = self.swin.features(x) 
        x = features.reshape(B, self.final_dim, -1).transpose(1, 2)
        x = self.proj(x)  # (B, num_patches, transformer_dim)
        
        # Add tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.reg_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x, reg_tokens], dim=1)
        x += self.pos_embed
        
        # Transformer processing
        for blk in self.blocks:
            x = blk(x)
            
        # Split features
        cls_feat = x[:, 0]
        patch_feat = x[:, 1:-self.reg_tokens.size(1)]
        reg_feat = x[:, -self.reg_tokens.size(1):]

        # Project features
        cls_feat = self.proj_head(cls_feat)
        patch_feat = self.proj_head(patch_feat)

        return cls_feat, patch_feat, reg_feat

class RetinaDINO(nn.Module):
    def __init__(
        self,
        img_size=swin_test_config["img_size"],
        patch_size=swin_test_config["patch_size"],
        embed_dim=swin_test_config["embed_dim"],
        depths=swin_test_config["depths"],
        num_heads=swin_test_config["num_heads"],
        window_size=swin_test_config["window_size"],
        momentum=0.9996,
        num_registers=swin_test_config["num_regs"],
    ):
        super().__init__()
        
        self.student = SwinRegs(**swin_test_config)
        self.teacher = SwinRegs(**swin_test_config)
        self._init_teacher()
        
        self.dino_loss = DINOLoss(
            output_dim=swin_test_config["transformer_embed_dim"],
            warmup_teacher_temp=0.04,
            teacher_temp=0.05,
            student_temp=0.1,
            reg_weight=0.15
        )
        self.momentum = momentum

    def _init_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

    def forward(self, x):
        return self.student(x)
    
class DINOLoss(nn.Module):
    """Combined DINO + register consistency loss."""
    def __init__(self, output_dim=65536, warmup_teacher_temp=0.04, 
                 teacher_temp=0.07, student_temp=0.1, reg_weight=0.05):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.reg_weight = reg_weight
        # The teacher center is stored as a parameter (but not trained by gradient descent)
        self.register_buffer("center", torch.zeros(1, output_dim))

    def forward(self, student_output, teacher_output, register_outputs):
        """
        student_output: (B, D)
        teacher_output: (B, D)
        register_outputs: tuple of (s_reg, t_reg) each (B, N, D)
        """
        student_out = student_output / self.student_temp
        teacher_out = (teacher_output - self.center) / self.teacher_temp
        
        student_probs = F.log_softmax(student_out, dim=-1)
        teacher_probs = F.softmax(teacher_out.detach(), dim=-1)
        
        loss = -torch.mean(torch.sum(teacher_probs * student_probs, dim=-1))
        
        # Register consistency loss
        s_reg, t_reg = register_outputs
        reg_loss = F.mse_loss(s_reg, t_reg.detach())
        
        return loss + self.reg_weight * reg_loss

# -----------------------------------------------------------------------------
# Single-GPU training with wandb logging, validation & checkpoint support
# -----------------------------------------------------------------------------
def train_single_gpu(train_dl, valid_dl, b_size, max_epoch, autocast=False):
    # -------------------------------------------------------------------------
    # INIT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    img_size = swin_config["img_size"]
    patch_size = swin_config["patch_size"]
    embed_dim = swin_config["embed_dim"]
    warmup_epochs = 2
    max_epochs = max_epoch  # use provided max_epoch
    weight_decay = 0.04
    momentum = 0.9998
    batch_size = b_size

    # Initialize wandb run
    wandb.init(
        project="retina-dino",
        config={
            "img_size": img_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "warmup_epochs": warmup_epochs,
            "max_epochs": max_epochs,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "batch_size": batch_size,
            "base_lr": 0.0001,
            "autocast": autocast
        },
        name=f"single_gpu_run_{wandb.util.generate_id()}"
    )

    # Initialize the model and move to device.
    model = RetinaDINO(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        momentum=momentum,
        num_registers=swin_config["num_regs"]
    ).to(device)

    wandb.watch(model.student, log="all", log_freq=100)

    # -------------------------------------------------------------------------
    # Optimizer & Scheduler
    decay_params = []
    no_decay_params = []
    for name, param in model.student.named_parameters():
        if 'bias' in name or 'norm' in name or 'register' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    base_lr = 0.0001
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=base_lr, betas=(0.9, 0.999))
    
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(max_epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )
    
    # -------------------------------------------------------------------------
    # DATALOADERS
    train_loader = train_dl
    valid_loader = valid_dl

    scaler = torch.cuda.amp.GradScaler()

    # -------------------------------------------------------------------------
    # TRAINING LOOP (with KeyboardInterrupt support)
    print("Starting single-GPU training...")
    global_step = 0
    try:
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0
            for i, batch in enumerate(train_loader):
                x1, x2 = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                
                optimizer.zero_grad()
                if autocast:
                    with torch.cuda.amp.autocast():
                        s_cls1, s_patch1, s_reg1 = model.student(x1)
                        s_cls2, s_patch2, s_reg2 = model.student(x2)
                        with torch.no_grad():
                            model.momentum_update()
                            t_cls1, t_patch1, t_reg1 = model.teacher(x1)
                            t_cls2, t_patch2, t_reg2 = model.teacher(x2)
                        loss1 = model.dino_loss(s_cls1, t_cls2, (s_reg1, t_reg2))
                        loss2 = model.dino_loss(s_cls2, t_cls1, (s_reg2, t_reg1))
                        loss = (loss1 + loss2) / 2
                        with torch.no_grad():
                            teacher_cat = torch.cat([t_cls1, t_cls2], dim=0)
                            model.dino_loss.center = model.dino_loss.center * 0.9 + \
                                teacher_cat.mean(dim=0, keepdim=True) * 0.1
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    s_cls1, s_patch1, s_reg1 = model.student(x1)
                    s_cls2, s_patch2, s_reg2 = model.student(x2)
                    with torch.no_grad():
                        model.momentum_update()
                        t_cls1, t_patch1, t_reg1 = model.teacher(x1)
                        t_cls2, t_patch2, t_reg2 = model.teacher(x2)
                    loss1 = model.dino_loss(s_cls1, t_cls2, (s_reg1, t_reg2))
                    loss2 = model.dino_loss(s_cls2, t_cls1, (s_reg2, t_reg1))
                    loss = (loss1 + loss2) / 2
                    with torch.no_grad():
                        teacher_cat = torch.cat([t_cls1, t_cls2], dim=0)
                        model.dino_loss.center = model.dino_loss.center * 0.9 + \
                            teacher_cat.mean(dim=0, keepdim=True) * 0.1
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": i,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "global_step": global_step
                })
                print(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

            scheduler.step()
            avg_epoch_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch})
            print(f"Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}")

            # Run validation at the end of each epoch.
            val_loss = validate_model(model, valid_loader, device)
            wandb.log({"validation_loss": val_loss, "epoch": epoch})
            print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

            # Save checkpoint after each epoch.
            save_checkpoint(model, optimizer, epoch, f"checkpoint_epoch_{epoch}.pt")

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Saving checkpoint before exiting.")
        save_checkpoint(model, optimizer, epoch, f"checkpoint_interrupt_epoch_{epoch}.pt")
        raise

    print("Single-GPU training completed.")
    wandb.finish()

# -----------------------------------------------------------------------------
# DDP training with wandb logging, validation & checkpoint support (rank 0 logs)
# -----------------------------------------------------------------------------
def ddp_main_worker(rank, world_size):
    # Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Use the same hyperparameters as single-GPU training.
    img_size = swin_config["img_size"]
    patch_size = swin_config["patch_size"]
    embed_dim = swin_config["embed_dim"]
    warmup_epochs = 2
    # Use the provided max_epoch from args (or default to 300)
    max_epochs = 300
    weight_decay = 0.04
    momentum = 0.9998
    batch_size = 32  # per GPU; same as single-GPU batch size
    num_registers = swin_config["num_regs"]
    base_lr = 0.0001

    # Only rank 0 initializes wandb.
    if rank == 0:
        print(f"Running DDP training on {world_size} GPUs.")
        wandb.init(
            project="retina-dino",
            config={
                "img_size": img_size,
                "patch_size": patch_size,
                "embed_dim": embed_dim,
                "warmup_epochs": warmup_epochs,
                "max_epochs": max_epochs,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "batch_size": batch_size,
                "base_lr": base_lr,
                "autocast": True
            },
            name=f"ddp_run_{wandb.util.generate_id()}"
        )

    # Initialize model.
    model = RetinaDINO(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        momentum=momentum,
        num_registers=num_registers
    ).to(device)

    # Wrap only the student with DDP.
    model.student = torch.nn.parallel.DistributedDataParallel(
        model.student, device_ids=[rank], output_device=rank
    )

    # Setup optimizer (using the same parameter grouping as in single-GPU training).
    decay_params = []
    no_decay_params = []
    for name, param in model.student.module.named_parameters():
        if 'bias' in name or 'norm' in name or 'register' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=base_lr, betas=(0.9, 0.999))
    
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(max_epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )

    # Prepare dataloaders using the same settings as single-GPU training.
    augmentor = DinowregAug(img_size=img_size)
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]

    # Use SSLTrainLoader (which should support distributed sampling when sampler=True)
    train_loader = DistSSLTrainLoader(
        dataset_names=dataset_names,
        img_size=swin_config["img_size"],
        batch_size=batch_size,
        num_work=4,
        world_size=world_size
    ).get_loader()
    # If a DistributedSampler is used, set the epoch each time.
    sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None

    valid_loader = DistSSLValidLoader(
        dataset_names=dataset_names,
        img_size=swin_config["img_size"],
        batch_size=8,
        num_work=4,
        world_size=world_size,
    ).get_loader()

    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    try:
        for epoch in range(max_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            for i, batch in enumerate(train_loader):
                x1, x2 = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    s_cls1, s_patch1, s_reg1 = model.student(x1)
                    s_cls2, s_patch2, s_reg2 = model.student(x2)
                    with torch.no_grad():
                        model.momentum_update()
                        t_cls1, t_patch1, t_reg1 = model.teacher(x1)
                        t_cls2, t_patch2, t_reg2 = model.teacher(x2)
                    loss1 = model.dino_loss(s_cls1, t_cls2, (s_reg1, t_reg2))
                    loss2 = model.dino_loss(s_cls2, t_cls1, (s_reg2, t_reg1))
                    loss = (loss1 + loss2) / 2
                    with torch.no_grad():
                        teacher_cat = torch.cat([t_cls1, t_cls2], dim=0)
                        center_update = teacher_cat.mean(dim=0, keepdim=True)
                        model.dino_loss.center = model.dino_loss.center * 0.9 + center_update * 0.1
                        # Synchronize the center across GPUs
                        dist.all_reduce(model.dino_loss.center, op=dist.ReduceOp.SUM)
                        model.dino_loss.center /= dist.get_world_size()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                global_step += 1

                if i % 50 == 0 and rank == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "batch": i,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "global_step": global_step
                    })
                    print(f"Rank {rank} | Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
            scheduler.step()
            if rank == 0:
                avg_epoch_loss = epoch_loss / len(train_loader)
                wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch})
                print(f"Rank {rank} | Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}")
                val_loss = validate_model(model, valid_loader, device)
                wandb.log({"validation_loss": val_loss, "epoch": epoch})
                print(f"Rank {rank} | Epoch {epoch} Validation Loss: {val_loss:.4f}")
                save_checkpoint(model, optimizer, epoch, f"ddp_checkpoint_epoch_{epoch}.pt")
    except KeyboardInterrupt:
        if rank == 0:
            print("KeyboardInterrupt caught on DDP. Saving checkpoint before exiting.")
            save_checkpoint(model, optimizer, epoch, f"ddp_checkpoint_interrupt_epoch_{epoch}.pt")
        raise

    if rank == 0:
        print("DDP training completed.")
        wandb.finish()
    dist.destroy_process_group()

def train_ddp():
    # Adjust world_size as needed (here we use 2 GPUs as an example)
    world_size = 1 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(ddp_main_worker, args=(world_size , ), nprocs=world_size, join=True)

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create data augmentations and dataloaders.
    augmentor = DinowregAug(img_size=swin_config['img_size'])
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]

    train_loader = DistSSLTrainLoader(
        dataset_names=dataset_names,
        img_size=swin_config['img_size'],
        batch_size=48,
        num_work=4,
    ).get_loader()

    valid_loader = DistSSLValidLoader(
        dataset_names=dataset_names,
        img_size=swin_config["img_size"],
        batch_size=8,
        num_work=4,
    ).get_loader()

    get_mem_info()
    max_epoch = 300
    # train_single_gpu(train_dl=train_loader, valid_dl=valid_loader, b_size=48, max_epoch=max_epoch)
    # To run DDP training instead, call train_ddp(args)
    
    
    train_ddp()
