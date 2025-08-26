import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.distributed as dist
import torch.multiprocessing as mp


from data_pipeline.data_aug import DinowregAug 
from data_pipeline.data_set import UniformTrainDataloader , SSLTrainLoader ,SSLValidLoader
from train.utils import vit_config


class ViTRegs(nn.Module):
    def __init__(
            self,
            img_size=vit_config["img_size"],
            patch_size=vit_config["patch_size"],
            in_chans=vit_config["in_chans"],
            embed_dim=vit_config["embed_dim"],
            depth=vit_config["depth"],
            num_heads=vit_config["num_heads"],
            mlp_ratio=vit_config["mlp_ratio"],
            num_regs=vit_config["num_regs"] 
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_registers = num_regs

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_tokens = nn.Parameter(torch.zeros(1, num_regs, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * .02)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    activation="gelu",
                    batch_first=True
                ) for _ in range(depth)
            ]
        )

        self.proj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.init_weight()

    def init_weight(self):
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        nn.init.trunc_normal_(self.reg_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # -> (B, D, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # -> (B, N, D)

        cls_token = self.cls_tokens.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embed

        reg_token = self.reg_tokens.expand(B, -1, -1)
        x = torch.cat([x, reg_token], dim=1)

        for blk in self.blocks:
            x = blk(x)

        cls_feat = x[:, 0]
        patch_feat = x[:, 1:-self.num_registers]
        reg_feat = x[:, -self.num_registers:]

        cls_feat = self.proj_head(cls_feat)
        patch_feat = self.proj_head(patch_feat)

        return cls_feat, patch_feat, reg_feat


class DINOLoss(nn.Module):
    """Combined DINO + register consistency loss."""
    def __init__(self, output_dim=65536, warmup_teacher_temp=0.04, 
                 teacher_temp=0.04, student_temp=0.1, reg_weight=0.1):
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
    
class RetinaDINO(nn.Module):
    def __init__(
            self, 
            img_size=512, 
            patch_size=32, 
            embed_dim=1024,
            momentum=0.9996, 
            num_registers=8
            ):
        super().__init__()
        # Update vit_config with the given hyperparameters.
        vit_config.update({
            "img_size": img_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depth": 12, 
            "num_heads": 16,  
            "mlp_ratio": 4,
            "num_regs": num_registers
        })
        
        # Initialize student and teacher networks.
        self.student = ViTRegs()
        self.teacher = ViTRegs()
        self._init_teacher()
        
        # Initialize the DINO loss (including register consistency)
        self.dino_loss = DINOLoss(
            output_dim=embed_dim,
            warmup_teacher_temp=0.04,
            teacher_temp=0.05,
            student_temp=0.1,  
            reg_weight=0.15  
        )
        
        self.momentum = momentum  # for teacher momentum update

    def _init_teacher(self):
        # Initialize teacher with student weights and disable grad.
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """Exponential moving average update for teacher."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

    def forward(self, x):
        return self.student(x)


def train_single_gpu(train_dl , b_size , autocast=False):

# ---------------------------------------------------------------------------
# INIT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters (you can also read these from args)
    img_size = vit_config["img_size"]
    patch_size = vit_config["patch_size"]
    embed_dim = vit_config["embed_dim"]
    warmup_epochs = 5
    max_epochs = 50
    weight_decay = 0.05
    momentum = 0.9996
    batch_size = b_size
    num_registers = 8

    # Initialize the model and move to device.
    model = RetinaDINO(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        momentum=momentum,
        num_registers=vit_config["num_regs"]
    ).to(device)

# ---------------------------------------------------------------------------
#   OPTIMIZERSS
    decay_params = []
    no_decay_params = []
    for name, param in model.student.named_parameters():
        if 'bias' in name or 'norm' in name or 'register' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    base_lr = 0.0003 * (batch_size / 256)
    # base_lr = 0.0003
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=base_lr, betas=(0.9, 0.999))
    
    # Learning rate scheduler: warmup + cosine annealing.
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
    

# ---------------------------------------------------------------------------
    # DATALOADER
    train_loader = train_dl

# ---------------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler()

# ---------------------------------------------------------------------------
# TRAINING

    print("Starting single-GPU training...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            # 2 views
            x1, x2 = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            optimizer.zero_grad()
            if autocast:
                with torch.cuda.amp.autocast():
                    # Student forward pass.
                    s_cls1, s_patch1, s_reg1 = model.student(x1)
                    s_cls2, s_patch2, s_reg2 = model.student(x2)
                    
                    # Teacher forward pass (with no gradient).
                    with torch.no_grad():
                        model.momentum_update()
                        t_cls1, t_patch1, t_reg1 = model.teacher(x1)
                        t_cls2, t_patch2, t_reg2 = model.teacher(x2)
                    
                    loss1 = model.dino_loss(s_cls1, t_cls2, (s_reg1, t_reg2))
                    loss2 = model.dino_loss(s_cls2, t_cls1, (s_reg2, t_reg1))
                    loss = (loss1 + loss2) / 2

                    # Update teacher center.
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
                
                # Teacher forward pass (with no gradient).
                with torch.no_grad():
                    model.momentum_update()
                    t_cls1, t_patch1, t_reg1 = model.teacher(x1)
                    t_cls2, t_patch2, t_reg2 = model.teacher(x2)
                
                loss1 = model.dino_loss(s_cls1, t_cls2, (s_reg1, t_reg2))
                loss2 = model.dino_loss(s_cls2, t_cls1, (s_reg2, t_reg1))
                loss = (loss1 + loss2) / 2

                # Update teacher center.
                with torch.no_grad():
                    teacher_cat = torch.cat([t_cls1, t_cls2], dim=0)
                    model.dino_loss.center = model.dino_loss.center * 0.9 + \
                        teacher_cat.mean(dim=0, keepdim=True) * 0.1
                    
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            # if i % 50 == 0:
            print(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
        # scheduler.step()
        print(f"Epoch {epoch} Average Loss: {epoch_loss/len(train_loader):.4f}")


        

    print("Single-GPU training completed.")


def ddp_main_worker(rank, world_size, args):
    # Initialize the process group.
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    if rank == 0:
        print(f"Running DDP training on {world_size} GPUs.")

    # Hyperparameters (same as before; adjust if needed)
    img_size = 512
    patch_size = 32
    embed_dim = 1024
    warmup_epochs = 5
    max_epochs = 50
    weight_decay = 0.05
    momentum = 0.9996
    batch_size = 32  # (batch size per GPU; if you want a global batch size, adjust accordingly)
    num_registers = 8

    # Initialize model.
    model = RetinaDINO(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        weight_decay=weight_decay,
        momentum=momentum,
        batch_size=batch_size,
        num_registers=num_registers
    ).to(device)

    # Wrap the student network in DDP. (Teacher is not updated by gradient, so we leave it as is.)
    model.student = torch.nn.parallel.DistributedDataParallel(
        model.student, device_ids=[rank], output_device=rank
    )

    # Build optimizer on the student’s parameters.
    decay_params = []
    no_decay_params = []
    # Note: when using DDP, use model.student.module.named_parameters() to access the original module.
    for name, param in model.student.module.named_parameters():
        if 'bias' in name or 'norm' in name or 'register' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    base_lr = 0.0003 * (batch_size / 256)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=base_lr, betas=(0.9, 0.999))
    
    # Set up the LR scheduler (same as single-GPU).
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

    # Data loader: use a distributed sampler.
    augmentor = DinowregAug(img_size=vit_config['img_size'])
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid"]
    uniform_data_ld = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=augmentor,
        batch_size=batch_size,
        num_workers=2,
        sampler=True  # Enable distributed sampler
    )
    train_loader = uniform_data_ld.get_loader()
    # If your dataloader returns a DistributedSampler, then you can call set_epoch on it.
    sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None

    scaler = torch.cuda.amp.GradScaler()

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
                # Student forward pass (via DDP-wrapped module).
                s_cls1, s_patch1, s_reg1 = model.student(x1)
                s_cls2, s_patch2, s_reg2 = model.student(x2)
                # Teacher forward pass.
                with torch.no_grad():
                    model.momentum_update()
                    t_cls1, t_patch1, t_reg1 = model.teacher(x1)
                    t_cls2, t_patch2, t_reg2 = model.teacher(x2)
                loss1 = model.dino_loss(s_cls1, t_cls2, (s_reg1, t_reg2))
                loss2 = model.dino_loss(s_cls2, t_cls1, (s_reg2, t_reg1))
                loss = (loss1 + loss2) / 2

                # Update teacher center and synchronize across GPUs.
                with torch.no_grad():
                    teacher_cat = torch.cat([t_cls1, t_cls2], dim=0)
                    center_update = teacher_cat.mean(dim=0, keepdim=True)
                    model.dino_loss.center = model.dino_loss.center * 0.9 + center_update * 0.1
                    # Synchronize the center across processes.
                    dist.all_reduce(model.dino_loss.center, op=dist.ReduceOp.SUM)
                    model.dino_loss.center /= dist.get_world_size()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            if i % 50 == 0 and rank == 0:
                print(f"Rank {rank} | Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
        scheduler.step()
        if rank == 0:
            print(f"Rank {rank} | Epoch {epoch} Average Loss: {epoch_loss/len(train_loader):.4f}")
    
    if rank == 0:
        print("DDP training completed.")
    dist.destroy_process_group()

def train_ddp(args):
    # Set the world size to the number of GPUs you wish to use.
    world_size = 2  # For example, 2 GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(ddp_main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    augmentor = DinowregAug(img_size=vit_config['img_size'])
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid" , "messdr"]


    train_loader = SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=augmentor,
        batch_size=32,
        num_work=4,
    )

    data_ld = train_loader.get_loader()

    train_single_gpu(train_dl=data_ld , b_size=32)
