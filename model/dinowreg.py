import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.distributed as dist


class RetAug:

    def __init__(self , img_size=512):
        
        self.base_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3,7)),
                A.GaussNoise(var_limit=(10.0, 50.0)),
            ], p=0.4),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            ToTensorV2()
        ])
        
        self.lesion_transform = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=7, alpha_affine=10, p=0.3),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.4),
                A.RandomSizedCrop(min_max_height=(32, 64), height=img_size, width=img_size, p=0.3)
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                          fill_value=0, mask_fill_value=0, p=0.5)
        ])
    
    def __call__(self, image):
        base_view = self.base_transform(image=image)['image']
        lesion_view = self.lesion_transform(image=image)['image']
        return base_view, lesion_view

class ViTRegs(nn.Module):

    def __init__(
            self,
            img_size=1024,
            patch_size = 16,
            in_chans = 3,
            embed_dim = 512,
            depth = 12,
            num_heads = 12,
            mlp_ratio = 4.,
            num_regs =4 
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_registers = num_regs


        self.patch_embed = nn.Conv2d(in_chans , embed_dim , kernel_size=patch_size , stride=patch_size)

        self.cls_tokens = nn.Parameter(torch.zeros(1,1, embed_dim))
        self.reg_tokens = nn.Parameter(torch.zeros(1 , num_regs , embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1 , self.num_patches + 1 , embed_dim) * .02)


        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim  *mlp_ratio),
                    activation="gelu",
                    batch_first=True
                ) for _ in range(depth)
            ]
        )

        self.proj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2 , embed_dim)
        )

        self.init_weight()

    def init_weight(self):
        nn.init.trunc_normal_(self.cls_tokens ,std=0.02)
        nn.init.trunc_normal_(self.reg_tokens , std=0.02)
        nn.init.trunc_normal_(self.pos_embed , std=0.02)


    def forward(self,x):

        B , C , H , W = x.shape

        x = self.patch_embed(x) # -> (B, D , H/p , W/p)

        x = x.flatten(2).transpose(1,2) # -> (B , N , D)


        cls_token = self.cls_tokens.expand(B , -1 , -1)
        x = torch.cat([cls_token , x] , dim=1)

        x += self.pos_embed


        reg_token = self.reg_tokens.expand(B , -1 , -1)
        x = torch.cat([x  ,reg_token] , dim=1)


        for blk in self.blocks:
            x = blk(x)

        cls_feat = x[: , 0]
        patch_feat = x[: , 1:-self.num_registers]
        reg_feat = x[: , -self.num_registers:]

        cls_feat = self.proj_head(cls_feat)
        patch_feat = self.proj_head(patch_feat)

        return cls_feat , patch_feat , reg_feat
    

class DINOwithReg(nn.Module):

    def __init__(
            self,
            student,
            teacher,
            embed_dim=768,
            out_dim=65536,
            temp_student=0.1,
            temp_teacher = 0.04,
            momentum = 0.996,
            center_momentum = 0.9,
            num_reg = 4
    ):
        
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.momentum = momentum
        self.center_momentum = center_momentum
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.num_registers = num_reg

       
        self._init_teacher()
        
        # Register buffers for teacher center
        self.register_buffer("teacher_center", torch.zeros(1, embed_dim))
        
        # Lesion-focused projection heads
        self.student_head = self._build_projection(embed_dim, out_dim)
        self.teacher_head = self._build_projection(embed_dim, out_dim)


    def _build_projection(self, embed_dim, out_dim):
        return nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def _init_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update teacher center"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.teacher_center = self.teacher_center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, x1 ,x2):

        s_cls1 , s_patch1 , s_reg1 = self.student(x1)
        s_cls2 , s_patch2 , s_reg2 = self.student(x2)

        with torch.no_grad():

            self.momentum_update()
            t_cls1 , t_patch1 , t_reg1 = self.teacher(x1)
            t_cls2, t_patch2, t_reg2 = self.teacher(x2)

        s1 = self.student_head(s_cls1)
        s2 = self.student_head(s_cls2)

        with torch.no_grad():
            t1 = self.teacher_head(t_cls2).detach()
            t2 = self.teacher_head(t_cls1).detach()
            
            # Center and sharpen teacher outputs
            t1 = F.softmax((t1 - self.teacher_center) / self.temp_teacher, dim=-1)
            t2 = F.softmax((t2 - self.teacher_center) / self.temp_teacher, dim=-1)
        
        # Compute cross-entropy loss
        loss1 = -torch.mean(torch.sum(t1 * F.log_softmax(s2 / self.temp_student, dim=-1), dim=-1))
        loss2 = -torch.mean(torch.sum(t2 * F.log_softmax(s1 / self.temp_student, dim=-1), dim=-1))
        total_loss = (loss1 + loss2) / 2

        # Update center
        self.update_center(torch.cat([t1, t2]))
        
        # Register consistency loss (novel component)
        reg_loss = F.mse_loss(s_reg1, t_reg2.detach()) + F.mse_loss(s_reg2, t_reg1.detach())
        total_loss += reg_loss * 0.1  # Weighted combination

        return total_loss
    
#######################
# for pytorch lightning module
#######################
import pytorch_lightning as pl

class DINOLoss(nn.Module):
    """Combined DINO + Register consistency loss"""
    def __init__(self, output_dim=65536, warmup_teacher_temp=0.04, 
                 teacher_temp=0.04, student_temp=0.1, reg_weight=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.reg_weight = reg_weight
        self.center = nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)
        
    def forward(self, student_output, teacher_output, register_outputs):
        """
        student_output: (B, D)
        teacher_output: (B, D)
        register_outputs: tuple of (s_reg, t_reg) each (B, N, D)
        """
        # Cross-entropy between softmax-normalized student and teacher outputs
        student_out = student_output / self.student_temp
        teacher_out = (teacher_output - self.center) / self.teacher_temp
        
        student_probs = F.log_softmax(student_out, dim=-1)
        teacher_probs = F.softmax(teacher_out.detach(), dim=-1)
        
        loss = -torch.mean(torch.sum(teacher_probs * student_probs, dim=-1))
        
        # Register consistency loss
        s_reg, t_reg = register_outputs
        reg_loss = F.mse_loss(s_reg, t_reg.detach())
        
        return loss + self.reg_weight * reg_loss
    

class RetinaDINOLightning(pl.LightningModule):
    def __init__(self, img_size=512, patch_size=16, embed_dim=768,
                 warmup_epochs=10, max_epochs=100, weight_decay=0.04,
                 momentum=0.996, batch_size=64, num_registers=4):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize student and teacher models
        self.student = ViTRegs(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_registers=num_registers
        )
        self.teacher = ViTRegs(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_registers=num_registers
        )
        
        # Initialize teacher from student
        self._init_teacher()
        
        # DINO loss with register consistency
        self.dino_loss = DINOLoss(
            output_dim=embed_dim,
            reg_weight=0.1
        )
        
        # Training parameters
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size

    def _init_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """Exponential moving average update for teacher"""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        x1, x2 = batch  # Two augmented views
        
        # Student outputs
        s_cls1, s_patch1, s_reg1 = self.student(x1)
        s_cls2, s_patch2, s_reg2 = self.student(x2)
        
        # Teacher outputs (no grad)
        with torch.no_grad():
            self.momentum_update()
            t_cls1, t_patch1, t_reg1 = self.teacher(x1)
            t_cls2, t_patch2, t_reg2 = self.teacher(x2)
        
        # Calculate losses for both views
        loss1 = self.dino_loss(s_cls1, t_cls2, (s_reg1, t_reg2))
        loss2 = self.dino_loss(s_cls2, t_cls1, (s_reg2, t_reg1))
        total_loss = (loss1 + loss2) / 2
        
        # Update center for teacher output
        with torch.no_grad():
            self.dino_loss.center = self.dino_loss.center * 0.9 + \
                                  torch.cat([t_cls1, t_cls2]).mean(dim=0) * 0.1
        
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.student.named_parameters():
            if 'bias' in name or 'norm' in name or 'register' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=0.0005 * (self.batch_size / 256), betas=(0.9, 0.95))
        
        # Cosine schedule with linear warmup
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=self.max_epochs - self.warmup_epochs,
                eta_min=1e-6
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.SequentialLR(
                    optim,
                    schedulers=[warmup_scheduler, lr_scheduler['scheduler']],
                    milestones=[self.warmup_epochs]
                ),
                'interval': 'epoch',
                'frequency': 1
            }

        return [optim], [lr_scheduler]

    def on_before_zero_grad(self, optimizer):
        # Sync centers across distributed processes
        if dist.is_initialized():
            dist.all_reduce(self.dino_loss.center, op=dist.ReduceOp.SUM)
            self.dino_loss.center /= dist.get_world_size()
