import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.swin_transformer import SwinTransformer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import time

# wandb integration
try:
    import wandb
except ImportError:
    wandb = None
    print("Weights & Biases not installed. Install it via `pip install wandb` for wandb logging.")

from model.utils import vit_config, swin_test_config
from data_pipeline.data_aug import IbotRetAug
from data_pipeline import data_set, data_aug  # adjust these imports as needed

wandb.init(
        project="custom_ibot_training",
        config={
            "learning_rate": 1.5e-4,
            "weight_decay": 0.05,
            "num_epochs": 300,
            "batch_size": 32,
            "model": "CustomiBOT with MaskedSwin",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR"
        }
    )
###############################
# Model Definitions
###############################
class MaskedSwin(nn.Module):
    def __init__(
        self,
        img_size=1024,
        patch_size=4,
        in_chan=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mask_ratio=0.6,
        lesion_mask_prob=0.7
    ):
        super().__init__()
        self.swin = SwinTransformer(
            patch_size=(patch_size, patch_size),
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,    
            window_size=(window_size, window_size),
            mlp_ratio=4.0,
        )
        self.swin.head = nn.Identity()  # Remove classification head
        self.swin.avgpool = nn.Identity() 
        # Calculate final feature dimensions
        self.num_stages = len(depths)
        self.final_res = img_size // (patch_size * (2 ** (self.num_stages - 1)))
        self.num_patches = self.final_res ** 2
        self.final_dim = embed_dim * (2 ** (self.num_stages - 1))
        
        # Masking parameters
        self.mask_ratio = mask_ratio
        self.lesion_mask_prob = lesion_mask_prob
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.final_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Projection heads
        self.global_proj = nn.Sequential(
            nn.LayerNorm(self.final_dim),
            nn.Linear(self.final_dim, self.final_dim)
        )
        self.local_proj = nn.Sequential(
            nn.LayerNorm(self.final_dim),
            nn.Linear(self.final_dim, self.final_dim)
        )

    def lesion_aware_masking(self, x, attention_maps=None):
        B, N, D = x.shape
        num_mask = int(N * self.mask_ratio)
        
        # Generate hierarchical attention weights for medical images
        if attention_maps is None:
            # Create dummy attention with center bias
            grid_size = int(N ** 0.5)
            y_coord, x_coord = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
            center = grid_size // 2
            dist = ((x_coord - center)**2 + (y_coord - center)**2).float()
            attention = 1 - dist / dist.max()
            attention = attention.view(1, N).repeat(B, 1).to(x.device)
        else:
            attention = attention_maps
        
        rand_mask = torch.rand(B, N, device=x.device)
        combined_mask = (self.lesion_mask_prob * attention + 
                         (1 - self.lesion_mask_prob) * rand_mask)
        
        mask_idx = combined_mask.topk(num_mask, dim=1).indices
        mask = torch.zeros(B, N, device=x.device).scatter(1, mask_idx, 1)
        return mask.bool()

    def forward(self, x, mask_ratio=0.4):
        B = x.shape[0]
        
        
        features = self.swin.features(x)
        
        # Reshape features to have shape [B, num_patches, self.final_dim]
        x = features.reshape(B, self.final_dim, -1).transpose(1, 2)
        
        # Generate the lesion-aware mask on patch tokens.
        mask = self.lesion_aware_masking(x)
        
        # Apply masking: replace masked tokens with the learnable mask token.
        mask_token = self.mask_token.expand(B, x.size(1), -1)
        x = x * ((~mask).float().unsqueeze(-1)) + mask_token * mask.float().unsqueeze(-1)
        
        # Add global token: compute a CLS token as the mean of patch tokens.
        cls_token = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([cls_token, x], dim=1)
        
        # Project features:
        global_feat = self.global_proj(x[:, 0])
        local_feat = self.local_proj(x[:, 1:])
        
        return global_feat, local_feat, mask

class CustomiBOT(nn.Module):
    def __init__(
        self,
        student=None,
        teacher=None,
        embed_dim=1024,
        temp=0.1,
        mask_ratio=0.6,
        momentum=0.996,
        swin_config=None
    ):
        super().__init__()
        default_swin_config = {
            'img_size': swin_test_config["img_size"],
            'patch_size': swin_test_config["patch_size"],
            'in_chan': 3,
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': 7
        }
        
        self.student = student if student else MaskedSwin(**(swin_config or default_swin_config))
        self.teacher = teacher if teacher else MaskedSwin(**(swin_config or default_swin_config))
        self.momentum = momentum
        self.temp = temp
        self.mask_ratio = mask_ratio

        # Initialize teacher with student weights
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

        # Projection heads
        self.global_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
        self.local_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )

    @torch.no_grad()
    def momentum_update(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

    def compute_loss(self, student_global, student_local, teacher_global, teacher_local, mask):
        # Global loss
        global_loss = F.cosine_embedding_loss(
            self.global_head(student_global),
            teacher_global.detach(),
            torch.ones(student_global.size(0), device=student_global.device)
        )
        
        # Local loss with medical image specific weighting
        masked_student = student_local[mask]
        masked_teacher = teacher_local[mask]
        local_loss = F.cosine_embedding_loss(
            self.local_head(masked_student),
            masked_teacher.detach(),
            torch.ones(masked_student.size(0), device=student_local.device)
        )
        
        return global_loss + 0.5 * local_loss

    def forward(self, x1, x2):
        # Student predictions
        s_global1, s_local1, mask1 = self.student(x1)
        s_global2, s_local2, mask2 = self.student(x2)
        
        # Teacher predictions
        with torch.no_grad():
            t_global1, t_local1, _ = self.teacher(x1)
            t_global2, t_local2, _ = self.teacher(x2)
        
        # Compute loss
        loss = 0.5 * (self.compute_loss(s_global1, s_local1, t_global2, t_local2, mask1) +
                      self.compute_loss(s_global2, s_local2, t_global1, t_local1, mask2))
        
        # Update teacher
        self.momentum_update()
        return loss

#######################################
# Training Functions with Fancy Logging
#######################################
def train_single_gpu(model, train_loader, num_epochs, val_loader=None, wandb_enabled=False):
    device = torch.device("cuda")
    model.to(device)
    
    # Create optimizer with parameter groups for weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.05},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=1.5e-4, betas=(0.9, 0.95))
    
    # Scheduler: Cosine Annealing (using num_epochs - 10 as T_max)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - 10, eta_min=1e-6)
  
    print("==============================================")
    print("Starting Single GPU Training")
    print("==============================================")
    global_step = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x1, x2) in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            loss = model(x1, x2)  # forward pass (loss computed internally)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            # Update tqdm description every 50 steps
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.6f}"
                })
        
        scheduler.step()
        epoch_train_loss = running_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start

        # Run validation if provided
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x1, x2 in val_loader:
                    x1, x2 = x1.to(device), x2.to(device)
                    loss = model(x1, x2)
                    val_loss += loss.item()
            epoch_val_loss = val_loss / len(val_loader)
        else:
            epoch_val_loss = None

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": epoch_duration
        }
        if epoch_val_loss is not None:
            log_dict["val_loss"] = epoch_val_loss

        # Fancy logging to console
        log_msg = f"Epoch [{epoch+1}/{num_epochs}] | LR: {log_dict['lr']:.6f} | "
        log_msg += f"Train Loss: {log_dict['train_loss']:.4f}"
        if epoch_val_loss is not None:
            log_msg += f" | Val Loss: {log_dict['val_loss']:.4f}"
        log_msg += f" | Time: {epoch_duration:.2f}s"
        print(log_msg)

        # Log metrics to wandb if enabled
        if wandb_enabled and wandb is not None:
            wandb.log(log_dict, step=global_step)

def train_multi_gpu(model, train_loader, val_loader, num_epochs, device, device_ids=[0, 1], wandb_enabled=False):
    # Wrap the model with DataParallel and move to the primary GPU.
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)  # typically device will be "cuda:0"
    
    decay_params = []
    no_decay_params = []
    for name, param in model.module.named_parameters():  # note the .module when using DataParallel
        if 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.05},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=1.5e-4, betas=(0.9, 0.95))
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - 10, eta_min=1e-6)
  
    print("==============================================")
    print("Starting Multi GPU Training")
    print("==============================================")
    global_step = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x1, x2) in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            loss = model(x1, x2)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.6f}"
                })
        
        scheduler.step()
        epoch_train_loss = running_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2 in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                loss = model(x1, x2)
                val_loss += loss.item()
        epoch_val_loss = val_loss / len(val_loader)
        
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": epoch_duration
        }
        log_msg = f"Epoch [{epoch+1}/{num_epochs}] | LR: {log_dict['lr']:.6f} | Train Loss: {log_dict['train_loss']:.4f} | Val Loss: {log_dict['val_loss']:.4f} | Time: {epoch_duration:.2f}s"
        print(log_msg)
        
        if wandb_enabled and wandb is not None:
            wandb.log(log_dict, step=global_step)

#######################################
# Main Training Script
#######################################
if __name__ == "__main__":
    
    
        
    
    # Define your dataset names and create dataloaders using your existing pipeline.
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid"]
    train_loader = data_set.SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=data_aug.IJEPAAugmentation(),
        batch_size=32,
        num_workers=4,
    ).get_loader()
    
    val_loader = data_set.SSLValidLoader(
        dataset_names=dataset_names,
        transformation=None,
        batch_size=32,
        num_workers=4,
    ).get_loader()
    
    # Swin configuration for your model (adjust as needed)
    swin_config = {
        'img_size':224,
        'patch_size': 4,
        'in_chan': 3,
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'window_size': 7,
        'mask_ratio': 0.6
    }
    
    # Initialize the iBOT model with Swin backbone.
    ibot_model = CustomiBOT(
        swin_config=swin_config,
        embed_dim=1024,
        momentum=0.996
    )
    
    num_epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_single_gpu(ibot_model , train_loader=train_loader , num_epochs=num_epochs)
    
    wandb.finish()
