import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
from tqdm import tqdm
import os
from timm import create_model
from model.utils import swin_test_config

class DINOHead(nn.Module):

    def __init__(self , in_dim , out_dim , hidden_dim=2048 , bottleneck_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim , hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim , hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim , bottleneck_dim),
        )

        self.last_layer = nn.Linear(bottleneck_dim , out_dim)
        self.apply(self._init_weights)
    def _init_weights(self , m):
        if isinstance(m , nn.Linear):
            nn.init.trunc_normal_(m.weight , std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias , 0)

    def forward(self ,x):

        x = self.mlp(x)
        x = F.normalize(x , dim=-1)
        x = self.last_layer(x)
        return x
    

class Student(nn.Module):
    def __init__(self , encoder: nn.Module , head: nn.Sequential):
        super().__init__()

        self.encoder = encoder
        self.head = head


    def forward(self ,x):
        out_encoder = self.encoder(x)
        out_head = self.head(out_encoder)
        return out_head


class Teacher(nn.Module):
    def __init__(self , encoder: nn.Module , head: nn.Sequential , centering):
        super().__init__()
        self.encoder = encoder
        self.head = head

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = False
    
    def forward(self , x):
        out_enc = self.encoder(x)
        out_head = self.head(out_enc)
        return out_head




class DINOLoss(nn.Module):

    def __init__(self , out_dim , teach_temp=0.04 , student_temp=0.1 , center_mom=0.9):

        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teach_temp
        self.center_mom = center_mom

        self.register_buffer("center" , torch.zeros(1, out_dim).cuda())

    def forward(self , student_out , teacher_out):

        student_out = student_out / self.student_temp
        teacher_out = teacher_out / self.teacher_temp

        student_soft = F.softmax(student_out , dim=-1)
        teacher_soft = F.softmax((teacher_out - self.center) , dim=-1)

        loss = torch.sum(-teacher_soft * torch.log_softmax(student_out , dim=-1) , dim=-1)
        self.update_center(teacher_out)

        return loss.mean()
    @torch.no_grad()
    def update_center(self , teacher_out):

        batch_center = torch.sum(teacher_out , dim=0 , keepdim=True)
        batch_center = batch_center / len(teacher_out)

        self.center = self.center * self.center_mom + batch_center * (1 - self.center_mom)


class SwinDINO:
    def __init__(self, 
                 embed_dim=96, 
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 out_dim=65536,
                 hidden_dim=2048,
                 bottleneck_dim=256,
                 momentum_teacher=0.996,
                 use_wandb=True):
        
        # Initialize Swin Transformer as encoder
        self.encoder = create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,  # Remove classification head
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads
        )
        
        # Get feature dimension from encoder
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.encoder(dummy_input)
            in_dim = features.shape[1]
        
        self.head = DINOHead(in_dim=in_dim, 
                            out_dim=out_dim,
                            hidden_dim=hidden_dim,
                            bottleneck_dim=bottleneck_dim)
        
        self.student = Student(encoder=self.encoder, head=self.head).cuda()
        self.teacher = Teacher(encoder=self.encoder, head=self.head).cuda()
        
        # Initialize loss
        self.dino_loss = DINOLoss(out_dim=out_dim).cuda()
        
        # Training parameters
        self.momentum_teacher = momentum_teacher
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project="dino-swin", config={
                "embed_dim": embed_dim,
                "depths": depths,
                "num_heads": num_heads,
                "out_dim": out_dim,
                "hidden_dim": hidden_dim,
                "bottleneck_dim": bottleneck_dim,
                "momentum_teacher": momentum_teacher
            })

    def update_teacher(self):
        with torch.no_grad():
            for param_t, param_s in zip(self.teacher.parameters(), 
                                      self.student.parameters()):
                param_t.data = self.momentum_teacher * param_t.data + \
                              (1 - self.momentum_teacher) * param_s.data

    def visualize_attention(self, image, epoch):
        """Visualize attention maps from Swin Transformer"""
        self.student.encoder.eval()
        with torch.no_grad():
            # Get attention weights from last layer
            att_mat = []
            def hook_fn(module, input, output):
                att_mat.append(output.detach())
            
            # Register hook for the last attention layer
            for name, module in self.student.encoder.named_modules():
                if "attn" in name and "layers.3" in name:  # Last stage
                    module.register_forward_hook(hook_fn)
            
            # Forward pass
            _ = self.student.encoder(image.unsqueeze(0).cuda())
            
            # Average attention weights across heads
            att_map = att_mat[0].mean(1).mean(1)
            att_map = att_map.reshape(att_map.size(0), int(np.sqrt(att_map.size(1))), 
                                    int(np.sqrt(att_map.size(1))))
            
            # Convert to heatmap
            plt.figure(figsize=(10, 10))
            plt.imshow(att_map[0].cpu(), cmap='viridis')
            plt.axis('off')
            
            if self.use_wandb:
                wandb.log({f"attention_map_epoch_{epoch}": wandb.Image(plt)})
            
            plt.close()

    def train(self, dataloader, num_epochs, batch_size, optimizer, lr_scheduler=None):
        

        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, views in enumerate(progress_bar):
                views = [view.cuda() for view in views]
                global_views = views[:2]
                local_views = views[2:]
                all_views = global_views + local_views

                # Teacher forward passes
                with torch.no_grad():
                    teacher_output = [self.teacher(view) for view in global_views]
                
                # Student forward passes
                student_output = [self.student(view) for view in all_views]
                
                # Compute loss
                total_loss = 0
                n_loss_terms = 0
                for i, t_out in enumerate(teacher_output):
                    for j, s_out in enumerate(student_output):
                        if j == i:
                            continue
                        total_loss += self.dino_loss(s_out, t_out.detach())
                        n_loss_terms += 1
                
                loss = total_loss / n_loss_terms
                epoch_loss += loss.item()
                
                # Update student
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update teacher
                self.update_teacher()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Compute average epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # Visualize attention maps every 15 epochs
            if (epoch + 1) % 15 == 0 and len(views) > 0:
                self.visualize_attention(views[0][0], epoch + 1)
            
            # Update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, path, epoch, optimizer, lr_scheduler=None):
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None, lr_scheduler=None):
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        return checkpoint['epoch']
    


def single_gpu_train(dataloader , n_epochs ,batch_size):
    device = torch.device("cuda")
    wandb.init(
        project="custom_dino_training",
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

    model = SwinDINO(
        embed_dim=swin_test_config["embed_dim"],
        depths=swin_test_config["depths"],
        num_heads=swin_test_config["num_heads"],
        out_dim=65536,
        hidden_dim=2048,
        bottleneck_dim=256,
        use_wandb=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=0.0001,
        weight_decay=0.04
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6
    )

    # try:
    model.train(
        dataloader=dataloader,
        num_epochs=n_epochs,
        batch_size=batch_size,
        optimizer = optimizer,
        lr_scheduler = scheduler,
    )
    
    # Save final checkpoint
    model.save_checkpoint(
        os.path.join("checkpoints", 'final_checkpoint.pth'),
        epoch=n_epochs,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )
        
    # except KeyboardInterrupt:
    #     print("Training interrupted. Saving checkpoint...")
    #     model.save_checkpoint(
    #         os.path.join("checkpoints", 'interrupted_checkpoint.pth'),
    #         epoch=n_epochs,
    #         optimizer=optimizer,
    #         lr_scheduler=scheduler
    #     )
        
    # finally:
    wandb.finish()