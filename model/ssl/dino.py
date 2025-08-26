import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as transforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import albumentations as A
import wandb



class Student(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        features = self.encoder(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        out = self.head(features)
        return out

class Teacher(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, centering=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        # Freeze teacher parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        out = self.head(features)
        return out

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teach_temp=0.04, student_temp=0.1, center_mom=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teach_temp
        self.center_mom = center_mom
        # Initialize center as a buffer
        self.register_buffer("center", torch.zeros(1, out_dim).cuda())

    def forward(self, student_out, teacher_out):
        # Temperature scaling
        student_out = student_out / self.student_temp
        teacher_out = teacher_out / self.teacher_temp

        student_soft = F.softmax(student_out, dim=-1)
        teacher_soft = F.softmax((teacher_out - self.center), dim=-1)
        loss = torch.sum(-teacher_soft * F.log_softmax(student_out, dim=-1), dim=-1)
        self.update_center(teacher_out)
        return loss.mean()

    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = torch.sum(teacher_out, dim=0, keepdim=True)
        batch_center = batch_center / teacher_out.shape[0]
        self.center = self.center * self.center_mom + batch_center * (1 - self.center_mom)

class DINO:
    def __init__(self, encoder: nn.Module, head: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module):
      
        self.Student = Student(encoder=encoder, head=head).cuda()
        self.Teacher = Teacher(encoder=copy.deepcopy(encoder), head=copy.deepcopy(head), centering=None).cuda()
        self.optim = optimizer
        self.loss_fn = loss_fn

    @torch.no_grad()
    def update_teacher(self, m=0.996):
        # Update teacher with exponential moving average of student weights.
        for param_s, param_t in zip(self.Student.parameters(), self.Teacher.parameters()):
            param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

    def train_loop(self, train_loader, val_loader, num_epoch):
        for epoch in range(num_epoch):
            self.Student.train()  # Student is trained; teacher is updated via EMA.
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                # batch is a list of augmented views per image
                # Move each view to GPU.
                views = [view.cuda(non_blocking=True) for view in batch]
                # For training, use the first two views as global views and the remaining as local views.
                global_views = views[:2]
                local_views = views[2:] if len(views) > 2 else []
                all_views = global_views + local_views

                # Compute teacher outputs for global views (without gradient)
                with torch.no_grad():
                    teacher_outs = [self.Teacher(view) for view in global_views]

                # Compute student outputs for all views
                student_outs = [self.Student(view) for view in all_views]

                loss = 0
                n_loss_term = 0
                for i, t_out in enumerate(teacher_outs):
                    for j, s_out in enumerate(student_outs):
                        # Avoid comparing the same view (if they overlap)
                        if j == i:
                            continue
                        loss += self.loss_fn(s_out, t_out.detach())
                        n_loss_term += 1
                if n_loss_term > 0:
                    loss = loss / n_loss_term

                total_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.update_teacher()

            avg_train_loss = total_loss / len(train_loader)

            self.Student.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    views = [view.cuda(non_blocking=True) for view in batch]
                    # For validation, we use only global views (or a single view per image)
                    global_views = views[:2] if len(views) >= 2 else views
                    teacher_outs = [self.Teacher(view) for view in global_views]
                    student_outs = [self.Student(view) for view in global_views]
                    loss = 0
                    n_loss_term = 0
                    for i, t_out in enumerate(teacher_outs):
                        for j, s_out in enumerate(student_outs):
                            if j == i:
                                continue
                            loss += self.loss_fn(s_out, t_out.detach())
                            n_loss_term += 1
                    if n_loss_term > 0:
                        loss = loss / n_loss_term
                        total_val_loss += loss.item()
                avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

            # Log metrics to wandb and print
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
            print(f"Epoch {epoch+1}/{num_epoch} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


if __name__ == "__main__":
    from data_pipeline.data_aug import DINOAugmentation
    from data_pipeline.data_set import SSLTrainLoader , SSLValidLoader

    

    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    augmentor = DINOAugmentation(img_size=256)
    train_loader = SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=augmentor,
        batch_size=8,
        num_work=4,
    ).get_loader()


    valid_loader = SSLValidLoader(
        dataset_names=dataset_names,
        transformation=augmentor,
        batch_size=2,
        num_work=4,
    ).get_loader()
   
    wandb.init(project="DINO-Training", config={
        "img_size": 256,
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "teacher_momentum": 0.996,
        "in_dim": 768,
        "out_dim": 256
    })
    config = wandb.config
    
  

    encoder = models.convnext_tiny(pretrained=False)
    encoder.classifier = nn.Identity()
    encoder = encoder.cuda()
    dino_head = DINOHead(in_dim=config.in_dim, out_dim=config.out_dim)
    dino_head = dino_head.cuda()

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(dino_head.parameters()),
                                  lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = DINOLoss(out_dim=config.out_dim)

    dino_trainer = DINO(encoder=encoder, head=dino_head, optimizer=optimizer, loss_fn=loss_fn)

    # Start training.
    dino_trainer.train_loop(train_loader, valid_loader, num_epoch=config.num_epochs)
