import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm
import wandb
from PIL import Image
import torch.nn.functional as F
import math
import numpy as np
from data_pipeline import data_set

class DINOv2Loss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.07, student_temp=0.1, 
                 center_momentum=0.9, epsilon=1e-6):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.epsilon = epsilon

    # def forward(self, student_output, teacher_output):
    #     """
    #     Cross-entropy between softmax outputs of the teacher and student networks.
    #     """
    #     # Normalize the teacher output with the current center
    #     teacher_output = teacher_output - self.center
        
    #     # Student and teacher sharpening 
    #     student_out = student_output / self.student_temp
    #     teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)

    #     teacher_out = teacher_out.detach()
        
    #     loss = -torch.sum(teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
    #     loss = loss.mean()
        
    #     self.update_center(teacher_output)
        
    #     return loss

    # def forward(self, student_output, teacher_output):
    #     # Normalize teacher output with the current center
    #     teacher_output = teacher_output - self.center   

    #     student_out = student_output / self.student_temp
    #     teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)
    #     teacher_out = teacher_out.detach()
        
    #     if teacher_out.shape[0] != student_out.shape[0]:
    #         # Assumes student_out.shape[0] is an integer multiple of teacher_out.shape[0]
    #         repeat_factor = student_out.shape[0] // teacher_out.shape[0]
    #         teacher_out = teacher_out.repeat_interleave(repeat_factor, dim=0)
        
    #     # Compute cross-entropy loss between teacher and student distributions
    #     loss = -torch.sum(teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
    #     loss = loss.mean()
        
    #     self.update_center(teacher_output)
        
    #     return loss
    def forward(self, student_output, teacher_output):
        teacher_output = teacher_output - self.center
        
        student_out = student_output / self.student_temp
        teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()
        
        # More efficient handling of batch size mismatches
        if teacher_out.shape[0] != student_out.shape[0]:
            # Assumes specific pattern of global vs local crops
            if student_out.shape[0] % teacher_out.shape[0] == 0:
                repeat_factor = student_out.shape[0] // teacher_out.shape[0]
                teacher_out = teacher_out.repeat_interleave(repeat_factor, dim=0)
            else:
                # Handle more complex cases if needed
                pass
        
        # Use cross_entropy with more numerical stability
        loss = torch.nn.functional.cross_entropy(
            student_out,
            teacher_out,
            reduction='mean'
        )
        
        self.update_center(teacher_output)
        return loss


    @torch.no_grad()
    def update_center(self, teacher_output):
        
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are concatenated.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # Convert list of tensors to stacked tensor
        if isinstance(x, list):
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-2] for inp in x]),
                return_counts=True,
            )[1], 0)
            
            start_idx, output = 0, []
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                output.append(_out)
                start_idx = end_idx
                
            output = [self.head(out) for out in output]
            return output
        else:
            # Single input case (validation)
            x = self.backbone(x)
            return self.head(x)


class DINOv2Model(nn.Module):
    def __init__(self, base_model='convnext_tiny', embedding_dim=384, hidden_dim=1536,
                 bottleneck_dim=384, output_dim=384, drop_path=0.1, pretrained=False):
        super().__init__()
        
        # Student backbone
        self.student = timm.create_model(
            base_model, 
            pretrained=pretrained, 
            num_classes=0,
            drop_path_rate=drop_path
        )
        self.student_head = DINOProjectionHead(
            self.student.num_features,
            hidden_dim,
            bottleneck_dim,
            output_dim
        )
        self.student = MultiCropWrapper(self.student, self.student_head)
        
        # Teacher backbone with the same architecture (no drop path)
        self.teacher = timm.create_model(
            base_model, 
            pretrained=pretrained, 
            num_classes=0,
            drop_path_rate=0.0  # No stochastic depth for teacher
        )
        self.teacher_head = DINOProjectionHead(
            self.teacher.num_features,
            hidden_dim,
            bottleneck_dim,
            output_dim
        )
        self.teacher = MultiCropWrapper(self.teacher, self.teacher_head)
        
        # Initialize teacher with student parameters and freeze teacher
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def get_student_params(self):
        """Return all parameters of the student network"""
        return self.student.parameters()

    @torch.no_grad()
    def update_teacher(self, momentum):
        """Update teacher model with EMA of student parameters"""
        for param_student, param_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            param_teacher.data = momentum * param_teacher.data + (1 - momentum) * param_student.data


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, bottleneck_dim, out_dim, norm_last_layer=True):
        super().__init__()
        
        # First layer: in_dim -> hidden_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Second layer: hidden_dim -> bottleneck_dim
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        
        # Last layer: bottleneck_dim -> out_dim
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        
        # Normalization option for last layer weights
        self.norm_last_layer = norm_last_layer
        if norm_last_layer:
            self.last_layer.weight.data.copy_(
                F.normalize(self.last_layer.weight.data.clone(), dim=1, p=2)
            )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.normalize(x, dim=-1, p=2)
        
      
        if self.norm_last_layer and self.training:
            self.last_layer.weight.data.copy_(
                F.normalize(self.last_layer.weight.data.clone(), dim=1, p=2)
            )
            
        x = self.last_layer(x)
        return x


class DinoAugmentation:
    def __init__(
        self, 
        global_crops_scale=(0.5, 1.0), 
        local_crops_scale=(0.1, 0.4), 
        local_crops_number=8,
        img_size=224
    ):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Transformations for global crops
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale),
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale),
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])
        
        # Transformations for local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            flip_and_color_jitter,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])
    
    def __call__(self, img):
        crops = []
        # Global crops
        crops.append(self.global_transfo1(img))
        crops.append(self.global_transfo2(img))
        # Local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(img))
        return crops


class CosineSchedulerWithWarmup:
    def __init__(self, optimizer, base_lr, min_lr, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Smoother warmup (square root scaling)
            progress = self.current_epoch / self.warmup_epochs
            factor = math.sqrt(progress)
            lr = self.min_lr + (self.base_lr - self.min_lr) * factor
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_factor
            
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def train_one_epoch(student_teacher, dino_loss, data_loader, optimizer, lr_scheduler, 
                    teacher_momentum_schedule, epoch, device, wandb_run):
    student_teacher.train()
    running_loss = 0.0
    
    for i, images in enumerate(data_loader):
        # Move image crops to device
        images = [im.to(device) for im in images]
        
        # Get teacher momentum for current step
        momentum_val = teacher_momentum_schedule[epoch * len(data_loader) + i]
        
        # Forward pass through student and teacher networks
        with torch.cuda.amp.autocast():
            # Student output on all crops
            student_output = student_teacher.student(images)
            
            # Teacher output on global crops only
            with torch.no_grad():
                teacher_output = student_teacher.teacher(images[:2])
            
            # Loss calculation between all student crops and teacher global crops
            loss = 0
            for iq, q in enumerate(teacher_output):
                for v in range(len(student_output)):
                    if v == iq:  # Skip same crop comparison
                        continue
                    loss += dino_loss(student_output[v], q)
            
            loss /= (2 * (len(images) - 2))  # Average over all comparisons
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update teacher
        student_teacher.update_teacher(momentum_val)
        
        running_loss += loss.item()
        
        if i % 10 == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(data_loader)}] "
                         f"Loss: {loss.item():.4f}, LR: {learning_rate:.6f}")
            wandb_run.log({
                "train_loss": loss.item(), 
                "learning_rate": learning_rate,
                "epoch": epoch + i/len(data_loader)
            })
    
    avg_loss = running_loss / len(data_loader)
    return avg_loss


def validate(student_teacher, dino_loss, data_loader, epoch, device, wandb_run):
    student_teacher.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, (images,) in enumerate(data_loader):
            # Move image crops to device
            images = [im.to(device) for im in images]
            
            # Only process global crops for validation
            with torch.cuda.amp.autocast():
                student_output = student_teacher.student(images[:2])
                teacher_output = student_teacher.teacher(images[:2])
                
                # Calculate loss on global crops only
                loss = 0
                for iq, q in enumerate(teacher_output):
                    for v, vs in enumerate(student_output):
                        if v == iq:  # Skip same crop comparison
                            continue
                        loss += dino_loss(vs, q)
                loss /= 2  # Average over comparisons
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(data_loader)
    logging.info(f"Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}")
    wandb_run.log({"val_loss": avg_loss, "epoch": epoch+1})
    return avg_loss


def save_checkpoint(state, checkpoint_dir, filename):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint: {filepath}")


def main():
    # Configuration
    config = {
        "base_model": "convnext_tiny",
        "embedding_dim": 384,
        "hidden_dim": 1536, 
        "bottleneck_dim": 384,
        "output_dim": 384,
        "teacher_temp": 0.05,
        "student_temp": 0.08,
        "warmup_teacher_temp": 0.04,
        "warmup_teacher_temp_epochs": 30,
        "learning_rate": 1e-3,
        "min_lr": 1e-6,
        "weight_decay": 0.05,
        "warmup_epochs": 10,
        "total_epochs": 250,
        "drop_path": 0.15,
        "local_crops_number": 8,
        "global_crops_scale": (0.4, 1.0),
        "local_crops_scale": (0.05, 0.3),
        "img_size": 384,
        "batch_size": 32,
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    os.makedirs("model/new/chckpt/dinov2", exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
   
    
    # Build DINOv2 model
    model = DINOv2Model(
        base_model=config["base_model"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        bottleneck_dim=config["bottleneck_dim"],
        output_dim=config["output_dim"],
        drop_path=config["drop_path"]
    ).to(device)
    
    # Loss function
    dino_loss = DINOv2Loss(
        out_dim=config["output_dim"],
        teacher_temp=config["teacher_temp"],
        student_temp=config["student_temp"]
    ).to(device)
    
    # Optimizer
    param_groups = [
        {'params': [p for p in model.get_student_params()]},
    ]
    optimizer = optim.AdamW(
        param_groups,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Learning rate scheduler
    lr_scheduler = CosineSchedulerWithWarmup(
        optimizer=optimizer,
        base_lr=config["learning_rate"],
        min_lr=config["min_lr"],
        warmup_epochs=config["warmup_epochs"],
        total_epochs=config["total_epochs"]
    )
    
    # Teacher temperature scheduler - linear warmup then constant
    warmup_teacher_temp_schedule = np.concatenate((
        np.linspace(config["warmup_teacher_temp"], 
                   config["teacher_temp"], 
                   config["warmup_teacher_temp_epochs"]),
        np.ones(config["total_epochs"] - config["warmup_teacher_temp_epochs"]) * config["teacher_temp"]
    ))
    
    # Teacher momentum schedule (cosine from 0.996 to 1.0)
    teacher_momentum_base = 0.996
    iterations_per_epoch = 1000  
    teacher_momentum_schedule = np.linspace(
        teacher_momentum_base, 1.0, config["total_epochs"] * iterations_per_epoch
    )

    
    # Create augmentation
    transforms_ = DinoAugmentation(
        global_crops_scale=config["global_crops_scale"],
        local_crops_scale=config["local_crops_scale"],
        local_crops_number=config["local_crops_number"],
        img_size=config["img_size"]
    )
    
    # Datasets
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    train_loader = data_set.SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=config["batch_size"],
        num_work=4
    ).get_loader()
    
    valid_loader = data_set.SSLValidLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=8,
        num_work=4
    ).get_loader()
    
    # Adjust momentum schedule based on actual number of iterations
    iterations_per_epoch = len(train_loader)
    teacher_momentum_schedule = np.concatenate((
        np.linspace(teacher_momentum_base, 1.0, config["total_epochs"] * iterations_per_epoch),
    ))

     # Initialize Weights & Biases
    wandb_run = wandb.init(project="DINOv2-DR", config=config)
    best_val_loss = float('inf')
    start_epoch = 0
    
    try:
        for epoch in range(start_epoch, config["total_epochs"]):
            logging.info(f"--- Epoch {epoch+1}/{config['total_epochs']} ---")
            
            current_lr = lr_scheduler.step()
            
            
            train_loss = train_one_epoch(
                model, dino_loss, train_loader, optimizer, 
                lr_scheduler, teacher_momentum_schedule, epoch, device, wandb_run
            )
            
            # Validate
            val_loss = validate(model, dino_loss, valid_loader, epoch, device, wandb_run)
            
            # Save checkpoint after each epoch
            checkpoint_state = {
                'epoch': epoch + 1,
                'student_state_dict': model.student.state_dict(),
                'teacher_state_dict': model.teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            epoch_ckpt = f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(checkpoint_state, "model/new/chckpt/dinov2", epoch_ckpt)
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_state, "model/new/chckpt/dinov2", "best_checkpoint.pth")
                
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected! Saving checkpoint before exiting...")
        checkpoint_state = {
            'epoch': epoch + 1,
            'student_state_dict': model.student.state_dict(),
            'teacher_state_dict': model.teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        save_checkpoint(checkpoint_state, "model/new/chckpt/dinov2", f"interrupt_checkpoint_epoch_{epoch+1}.pth")
    finally:
        wandb_run.finish()

if __name__ == "__main__":
    main()