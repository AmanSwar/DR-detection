import os
import logging
import datetime
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import wandb
import numpy as np

from data_pipeline import data_aug, data_set


class DinoV2Model(nn.Module):
    def __init__(self, base_model='convnext_small', projection_dim=256, hidden_dim=1024, 
                 bottleneck_dim=128, teacher_temp=0.04, student_temp=0.1, 
                 center_momentum=0.9, pretrained=True):
        super(DinoV2Model, self).__init__()
        # Create student and teacher networks
        self.student = self._create_encoder(base_model, projection_dim, hidden_dim, bottleneck_dim, pretrained)
        self.teacher = self._create_encoder(base_model, projection_dim, hidden_dim, bottleneck_dim, pretrained)
        
        # Initialize teacher weights as a copy of student weights
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False  # Teacher is updated via EMA
        
        # Temperature parameters
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        
        # Center momentum for the teacher output
        self.register_buffer("center", torch.zeros(1, projection_dim))
        self.center_momentum = center_momentum
        
    def _create_encoder(self, base_model, projection_dim, hidden_dim, bottleneck_dim, pretrained):
        """Create backbone + projection head."""
        backbone = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        feature_dim = backbone.num_features
        
        # DINOv2 uses a 3-layer projection head with a bottleneck structure
        projection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim, eps=1e-6),
            nn.Linear(bottleneck_dim, projection_dim)
        )
        
        return nn.Sequential(backbone, projection_head)
    
    def forward(self, x):
        """Process one batch of views."""
        student_output = self.student(x)
        student_output = F.normalize(student_output, dim=-1)
        
        with torch.no_grad():
            teacher_output = self.teacher(x)
            teacher_output = F.normalize(teacher_output, dim=-1)
        
        return student_output, teacher_output
    
    @torch.no_grad()
    def update_teacher(self, m):
        """EMA update of the teacher network."""
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = param_t.data * m + param_s.data * (1. - m)
            
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def dino_loss(student_output, teacher_output, teacher_temp, student_temp, center):
    """
    Compute DINO loss
    """
    student_out = student_output / student_temp
    teacher_out = teacher_output / teacher_temp
    teacher_out = teacher_out.detach()  # stop gradient
    
    # Center the teacher output
    teacher_centered = teacher_out - center
    
    # Cross-entropy between student and centered teacher predictions
    student_out = student_out.chunk(2)
    teacher_centered = teacher_centered.chunk(2)
    
    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_centered):
        for iv, v in enumerate(student_out):
            if iq == iv:  # Skip same view
                continue
            loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
            total_loss += loss
            n_loss_terms += 1
    
    total_loss /= n_loss_terms
    return total_loss


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, wandb_run, 
                     m_teacher_momentum, mask_ratio=0.75):
    model.train()
    running_loss = 0.0
    
    for i, (views) in enumerate(dataloader):
        
        views = [v.to(device) for v in views]  
        
        # Process both views through the model
        all_student_outputs = []
        all_teacher_outputs = []
        
        # Process the multi-crop views batch 
        for view in views:
            student_out, teacher_out = model(view)
            all_student_outputs.append(student_out)
            all_teacher_outputs.append(teacher_out)
        
        # Concatenate all outputs
        student_output = torch.cat(all_student_outputs, dim=0)
        teacher_output = torch.cat(all_teacher_outputs, dim=0)
        
        # Update center for teacher output
        model.update_center(teacher_output)
        
        # Compute loss
        loss = dino_loss(
            student_output, 
            teacher_output, 
            model.teacher_temp, 
            model.student_temp, 
            model.center
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update teacher network
        model.update_teacher(m_teacher_momentum)
        
        running_loss += loss.item()
        if i % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")
            wandb_run.log({
                "train_loss": loss.item(), 
                "epoch": epoch+1,
                "learning_rate": current_lr
            })
    
    scheduler.step()
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device, epoch, wandb_run):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, (views, _) in enumerate(dataloader):
            views = [v.to(device) for v in views]
            
            all_student_outputs = []
            all_teacher_outputs = []
            
            for view in views:
                student_out, teacher_out = model(view)
                all_student_outputs.append(student_out)
                all_teacher_outputs.append(teacher_out)
            
            student_output = torch.cat(all_student_outputs, dim=0)
            teacher_output = torch.cat(all_teacher_outputs, dim=0)
            
            loss = dino_loss(
                student_output, 
                teacher_output, 
                model.teacher_temp, 
                model.student_temp, 
                model.center
            )
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    logging.info(f"Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}")
    wandb_run.log({"val_loss": avg_loss, "epoch": epoch+1})
    return avg_loss


def save_checkpoint(state, checkpoint_dir, filename):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint: {filepath}")


class LinearProbeHead(nn.Module):
    """A simple linear classifier for evaluating SSL embeddings."""
    def __init__(self, embed_dim, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract embeddings from the backbone (no projection head)"""
    model.eval()
    all_feats = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        # Extract features from the backbone only (first part of student network)
        feats = model.student[0](images)  # Access the backbone
        all_feats.append(feats.cpu())
        all_labels.append(labels)
        
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_feats, all_labels


def linear_probe_evaluation(model, train_loader, val_loader, device, wandb_run):
    """Evaluate representations by training a linear classifier on frozen features"""
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)

    embed_dim = train_feats.shape[1]
    num_classes = len(train_labels.unique())
    probe = LinearProbeHead(embed_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the linear probe
    probe_epochs = 5
    for ep in range(probe_epochs):
        probe.train()
        perm = torch.randperm(train_feats.size(0))
        train_feats_shuf = train_feats[perm].to(device)
        train_labels_shuf = train_labels[perm].to(device)

        batch_size = 64
        for i in range(0, train_feats_shuf.size(0), batch_size):
            end = min(i + batch_size, train_feats_shuf.size(0))
            batch_feats = train_feats_shuf[i:end]
            batch_labels = train_labels_shuf[i:end]

            optimizer.zero_grad()
            outputs = probe(batch_feats)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    val_feats_gpu = val_feats.to(device)
    with torch.no_grad():
        logits = probe(val_feats_gpu)
        pred = torch.argmax(logits, dim=1).cpu()
        acc = (pred == val_labels).float().mean().item() * 100.0

    logging.info(f"[Linear Probe] Validation Accuracy: {acc:.2f}%")
    wandb_run.log({"linear_probe_accuracy": acc})
    return acc


def knn_evaluation(model, train_loader, val_loader, device, k=5, wandb_run=None):
    """K-nearest neighbors classifier on the embeddings"""
    from collections import Counter
    import numpy as np

    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)

    train_feats_np = train_feats.numpy()
    train_labels_np = train_labels.numpy()
    val_feats_np = val_feats.numpy()
    val_labels_np = val_labels.numpy()

    correct = 0
    for i in range(len(val_feats_np)):
        diff = train_feats_np - val_feats_np[i]
        dist = np.sum(diff**2, axis=1)
        idx = np.argsort(dist)[:k]
        neighbors = train_labels_np[idx]
        majority = Counter(neighbors).most_common(1)[0][0]
        if majority == val_labels_np[i]:
            correct += 1

    acc = 100.0 * correct / len(val_feats_np)
    logging.info(f"[k-NN (k={k})] Validation Accuracy: {acc:.2f}%")
    if wandb_run is not None:
        wandb_run.log({"knn_accuracy": acc})
    return acc


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    """Cosine scheduler with warmup for updating teacher momentum and learning rate"""
    warmup_schedule = np.linspace(0, base_value, warmup_epochs * niter_per_ep)
    iters = np.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def main():
    import numpy as np
    
    config = {
        "epochs": 300,
        "batch_size": 128,
        "lr": 5e-4,
        "lr_min": 1e-5,
        "warm_up_epochs": 10,
        "teacher_temp": 0.04,  # Temperature for teacher
        "student_temp": 0.1,   # Temperature for student
        "center_momentum": 0.9,  # EMA factor for center update
        "teacher_momentum_base": 0.996,  # Base EMA factor for teacher update
        "teacher_momentum_final": 1.0,   # Final EMA factor for teacher update
        "base_model": "convnext_small",
        "projection_dim": 256,
        "hidden_dim": 1024,
        "bottleneck_dim": 128,
        "pretrained": True
    }

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    os.makedirs("model/new/chckpt/dinov2", exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb_run = wandb.init(project="DINOv2-DR", config=config)

    model = DinoV2Model(
        base_model=config["base_model"],
        projection_dim=config["projection_dim"],
        hidden_dim=config["hidden_dim"],
        bottleneck_dim=config["bottleneck_dim"],
        teacher_temp=config["teacher_temp"],
        student_temp=config["student_temp"],
        center_momentum=config["center_momentum"],
        pretrained=config["pretrained"]
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.student.parameters(), lr=config["lr"])
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config["epochs"] - config["warm_up_epochs"],
        eta_min=config["lr_min"]
    )
    
    # Check for existing checkpoints to resume training
    checkpoint_path = "model/new/chckpt/dinov2/checkpoint_latest.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        logging.info("No checkpoint found, starting from scratch")
    
    # Warm-up learning rate
    if config["warm_up_epochs"] > 0 and start_epoch < config["warm_up_epochs"]:
        initial_lr = config["lr"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = config["lr_min"]  # Start with minimum lr
    
   
    transforms_ = data_aug.DiNOV2Aug(img_size=256) 
    
    # Unlabeled data loaders for SSL
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    train_loader = data_set.SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=config["batch_size"],
        num_work=4,
    ).get_loader()

    valid_loader = data_set.SSLValidLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=8,
        num_work=4,
    ).get_loader()

    # Labeled data loaders for evaluation
    train_aug = data_aug.DiNOSingleAug(img_size=256)  # Create appropriate single-view augmentation for evaluation
    probe_train_loader = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_aug,
        batch_size=64,
        num_workers=0,
        sampler=True
    ).get_loader()

    val_aug = data_aug.DiNOSingleAug(img_size=256)
    probe_val_loader = data_set.UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=val_aug,
        batch_size=8,
        num_workers=0,
        sampler=True
    ).get_loader()

    # Compute the number of iterations per epoch
    iters_per_epoch = len(train_loader)
    
    # Create schedule for teacher momentum
    teacher_momentum_schedule = cosine_scheduler(
        config["teacher_momentum_base"],
        config["teacher_momentum_final"],
        config["epochs"],
        iters_per_epoch,
        config["warm_up_epochs"]
    )

    best_val_loss = float('inf')
    best_linear_probe_acc = 0.0
    train_loss = 0
    val_loss = 0
    
    try:
        for epoch in range(start_epoch, config["epochs"]):
            logging.info(f"--- Epoch {epoch+1}/{config['epochs']} ---")
            
            # Apply learning rate warm-up
            if epoch < config["warm_up_epochs"]:
                progress = (epoch + 1) / config["warm_up_epochs"]
                lr = config["lr_min"] + progress * (config["lr"] - config["lr_min"])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                logging.info(f"Warm-up phase: LR set to {lr:.6f}")
                wandb_run.log({"learning_rate": lr, "epoch": epoch+1})
            
            # Train for one epoch
            # We need to pass the correct teacher momentum for this epoch's iterations
            momentum_idx_start = epoch * iters_per_epoch
            momentum_idx_end = (epoch + 1) * iters_per_epoch
            
            # Pass the momentum schedule to the training function
            train_loss = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                scheduler, 
                device, 
                epoch, 
                wandb_run,
                m_teacher_momentum=teacher_momentum_schedule[momentum_idx_start]  # Use the first value for logging simplicity
            )
            
            # Validate
            val_loss = validate(model, valid_loader, device, epoch, wandb_run)

            # Save checkpoint after each epoch
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            
            # Save latest checkpoint
            save_checkpoint(checkpoint_state, "model/new/chckpt/dinov2", "checkpoint_latest.pth")
            
            # Save epoch checkpoint
            if (epoch + 1) % 10 == 0:  # Save every 10 epochs to avoid filling up storage
                epoch_ckpt = f"checkpoint_epoch_{epoch+1}.pth"
                save_checkpoint(checkpoint_state, "model/new/chckpt/dinov2", epoch_ckpt)

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_state, "model/new/chckpt/dinov2", "best_loss_checkpoint.pth")
                logging.info(f"New best validation loss: {best_val_loss:.4f}")

            # Evaluate representations with linear probe and k-NN
            if (epoch + 1) % 5 == 0 or epoch == config["epochs"] - 1:  # Evaluate periodically
                linear_probe_acc = linear_probe_evaluation(model, probe_train_loader, probe_val_loader, device, wandb_run)
                knn_acc = knn_evaluation(model, probe_train_loader, probe_val_loader, device, k=6, wandb_run=wandb_run)
                
                # Save best model based on linear probe accuracy
                if linear_probe_acc > best_linear_probe_acc:
                    best_linear_probe_acc = linear_probe_acc
                    save_checkpoint(checkpoint_state, "model/new/chckpt/dinov2", "best_acc_checkpoint.pth")
                    logging.info(f"New best linear probe accuracy: {best_linear_probe_acc:.2f}%")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected! Saving checkpoint before exiting...")
        checkpoint_state = {
            'epoch': epoch + 1, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        save_checkpoint(checkpoint_state, "model/new/chckpt/dinov2", f"interrupt_checkpoint_epoch_{epoch+1}.pth")
    finally:
        wandb_run.finish()

if __name__ == "__main__":
    main()