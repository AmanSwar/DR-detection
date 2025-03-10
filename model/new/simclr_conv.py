import os
import argparse
import logging
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.transforms as transforms
from PIL import Image

import timm
import wandb

from data_pipeline import data_aug, data_set


class SimCLRModel(nn.Module):
    def __init__(self, base_model='convnext_tiny', projection_dim=128, hidden_dim=512, pretrained=False):
        super(SimCLRModel, self).__init__()
        self.encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        # The backbone's feature dimension (e.g., 768 for convnext_tiny)
        feature_dim = self.encoder.num_features

        # Define a projection head (2-layer MLP with ReLU)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)  
        z = self.projector(h) 
        return h, z


class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().to(device)

    def _get_correlated_mask(self):
        # Create a mask to remove similarity of samples with themselves and their positive pair
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(False)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = False
            mask[self.batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss given two batches of projections.
        Args:
            z_i: Tensor of shape [batch_size, dim]
            z_j: Tensor of shape [batch_size, dim]
        """
       
        z = torch.cat([z_i, z_j], dim=0)  
     
        z = nn.functional.normalize(z, dim=1)
     
        similarity_matrix = torch.matmul(z, z.T)  # shape: (2N, 2N)

        # Positive pairs: diagonal offset by batch_size
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0).unsqueeze(1)  # shape: (2N, 1)

        # Negatives: filter out positives and self-similarities
        negatives = similarity_matrix[self.mask].view(2 * self.batch_size, -1)

        # Concatenate positive and negatives
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        # Labels: positive at index 0 for each example
        labels = torch.zeros(2 * self.batch_size, dtype=torch.long).to(self.device)

        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss


class LinearProbeHead(nn.Module):
    """A simple linear classifier for evaluating SSL embeddings."""
    def __init__(self, embed_dim, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    Extract embeddings from the frozen encoder for each (image, label).
    Return: features (N, embed_dim), labels (N,)
    """
    model.eval()
    all_feats = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device)
        # Use only the encoder (no projection) or encoder+projection
        feats, _ = model(images)  # shape: [B, feature_dim]
        # Optionally apply the projector if you want to evaluate that space
        # _, feats = model(images)  # uncomment if you want to use projection space
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_feats, all_labels

def linear_probe_evaluation(model, train_loader, val_loader, device, wandb_run):
    """
    Freeze the encoder, extract features, and train a linear classifier.
    Then evaluate on a validation set.
    """
    # 1. Extract features
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)

    # 2. Create linear probe
    embed_dim = train_feats.shape[1]
    num_classes = len(train_labels.unique())  # e.g., 5 DR classes
    probe = LinearProbeHead(embed_dim, num_classes).to(device)

    # 3. Train the linear probe
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Simple training loop (few epochs)
    probe_epochs = 5
    for ep in range(probe_epochs):
        probe.train()
        perm = torch.randperm(train_feats.size(0))
        train_feats_shuf = train_feats[perm].to(device)
        train_labels_shuf = train_labels[perm].to(device)

        # mini-batch training
        batch_size = 64
        for i in range(0, train_feats_shuf.size(0), batch_size):
            end = i + batch_size
            batch_feats = train_feats_shuf[i:end]
            batch_labels = train_labels_shuf[i:end]

            optimizer.zero_grad()
            outputs = probe(batch_feats)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # 4. Evaluate
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
    """
    A simple k-NN classifier on top of extracted embeddings.
    """
    from collections import Counter
    import numpy as np

    # 1. Extract features
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)

    # Convert to numpy for quick distance computations
    train_feats_np = train_feats.numpy()
    train_labels_np = train_labels.numpy()
    val_feats_np = val_feats.numpy()
    val_labels_np = val_labels.numpy()

    # 2. For each val sample, find the k nearest neighbors
    correct = 0
    for i in range(len(val_feats_np)):
        diff = train_feats_np - val_feats_np[i]  # shape: [N, D]
        dist = np.sum(diff**2, axis=1)           # shape: [N]
        idx = np.argsort(dist)[:k]               # k nearest
        neighbors = train_labels_np[idx]
        # majority vote
        majority = Counter(neighbors).most_common(1)[0][0]
        if majority == val_labels_np[i]:
            correct += 1

    acc = 100.0 * correct / len(val_feats_np)
    logging.info(f"[k-NN (k={k})] Validation Accuracy: {acc:.2f}%")
    if wandb_run is not None:
        wandb_run.log({"knn_accuracy": acc})
    return acc

def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, epoch, wandb_run):
    model.train()
    running_loss = 0.0
    for i, (x1, x2) in enumerate(dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)

        optimizer.zero_grad()
        # Forward pass for both augmented views
        _, z1 = model(x1)
        _, z2 = model(x2)

        loss = loss_fn(z1, z2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")
            wandb_run.log({
                "train_loss": loss.item(), 
                "epoch": epoch+1,
                "learning_rate": current_lr
            })
    
    # Step the scheduler after each epoch
    scheduler.step()
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, loss_fn, device, epoch, wandb_run):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (x1, x2) in enumerate(dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = loss_fn(z1, z2)
            running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    logging.info(f"Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}")
    wandb_run.log({"val_loss": avg_loss, "epoch": epoch+1})
    return avg_loss

def save_checkpoint(state, checkpoint_dir, filename):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint: {filepath}")

# -----------------------------
# 6. Main Training Loop
# -----------------------------
def main():
    config = {
        "epochs": 300,
        "batch_size": 128,
        "lr": 5e-4  ,
        "lr_min": 1e-5,  
        "warm_up_epochs": 10,
        "temperature": 0.5,
        "base_model": "convnext_small",
        "projection_dim": 256,
        "hidden_dim": 512,
        "pretrained": False,
        "checkpoint": "model/new/chckpt/simclr"
    }

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    os.makedirs(config["checkpoint"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb_run = wandb.init(project="SimCLR-DR", config=config)

    # Build the SimCLR model with ConvNeXt backbone
    model = SimCLRModel(
        base_model=config["base_model"],
        projection_dim=config["projection_dim"],
        hidden_dim=config["hidden_dim"],
        pretrained=config["pretrained"]
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # Learning rate scheduler - Cosine Annealing
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config["epochs"] - config["warm_up_epochs"],
        eta_min=config["lr_min"]
    )

    loss_fn = NTXentLoss(batch_size=config["batch_size"], temperature=config["temperature"], device=device)

    checkpoint_path = "NaN"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        logging.info("No checkpoint found, starting from scratch")
    
    # Configure warm-up
    if config["warm_up_epochs"] > 0:
        initial_lr = config["lr"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = config["lr_min"]  # Start with minimum lr

    # Data loaders for SSL
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    transforms_ = data_aug.SimCLRAug(img_size=256)
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
    
    # Optional: Labeled data loaders for evaluation
    train_aug = data_aug.SimCLRAug(img_size=256)  # Should implement this class similar to MoCoSingleAug
    probe_train_loader = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_aug,
        batch_size=64,
        num_workers=0,
        sampler=True
    ).get_loader()

    val_aug = data_aug.SimCLRAug    (img_size=256)  # Should implement this class similar to MoCoSingleAug
    probe_val_loader = data_set.UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=val_aug,
        batch_size=8,
        num_workers=0,
        sampler=True
    ).get_loader()

    best_val_loss = float('inf')
    train_loss = 0
    val_loss = 0
    
    try:
        for epoch in range(start_epoch, config["epochs"]):
            logging.info(f"--- Epoch {epoch+1}/{config['epochs']} ---")
            
            # Implement warm-up if configured
            if epoch < config["warm_up_epochs"]:
                # Linear warm-up
                progress = (epoch + 1) / config["warm_up_epochs"]
                lr = config["lr_min"] + progress * (initial_lr - config["lr_min"])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                logging.info(f"Warm-up phase: LR set to {lr:.6f}")
                wandb_run.log({"learning_rate": lr, "epoch": epoch+1})
                
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch, wandb_run)
            val_loss = validate(model, valid_loader, loss_fn, device, epoch, wandb_run)

            # Save checkpoint after each epoch
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            epoch_ckpt = f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(checkpoint_state, config["checkpoint"], epoch_ckpt)

            # Save best model (based on validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_state, config["checkpoint"], "best_checkpoint.pth")
                
            # Evaluate representations with a linear probe or k-NN
            if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs to save time
                linear_probe_evaluation(model, probe_train_loader, probe_val_loader, device, wandb_run)
                knn_evaluation(model, probe_train_loader, probe_val_loader, device, k=5, wandb_run=wandb_run)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected! Saving checkpoint before exiting...")
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        save_checkpoint(checkpoint_state, config["checkpoint"], f"interrupt_checkpoint_epoch_{epoch+1}.pth")
    finally:
        wandb_run.finish()

if __name__ == "__main__":
    main()