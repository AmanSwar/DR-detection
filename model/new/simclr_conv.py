import os
import argparse
import logging
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from PIL import Image

import timm
import wandb

from data_pipeline import data_aug ,data_set


class SimCLRModel(nn.Module):
    def __init__(self, base_model='convnext_tiny', projection_dim=128, hidden_dim=512, pretrained=False):
        super(SimCLRModel, self).__init__()
        self.encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        # The backbone’s feature dimension (e.g., 768 for convnext_tiny)
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

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch, wandb_run):
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
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
            wandb_run.log({"train_loss": loss.item(), "epoch": epoch+1})
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
        "epochs" : 300,
        "batch_size" : 64,
        "lr" : 1e-3,
        "temperature" : 0.5,
        "base_model" : "convnext_tiny",
        "projection_dim" : 128,
        "hidden_dim" : 512,
        "pretrained" : False,
        "checkpoint" : "model/new/chckpt/moco"
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
    model = SimCLRModel()
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Loss function
    loss_fn = NTXentLoss(batch_size=config["batch_size"], temperature=config["temperature"], device=device)

    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    transforms_ = data_aug.SimCLRAug(img_size=256)
    train_loader = data_set.SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=config["batch_size"],
        num_work=0,
    ).get_loader()

    valid_loader = data_set.SSLValidLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=8,
        num_work=0,
    ).get_loader()
    

    best_val_loss = float('inf')
    start_epoch = 0

    try:
        for epoch in range(start_epoch, config["epochs"]):
            logging.info(f"--- Epoch {epoch+1}/{config["epochs"]} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, wandb_run)
            val_loss = validate(model, valid_loader, loss_fn, device, epoch, wandb_run)

            # Save checkpoint after each epoch
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected! Saving checkpoint before exiting...")
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        save_checkpoint(checkpoint_state, config["checkpoint"], f"interrupt_checkpoint_epoch_{epoch+1}.pth")
    finally:
        wandb_run.finish()

if __name__ == "__main__":
    main()
