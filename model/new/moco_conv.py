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

from data_pipeline import data_aug , data_set


class MoCoV3Model(nn.Module):
    def __init__(self, base_model='convnext_tiny', projection_dim=128, hidden_dim=512,
                 queue_size=1024, momentum=0.99, pretrained=False):
        super(MoCoV3Model, self).__init__()
        self.m = momentum  # momentum coefficient for key encoder update

        # Build the query encoder with ConvNeXt backbone and projection head
        self.query_encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        feature_dim = self.query_encoder.num_features
        self.query_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

        # Build the key encoder (same architecture as query encoder)
        self.key_encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        self.key_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        #key weights
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the negative key queue (size: queue_size x projection_dim)
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(queue_size, projection_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, im_q, im_k):
        # Compute query features
        q = self.query_encoder(im_q)
        q = self.query_projector(q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            k = self.key_encoder(im_k)
            k = self.key_projector(k)
            k = nn.functional.normalize(k, dim=1)
        return q, k

    @torch.no_grad()
    def update_key_encoder(self):
        """Momentum update of key encoder's parameters."""
        
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """Update the negative key queue with the new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size > self.queue_size:
            # find overflow amt
            overflow = (ptr + batch_size) - self.queue_size
            self.queue[ptr:self.queue_size] = keys[:(batch_size - overflow)]
            self.queue[0:overflow] = keys[(batch_size - overflow):]
            self.queue_ptr[0] = overflow
        else:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size


def moco_loss(q, k, queue, temperature=0.2):
    # cosine similarity
    # dot product
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.mm(q, queue.T)
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= temperature

    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


def train_one_epoch(model, dataloader, optimizer, temperature, device, epoch, wandb_run):
    model.train()
    running_loss = 0.0
    for i, (im_q, im_k) in enumerate(dataloader):
        im_q = im_q.to(device)
        im_k = im_k.to(device)
        optimizer.zero_grad()

        # Forward pass: get query and key features
        q, k = model(im_q, im_k)
        loss = moco_loss(q, k, model.queue, temperature=temperature)
        loss.backward()
        optimizer.step()

        # Update key encoder and update the queue with new keys
        model.update_key_encoder()
        with torch.no_grad():
            model.dequeue_and_enqueue(k)

        running_loss += loss.item()

        if i % 10 == 0:
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
            wandb_run.log({"train_loss": loss.item(), "epoch": epoch+1})
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, temperature, device, epoch, wandb_run):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (im_q, im_k) in enumerate(dataloader):
            im_q = im_q.to(device)
            im_k = im_k.to(device)
            q, k = model(im_q, im_k)
            loss = moco_loss(q, k, model.queue, temperature=temperature)
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
        "temperature" : 0.2,
        "momentum" : 0.99,
        "queue_size" : 1024,
        "base_model" : "convnext_tiny",
        "projection_dim" : 128,
        "hidden_dim" : 512,
        "pretrained" : False
    }


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    os.makedirs("model/new/chckpt/moco", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    

    # Initialize Weights & Biases
    wandb_run = wandb.init(project="MoCoV3-DR", config=config)

    # Build the MoCo v3 model with ConvNeXt backbone
    model = MoCoV3Model()
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    transforms_ = data_aug.MoCoAug(img_size=256)
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

    best_val_loss = float('inf')
    start_epoch = 0

    try:
        for epoch in range(start_epoch, config["epochs"]):
            logging.info(f"--- Epoch {epoch+1}/{config["epochs"]} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, config["temperature"], device, epoch, wandb_run)
            val_loss = validate(model, valid_loader, config["temperature"], device, epoch, wandb_run)

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
            save_checkpoint(checkpoint_state, "model/new/chckpt/moco", epoch_ckpt)

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_state, "model/new/chckpt/moco", "best_checkpoint.pth")

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
        save_checkpoint(checkpoint_state, "model/new/chckpt/moco", f"interrupt_checkpoint_epoch_{epoch+1}.pth")
    finally:
        wandb_run.finish()

if __name__ == "__main__":
    main()
