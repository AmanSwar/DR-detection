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

from data_pipeline import data_aug, data_set

# [MoCoV3Model, moco_loss, train_one_epoch, validate, save_checkpoint remain unchanged]

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

        # Initialize key encoder with query encoder weights and freeze its gradients
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

        # Compute key features with no gradient and update key encoder later via momentum
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
        # If the new keys exceed the remaining space in the queue, wrap around
        if ptr + batch_size > self.queue_size:
            overflow = (ptr + batch_size) - self.queue_size
            self.queue[ptr:self.queue_size] = keys[:(batch_size - overflow)]
            self.queue[0:overflow] = keys[(batch_size - overflow):]
            self.queue_ptr[0] = overflow
        else:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

# -----------------------------
# 4. MoCo v3 Loss Function (InfoNCE Loss)
# -----------------------------
def moco_loss(q, k, queue, temperature=0.2):
    """
    Args:
        q: Query features of shape [N, dim]
        k: Key features (positive) of shape [N, dim]
        queue: Negative keys of shape [K, dim]
        temperature: Temperature parameter for scaling logits
    Returns:
        InfoNCE loss
    """
    # Positive logits: (N, 1)
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    # Negative logits: (N, K)
    l_neg = torch.mm(q, queue.T)
    # Concatenate logits and apply temperature scaling
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= temperature

    # Labels: positives are the 0-th index in logits
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

# -----------------------------
# 5. Training and Validation Functions
# -----------------------------
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


    
def train(config=None):
    """
    Training function for MoCoV3 with optional config parameter for final training.
    If config is None, uses wandb.config from sweep.
    """
    # Initialize wandb
    if config is None:
        wandb.init(project="MoCoV3-DR")
        config = wandb.config
    else:
        wandb.init(project="MoCoV3-DR", config=config)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Unique checkpoint directory per run
    checkpoint_dir = os.path.join("model/new/chckpt/moco", wandb.run.id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build model
    model = MoCoV3Model(
        base_model=config["base_model"],
        projection_dim=config["projection_dim"],
        hidden_dim=config["hidden_dim"],
        queue_size=config["queue_size"],
        momentum=config["momentum"],
        pretrained=config["pretrained"]
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Data loaders
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
        batch_size=config["batch_size"],  # Match training batch_size for consistency
        num_work=4,
    ).get_loader()

    # Training loop
    best_val_loss = float('inf')
    last_val_loss = None

    try:
        for epoch in range(config["epochs"]):
            logging.info(f"--- Epoch {epoch+1}/{config['epochs']} ---")
            train_loss = train_one_epoch(
                model, train_loader, optimizer, config["temperature"], device, epoch, wandb
            )
            val_loss = validate(
                model, valid_loader, config["temperature"], device, epoch, wandb
            )
            last_val_loss = val_loss

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
            save_checkpoint(checkpoint_state, checkpoint_dir, epoch_ckpt)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_state, checkpoint_dir, "best_checkpoint.pth")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected! Saving checkpoint before exiting...")
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss if 'train_loss' in locals() else None,
            'val_loss': last_val_loss,
            'config': config
        }
        save_checkpoint(checkpoint_state, checkpoint_dir, f"interrupt_checkpoint_epoch_{epoch+1}.pth")

    finally:
        wandb.finish()

# Sweep configuration for Bayesian optimization
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'temperature': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.999
        },
        'queue_size': {
            'values': [1024, 2048, 4096]
        },
        'epochs': {
            'value': 50  # Reduced for sweep runs
        },
        'base_model': {
            'value': 'convnext_tiny'
        },
        'projection_dim': {
            'value': 128
        },
        'hidden_dim': {
            'value': 512
        },
        'pretrained': {
            'value': False
        }
    }
}

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create and run the sweep
    sweep_id = wandb.sweep(sweep_config, project="MoCoV3-DR")
    wandb.agent(sweep_id, train, count=20)  # Run 20 trials

    # After sweep, manually run final training with best hyperparameters
    # Example (uncomment and fill in best values after sweep):
    # best_config = {
    #     'lr': <best_lr>,
    #     'batch_size': <best_batch_size>,
    #     'temperature': <best_temperature>,
    #     'momentum': <best_momentum>,
    #     'queue_size': <best_queue_size>,
    #     'epochs': 300,
    #     'base_model': 'convnext_tiny',
    #     'projection_dim': 128,
    #     'hidden_dim': 512,
    #     'pretrained': False
    # }
    # train(best_config)