import os
import logging
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR  # Added import for scheduler

import timm
import wandb

from data_pipeline import data_aug, data_set

class MoCoV3Model(nn.Module):
    def __init__(self, base_model='convnext_tiny', projection_dim=128, hidden_dim=512,
                 queue_size=1024, momentum=0.99, pretrained=False):
        super(MoCoV3Model, self).__init__()
        self.m = momentum  # momentum coefficient for key encoder update

        # Build the query encoder with ConvNeXt backbone + projection head
        self.query_encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        feature_dim = self.query_encoder.num_features
        self.query_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

        # Build the key encoder (same architecture)
        self.key_encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        self.key_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

        # Initialize key encoder to match query encoder
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the negative key queue
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(queue_size, projection_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, im_q, im_k):
        # Compute query features
        q = self.query_encoder(im_q)
        q = self.query_projector(q)
        q = nn.functional.normalize(q, dim=1)

        # Compute key features (no grad)
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
        """Update the negative key queue with new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            overflow = (ptr + batch_size) - self.queue_size
            self.queue[ptr:self.queue_size] = keys[:(batch_size - overflow)]
            self.queue[0:overflow] = keys[(batch_size - overflow):]
            self.queue_ptr[0] = overflow
        else:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

# -----------------------------
# 2. MoCo v3 Loss (InfoNCE)
# -----------------------------
def moco_loss(q, k, queue, temperature=0.2):
    """
    Args:
        q: Query features [N, dim]
        k: Key features (positive) [N, dim]
        queue: Negative keys [K, dim]
        temperature: Temperature for scaling logits
    Returns:
        InfoNCE loss
    """
    # Positive logits: (N, 1)
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    # Negative logits: (N, K)
    l_neg = torch.mm(q, queue.T)
    # Combine and apply temperature
    logits = torch.cat([l_pos, l_neg], dim=1)  # (N, 1+K)
    logits /= temperature

    # Labels: positive is index 0
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

# -----------------------------
# 3. Training & Validation
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, temperature, device, epoch, wandb_run):
    model.train()
    running_loss = 0.0
    for i, (im_q, im_k) in enumerate(dataloader):
        im_q = im_q.to(device)
        im_k = im_k.to(device)
        optimizer.zero_grad()

        # Forward pass
        q, k = model(im_q, im_k)
        loss = moco_loss(q, k, model.queue, temperature=temperature)
        loss.backward()
        optimizer.step()

        # Momentum update + enqueue
        model.update_key_encoder()
        with torch.no_grad():
            model.dequeue_and_enqueue(k)

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
# 4. (NEW) Linear Probe & k-NN Evaluation
# -----------------------------
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
    Extract embeddings from the frozen query encoder for each (image, label).
    Return: features (N, embed_dim), labels (N,)
    """
    model.eval()
    all_feats = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device)
        # Use only the query encoder (no projection) or query_encoder+projection
        feats = model.query_encoder(images)  # shape: [B, feature_dim]
        # Optionally apply the projector if you want to evaluate that space
        # feats = model.query_projector(feats)  # uncomment if needed
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_feats, all_labels

def linear_probe_evaluation(model, train_loader, val_loader, device, wandb_run):
    """
    Freeze the query encoder, extract features, and train a linear classifier.
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

def knn_evaluation(model, train_loader, val_loader, device, k=5, wandb_run=None):
    """
    A simple k-NN classifier on top of extracted embeddings.
    """
    from collections import Counter

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
    import numpy as np
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


def main():
    """
    You might try a linear probe or a k-NN classifier on the embeddings after each epoch.
    If accuracy on a downstream task is improving, your model is learning useful representations—
    even if the loss is somewhat flat.
    """

    config = {
        "epochs": 300,
        "batch_size": 64,
        "lr": 5e-4,
        "lr_min": 1e-5,  # Minimum learning rate for cosine scheduler
        "warm_up_epochs": 10,  # Optional: number of warm-up epochs
        "temperature": 0.2,
        "momentum": 0.99,
        "queue_size": 4096,
        "base_model": "convnext_tiny",
        "projection_dim": 256,
        "hidden_dim": 512,
        "pretrained": False
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
    
    # Learning rate scheduler - Cosine Annealing
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config["epochs"] - config["warm_up_epochs"],
        eta_min=config["lr_min"]
    )
    
    checkpoint_path = "model/new/chckpt/moco/checkpoint_epoch_80.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        logging.info("No checkpoint found, starting from scratch")
    
    if config["warm_up_epochs"] > 0:
        initial_lr = config["lr"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = config["lr_min"]  # Start with minimum lr
    
    # Unlabeled data loaders for SSL
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

    # Optional: Labeled data loaders for evaluation
    train_aug = data_aug.MoCoSingleAug(img_size=256)
    probe_train_loader = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_aug,
        batch_size=64,
        num_workers=0,
        sampler=True
    ).get_loader()

    val_aug = data_aug.MoCoSingleAug(img_size=256)
    probe_val_loader = data_set.UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=val_aug,
        batch_size=8,
        num_workers=0,
        sampler=True
    ).get_loader()

    best_val_loss = float('inf')
    start_epoch = 0
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
            
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, 
                                         config["temperature"], device, epoch, wandb_run)
            val_loss = validate(model, valid_loader, config["temperature"], device, epoch, wandb_run)

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
            save_checkpoint(checkpoint_state, "model/new/chckpt/moco", epoch_ckpt)

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_state, "model/new/chckpt/moco", "best_checkpoint.pth")

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
        save_checkpoint(checkpoint_state, "model/new/chckpt/moco", f"interrupt_checkpoint_epoch_{epoch+1}.pth")
    finally:
        wandb_run.finish()

if __name__ == "__main__":
    main()