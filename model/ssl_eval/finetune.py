import os
import logging
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import wandb

from data_pipeline import data_aug, data_set

class DRClassifier(nn.Module):
    def __init__(self, checkpoint_path, num_classes=5, freeze_backbone=True):
        super(DRClassifier, self).__init__()
        
        # Load the pre-trained MoCo model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        moco_state_dict = checkpoint['model_state_dict']
        
        # Create the backbone model (same as in MoCo)
        config = checkpoint['config']
        self.backbone = timm.create_model(config['base_model'], pretrained=False, num_classes=0)
        
        # Load only the query encoder (backbone) weights
        backbone_state_dict = {}
        for k, v in moco_state_dict.items():
            if k.startswith('query_encoder.'):
                backbone_state_dict[k.replace('query_encoder.', '')] = v
        
        # Load the weights into the backbone
        self.backbone.load_state_dict(backbone_state_dict)
        
        # Freeze backbone if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create classifier head
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, wandb_run):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # For metrics calculation
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        if i % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")
            wandb_run.log({
                "train_loss": loss.item(), 
                "learning_rate": current_lr
            })
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    wandb_run.log({
        "train_epoch_loss": avg_loss,
        "train_accuracy": acc,
        "train_f1": f1,
        "epoch": epoch + 1
    })
    
    return avg_loss, acc, f1

def validate(model, dataloader, criterion, device, epoch, wandb_run):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # For metrics calculation
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    sensitivity = []
    specificity = []
    
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)
    
    logging.info(f"Validation - Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
    logging.info(f"Sensitivity: {avg_sensitivity:.4f}, Specificity: {avg_specificity:.4f}")
    
    wandb_run.log({
        "val_loss": avg_loss,
        "val_accuracy": acc,
        "val_f1": f1,
        "val_sensitivity": avg_sensitivity,
        "val_specificity": avg_specificity,
        "epoch": epoch + 1
    })
    
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Log class-wise metrics
    for i in range(len(sensitivity)):
        wandb_run.log({
            f"sensitivity_class{i}": sensitivity[i],
            f"specificity_class{i}": specificity[i],
            f"f1_class{i}": class_report[str(i)]['f1-score'] if str(i) in class_report else 0,
            "epoch": epoch + 1
        })
    
    return avg_loss, acc, f1, avg_sensitivity, avg_specificity

def save_checkpoint(state, checkpoint_dir, filename):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint: {filepath}")

def get_class_weights(dataset_names):
    # Count samples per class across all datasets
    class_counts = np.zeros(5)  # 5 DR classes
    for name in dataset_names:
        # You'll need to implement this to count samples per class
        counts = count_samples_per_class(name)  
        class_counts += counts
    
    # Inverse frequency weighting
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)  # Normalize
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MoCo model for Diabetic Retinopathy Classification")
    parser.add_argument("--checkpoint", type=str, default="model/new/chckpt/moco/checkpoint_epoch_90.pth",
                        help="Path to MoCo checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=10, 
                        help="Number of epochs to train with frozen backbone")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of DR classes")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create checkpoints directory
    checkpoint_dir = "model/new/chckpt/finetune"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize wandb
    config = {
        "checkpoint": args.checkpoint,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "num_classes": args.num_classes,
        "img_size": args.img_size
    }
    wandb_run = wandb.init(project="MoCoV3-DR-Finetune", config=config)

    # Initialize model
    model = DRClassifier(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        freeze_backbone=True
    ).to(device)

    # Data augmentation and loading
    train_transform = data_aug.TrainTransform(img_size=args.img_size)
    val_transform = data_aug.ValidTransform(img_size=args.img_size)

    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    
    train_loader = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_transform,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=True
    ).get_loader()

    val_loader = data_set.UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=val_transform,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=True
    ).get_loader()

    
    class_weights = get_class_weights(dataset_names).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # Training loop
    best_val_metrics = {
        "loss": float('inf'),
        "accuracy": 0,
        "f1": 0,
        "sensitivity": 0,
        "specificity": 0
    }

    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Unfreeze backbone after specified epochs
        if epoch == args.freeze_epochs:
            logging.info("Unfreezing backbone for full fine-tuning")
            model.unfreeze_backbone()
            # Adjust learning rate for full fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr / 10
        
        # Train and validate
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, wandb_run
        )
        
        val_loss, val_acc, val_f1, val_sensitivity, val_specificity = validate(
            model, val_loader, criterion, device, epoch, wandb_run
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_sensitivity': val_sensitivity,
            'val_specificity': val_specificity,
            'config': config
        }
        
        save_checkpoint(checkpoint_state, checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save best models based on different metrics
        if val_loss < best_val_metrics["loss"]:
            best_val_metrics["loss"] = val_loss
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_loss_checkpoint.pth")
        
        if val_acc > best_val_metrics["accuracy"]:
            best_val_metrics["accuracy"] = val_acc
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_accuracy_checkpoint.pth")
        
        if val_f1 > best_val_metrics["f1"]:
            best_val_metrics["f1"] = val_f1
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_f1_checkpoint.pth")
        
        if val_sensitivity > best_val_metrics["sensitivity"]:
            best_val_metrics["sensitivity"] = val_sensitivity
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_sensitivity_checkpoint.pth")
        
        if val_specificity > best_val_metrics["specificity"]:
            best_val_metrics["specificity"] = val_specificity
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_specificity_checkpoint.pth")

    wandb_run.finish()
    logging.info("Training complete!")

if __name__ == "__main__":
    main()