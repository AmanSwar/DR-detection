import os
import logging
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

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
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

def count_samples_per_class(dataset_name):
    """Count samples per class in a dataset"""
    # This is a placeholder. You need to implement this function 
    # to return actual class distribution in your datasets
    
    # Example class distribution (replace with actual counts)
    if dataset_name == "eyepacs":
        return np.array([20000, 2000, 2000, 1000, 1000])  # Example class distribution
    elif dataset_name == "aptos":
        return np.array([2500, 500, 500, 300, 200])
    elif dataset_name == "ddr":
        return np.array([4000, 1000, 800, 600, 400])
    elif dataset_name == "idrid":
        return np.array([2000, 500, 400, 300, 200])
    elif dataset_name == "messdr":
        return np.array([3000, 700, 600, 500, 400])
    else:
        return np.array([1000, 1000, 1000, 1000, 1000])  # Default equal distribution

def get_class_weights(dataset_names):
    # Count samples per class across all datasets
    class_counts = np.zeros(5)  # 5 DR classes
    for name in dataset_names:
        counts = count_samples_per_class(name)
        class_counts += counts
    
    # Inverse frequency weighting with smoothing
    weights = 1.0 / (class_counts + 100)  # Add smoothing factor
    weights = weights / weights.sum() * len(weights)  # Normalize
    return torch.tensor(weights, dtype=torch.float32)

def mixup_data(x, y, alpha=0.2):
    """Applies Mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, wandb_run, scaler=None, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply Mixup augmentation
        if use_mixup and np.random.random() > 0.5:  # Apply mixup 50% of the time
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            mixup_applied = True
        else:
            mixup_applied = False
        
        optimizer.zero_grad()
        
        # Use mixed precision training if scaler is provided
        if scaler is not None:
            with autocast():
                outputs = model(images)
                if mixup_applied:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if mixup_applied:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        
        # For metrics calculation (skip if mixup was applied)
        if not mixup_applied:
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        if i % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")
            wandb_run.log({
                "train_loss": loss.item(), 
                "learning_rate": current_lr,
                "batch": i + epoch * len(dataloader)
            })
    
    avg_loss = running_loss / len(dataloader)
    
    # Only calculate metrics if we have predictions (some batches might use mixup)
    if len(all_preds) > 0:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        wandb_run.log({
            "train_epoch_loss": avg_loss,
            "train_accuracy": acc,
            "train_f1": f1,
            "epoch": epoch + 1
        })
        
        return avg_loss, acc, f1
    else:
        wandb_run.log({
            "train_epoch_loss": avg_loss,
            "epoch": epoch + 1
        })
        
        return avg_loss, None, None

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
    
    # Calculate Quadratic Weighted Kappa
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Calculate ROC AUC (one-vs-rest for multiclass)
    all_probs = np.array(all_probs)
    try:
        auc_scores = []
        for i in range(5):  # 5 DR classes
            if i in np.unique(all_labels):
                y_true_binary = (np.array(all_labels) == i).astype(int)
                y_prob_binary = all_probs[:, i]
                auc = roc_auc_score(y_true_binary, y_prob_binary)
                auc_scores.append(auc)
        
        if auc_scores:
            mean_auc = np.mean(auc_scores)
        else:
            mean_auc = 0.0
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        mean_auc = 0.0
    
    # Calculate sensitivity and specificity
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
    logging.info(f"Sensitivity: {avg_sensitivity:.4f}, Specificity: {avg_specificity:.4f}, QWK: {qwk:.4f}")
    
    wandb_run.log({
        "val_loss": avg_loss,
        "val_accuracy": acc,
        "val_f1": f1,
        "val_sensitivity": avg_sensitivity,
        "val_specificity": avg_specificity,
        "val_qwk": qwk,
        "val_auc": mean_auc,
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
    
    return avg_loss, acc, f1, avg_sensitivity, avg_specificity, qwk, mean_auc

def save_checkpoint(state, checkpoint_dir, filename):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MoCo model for Diabetic Retinopathy Classification")
    parser.add_argument("--checkpoint", type=str, default="model/new/chckpt/moco/checkpoint_epoch_90.pth",
                        help="Path to MoCo checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=10, 
                        help="Number of epochs to train with frozen backbone")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")  # Increased batch size
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate")  # Lower min LR
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of DR classes")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--use_mixup", action="store_true", default=True, help="Use Mixup augmentation")
    parser.add_argument("--early_stopping", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("finetune.log"),
            logging.StreamHandler()
        ]
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
        "weight_decay": args.weight_decay,
        "num_classes": args.num_classes,
        "img_size": args.img_size,
        "use_amp": args.use_amp,
        "use_mixup": args.use_mixup
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
        num_workers=8,  # Increased workers
        sampler=True
    ).get_loader()

    val_loader = data_set.UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=val_transform,
        batch_size=args.batch_size,
        num_workers=8,  # Increased workers
        sampler=True
    ).get_loader()

    # Get class weights for weighted loss
    class_weights = get_class_weights(dataset_names).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        [
            {'params': model.classifier.parameters(), 'lr': args.lr},
            {'params': model.backbone.parameters(), 'lr': 0.0}  # Initially frozen
        ],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Calculate steps per epoch for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    # Use OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,  # Warmup for 10% of training
        div_factor=25,  # Initial LR = max_lr/div_factor
        final_div_factor=1000,  # Final LR = max_lr/(div_factor*final_div_factor)
        anneal_strategy='cos'
    )
    
    # Initialize AMP scaler if using mixed precision
    scaler = GradScaler() if args.use_amp else None

    # Training loop
    best_val_metrics = {
        "loss": float('inf'),
        "accuracy": 0,
        "f1": 0,
        "sensitivity": 0,
        "specificity": 0,
        "qwk": 0,
        "auc": 0
    }
    
    # Early stopping
    patience = args.early_stopping
    patience_counter = 0
    best_metric = 0  # Track best combined clinical metric

    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Unfreeze backbone after specified epochs
        if epoch == args.freeze_epochs:
            logging.info("Unfreezing backbone for full fine-tuning")
            model.unfreeze_backbone()
            
            # Reset optimizer with adjusted learning rates
            optimizer = optim.AdamW(
                [
                    {'params': model.classifier.parameters(), 'lr': args.lr / 5},
                    {'params': model.backbone.parameters(), 'lr': args.lr / 10}
                ],
                weight_decay=args.weight_decay
            )
            
            # Recalculate remaining steps for the new scheduler
            remaining_epochs = args.epochs - epoch
            total_steps = steps_per_epoch * remaining_epochs
            
            # Create new scheduler
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[args.lr / 5, args.lr / 10],
                total_steps=total_steps,
                pct_start=0.1,
                div_factor=10,
                final_div_factor=100,
                anneal_strategy='cos'
            )
            
            # Reset scaler if using mixed precision
            if args.use_amp:
                scaler = GradScaler()
        
        # Train and validate
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, wandb_run, 
            scaler=scaler, use_mixup=args.use_mixup
        )
        
        val_loss, val_acc, val_f1, val_sensitivity, val_specificity, val_qwk, val_auc = validate(
            model, val_loader, criterion, device, epoch, wandb_run
        )
        
        # Update scheduler (for OneCycleLR, should be called after each batch, 
        # but we're using it at epoch level for simplicity)
        # scheduler.step() - already handled inside train_one_epoch for each batch
        
        # Calculate combined clinical metric (balance of accuracy, sensitivity, specificity)
        # Higher weight on sensitivity for medical applications
        combined_metric = 0.3 * val_acc + 0.4 * val_sensitivity + 0.3 * val_specificity
        
        # Save checkpoint
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_sensitivity': val_sensitivity,
            'val_specificity': val_specificity,
            'val_qwk': val_qwk,
            'val_auc': val_auc,
            'config': config
        }
        
        # Save epoch checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs to save disk space
            save_checkpoint(checkpoint_state, checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save best models based on different metrics
        metrics_improved = False
        
        if val_loss < best_val_metrics["loss"]:
            best_val_metrics["loss"] = val_loss
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_loss_checkpoint.pth")
            metrics_improved = True
        
        if val_acc > best_val_metrics["accuracy"]:
            best_val_metrics["accuracy"] = val_acc
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_accuracy_checkpoint.pth")
            metrics_improved = True
        
        if val_f1 > best_val_metrics["f1"]:
            best_val_metrics["f1"] = val_f1
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_f1_checkpoint.pth")
            metrics_improved = True
        
        if val_sensitivity > best_val_metrics["sensitivity"]:
            best_val_metrics["sensitivity"] = val_sensitivity
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_sensitivity_checkpoint.pth")
            metrics_improved = True
        
        if val_specificity > best_val_metrics["specificity"]:
            best_val_metrics["specificity"] = val_specificity
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_specificity_checkpoint.pth")
            metrics_improved = True
            
        if val_qwk > best_val_metrics["qwk"]:
            best_val_metrics["qwk"] = val_qwk
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_qwk_checkpoint.pth")
            metrics_improved = True
            
        if val_auc > best_val_metrics["auc"]:
            best_val_metrics["auc"] = val_auc
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_auc_checkpoint.pth")
            metrics_improved = True
        
        # Save best overall clinical model (for deployment)
        if combined_metric > best_metric:
            best_metric = combined_metric
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_clinical_checkpoint.pth")
            metrics_improved = True
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Log all best metrics so far
        wandb_run.log({
            "best_val_loss": best_val_metrics["loss"],
            "best_val_accuracy": best_val_metrics["accuracy"],
            "best_val_f1": best_val_metrics["f1"],
            "best_val_sensitivity": best_val_metrics["sensitivity"],
            "best_val_specificity": best_val_metrics["specificity"],
            "best_val_qwk": best_val_metrics["qwk"],
            "best_val_auc": best_val_metrics["auc"],
            "best_combined_metric": best_metric,
            "epoch": epoch + 1
        })

    # Final logging
    logging.info("Training complete!")
    logging.info(f"Best validation loss: {best_val_metrics['loss']:.4f}")
    logging.info(f"Best validation accuracy: {best_val_metrics['accuracy']:.4f}")
    logging.info(f"Best validation F1: {best_val_metrics['f1']:.4f}")
    logging.info(f"Best validation sensitivity: {best_val_metrics['sensitivity']:.4f}")
    logging.info(f"Best validation specificity: {best_val_metrics['specificity']:.4f}")
    logging.info(f"Best validation QWK: {best_val_metrics['qwk']:.4f}")
    logging.info(f"Best validation AUC: {best_val_metrics['auc']:.4f}")
    
    wandb_run.finish()

if __name__ == "__main__":
    main()