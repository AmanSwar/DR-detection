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
from torch.optim.lr_scheduler import OneCycleLR #, CosineAnnealingLR # Removed CosineAnnealingLR as it wasn't used
# from torch.nn.utils import clip_grad_norm_ # Removed as it was commented out

import timm
import wandb
from data_pipeline import data_aug, data_set # Assuming these imports are correct

import random
torch.manual_seed(13102021)
np.random.seed(13102021)
random.seed(13102021)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(13102021)

# Based on CBAM
class LesionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(LesionAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Reduced channel complexity slightly for potential efficiency
        reduction_ratio = 16
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # Channel Attention
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        
        channel_att = torch.sigmoid(avg_out + max_out)
        x_channel = x * channel_att
        
        # Spatial Attention
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = torch.sigmoid(self.conv_spatial(spatial_input))
        
        attended_output = x_channel * spatial_att
        return attended_output

# Removed GradeConsistencyHead class

class EnhancedDRClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=False): # Changed default freeze_backbone to False as per original script's main logic
        super(EnhancedDRClassifier, self).__init__()
        # Using convnext_small, pretrained=False as in original code
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0) 
        self.feature_dim = self.backbone.num_features
        self.attention = LesionAttentionModule(self.feature_dim)
        
        # Simplified Classifier (adjust complexity if needed)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512), # Reduced intermediate layer size slightly
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            # nn.Linear(1024, 512), # Removed one layer for simplicity
            # nn.BatchNorm1d(512),
            # nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Removed GradeConsistencyHead initialization
        # Removed prototype buffers

        if freeze_backbone:
             for param in self.backbone.parameters():
                 param.requires_grad = False
            
    def forward(self, x, get_attention=False): # Removed update_prototypes, labels
        features = self.backbone.forward_features(x)
        attended_features = self.attention(features)
        # Global Average Pooling after attention
        h = torch.mean(attended_features, dim=(2, 3)) 
        
        logits = self.classifier(h)
        # Removed grade_head call
        # Removed prototype update logic
        
        if get_attention:
            # Return logits, pooled features, and spatial attention map
            return logits, h, attended_features 
        # Return only logits by default
        return logits


# Removed OrdinalDomainLoss function

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, wandb_run, scaler=None, 
                    scheduler=None): # Removed lambda_consistency
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for i, batch_data in enumerate(dataloader):
        # Assuming dataloader yields (images, labels, *optional_other_data)
        # Adjust unpacking based on your actual data_set implementation
        if len(batch_data) == 3:
             images, labels, _ = batch_data
        elif len(batch_data) == 2:
             images, labels = batch_data
        else:
            raise ValueError("Unexpected data format from DataLoader")

        images = images.to(device)
        labels = labels.to(device)
            
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                # Model forward pass returns only logits now
                logits = model(images) # Removed update_prototypes=True, labels=labels
                # Use standard CrossEntropyLoss
                loss = criterion(logits, labels) 
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # grad_norm = clip_grad_norm_(model.parameters(), max_norm=0.01) # Keep commented if not used
            scaler.step(optimizer)
            scaler.update()
        else:
            # Model forward pass returns only logits now
            logits = model(images) # Removed update_prototypes=True, labels=labels
            # Use standard CrossEntropyLoss
            loss = criterion(logits, labels) 
            loss.backward()
            # grad_norm = clip_grad_norm_(model.parameters(), max_norm=0.01) # Keep commented if not used
            optimizer.step()
        
        # OneCycleLR steps per batch
        if scheduler is not None:
            scheduler.step() 
        
        running_loss += loss.item()
        
        _, predicted = torch.max(logits.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        if i % 10 == 0: # Log every 10 steps
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")
            wandb_run.log({
                "train_loss_step": loss.item(), 
                "learning_rate": current_lr,
                # Removed grad_norm logging
                "batch": i + epoch * len(dataloader) # Global step
            })
    
    avg_loss = running_loss / len(dataloader)
    
    # Calculate epoch metrics if possible
    if len(all_preds) > 0 and len(all_labels) > 0:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) # Added zero_division handling
        wandb_run.log({
            "train_epoch_loss": avg_loss,
            "train_accuracy": acc,
            "train_f1": f1,
            "epoch": epoch + 1
        })
        logging.info(f"Epoch [{epoch+1}] Train Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return avg_loss, acc, f1
    else:
        wandb_run.log({
            "train_epoch_loss": avg_loss,
            "epoch": epoch + 1
        })
        logging.info(f"Epoch [{epoch+1}] Train Avg Loss: {avg_loss:.4f} (Metrics calculation skipped)")
        return avg_loss, None, None


def validate(model, dataloader, criterion, device, epoch, wandb_run): # Removed lambda_consistency
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        # Adjust unpacking based on your validation dataloader structure
        for batch_data in dataloader:
            if len(batch_data) == 3: # Assuming validation might also have 3 elements like train
                images, labels, _ = batch_data
            elif len(batch_data) == 2:
                images, labels = batch_data
            else:
                 raise ValueError("Unexpected data format from Validation DataLoader")

            images = images.to(device)
            labels = labels.to(device)
            
            # Model returns only logits
            logits = model(images) 
            # Use standard CrossEntropyLoss
            loss = criterion(logits, labels) 
            
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    
    # Ensure labels and predictions are available
    if not all_labels or not all_preds:
        logging.warning(f"Validation Epoch {epoch+1}: No labels or predictions collected, skipping metrics.")
        wandb_run.log({"val_loss": avg_loss, "epoch": epoch + 1})
        return avg_loss, 0, 0, 0, 0, 0, 0 # Return default values

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    all_probs = np.array(all_probs)
    num_classes = all_probs.shape[1] # Get number of classes from probs
    unique_labels = np.unique(all_labels)

    # Calculate AUC
    mean_auc = 0.0
    auc_scores = []
    if len(unique_labels) > 1: # Check for multi-class case for AUC
        try:
             # One-vs-Rest AUC calculation
            for i in range(num_classes):
                 # Check if class `i` is present in true labels
                if i in unique_labels:
                    y_true_binary = (np.array(all_labels) == i).astype(int)
                     # Check if there's variance in the binary true labels for this class
                    if len(np.unique(y_true_binary)) > 1:
                        y_prob_binary = all_probs[:, i]
                        auc = roc_auc_score(y_true_binary, y_prob_binary)
                        auc_scores.append(auc)
                    else:
                        auc_scores.append(0.5) # Assign neutral AUC if only one class present
                # else: # If class `i` is not in labels, AUC is undefined/irrelevant for it
                #     pass 
            if auc_scores:
                mean_auc = np.mean(auc_scores)
        except Exception as e:
            logging.warning(f"Could not calculate AUC for epoch {epoch+1}: {e}")
            mean_auc = 0.0 # Default to 0 if calculation fails
    elif len(unique_labels) == 1:
         logging.warning(f"Validation Epoch {epoch+1}: Only one class present in labels, AUC is not well-defined.")
         mean_auc = 0.0 # Or handle as appropriate (e.g., 0.5 if you prefer)


    # Calculate Sensitivity (Recall) and Specificity per class
    sensitivity = []
    specificity = []
    # Ensure cm shape matches num_classes if labels don't cover all classes
    if cm.shape[0] < num_classes:
        cm_full = np.zeros((num_classes, num_classes), dtype=int)
        present_labels = sorted(unique_labels)
        label_map = {lbl: idx for idx, lbl in enumerate(present_labels)}
        for r_idx, r_lbl in enumerate(present_labels):
            for c_idx, c_lbl in enumerate(present_labels):
                cm_full[r_lbl, c_lbl] = cm[r_idx, c_idx]
        cm = cm_full # Use the padded confusion matrix

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        sensitivity_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity.append(sensitivity_i)
        specificity.append(specificity_i)
    
    # Average sensitivity/specificity (macro average)
    avg_sensitivity = np.mean(sensitivity) if sensitivity else 0
    avg_specificity = np.mean(specificity) if specificity else 0
    
    logging.info(f"Validation - Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
    logging.info(f"Avg Sensitivity: {avg_sensitivity:.4f}, Avg Specificity: {avg_specificity:.4f}, QWK: {qwk:.4f}, Avg AUC: {mean_auc:.4f}")
    
    wandb_run.log({
        "val_loss": avg_loss,
        "val_accuracy": acc,
        "val_f1": f1,
        "val_avg_sensitivity": avg_sensitivity, # Log average
        "val_avg_specificity": avg_specificity, # Log average
        "val_qwk": qwk,
        "val_avg_auc": mean_auc, # Log average AUC
        "epoch": epoch + 1
    })
    
    # Log per-class metrics
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    present_labels_str = [str(l) for l in sorted(unique_labels)] # Get string labels present

    for i in range(num_classes):
        class_key = str(i)
        metrics_to_log = {
            f"sensitivity_class_{class_key}": sensitivity[i],
            f"specificity_class_{class_key}": specificity[i],
            "epoch": epoch + 1
        }
         # Check if class exists in report before logging F1
        if class_key in class_report and isinstance(class_report[class_key], dict):
            metrics_to_log[f"f1_class_{class_key}"] = class_report[class_key].get('f1-score', 0)
        else:
             metrics_to_log[f"f1_class_{class_key}"] = 0 # Log 0 if class not present or report malformed

        wandb_run.log(metrics_to_log)

        # Log confusion matrix as image to wandb (optional, can be large)
        # if epoch % 5 == 0: # Log every 5 epochs for example
        #     try:
        #         wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        #             probs=None, y_true=all_labels, preds=all_preds, class_names=[str(i) for i in range(num_classes)])}, step=epoch + 1)
        #     except Exception as e:
        #         logging.warning(f"Could not log confusion matrix: {e}")


    return avg_loss, acc, f1, avg_sensitivity, avg_specificity, qwk, mean_auc

def save_checkpoint(state, checkpoint_dir, filename):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Model for Diabetic Retinopathy Classification")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Max learning rate (for OneCycleLR)")
    # parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate") # Not directly used by OneCycleLR args here
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="Weight decay for optimizer")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of DR classes")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision")
    # parser.add_argument("--use_mixup", action="store_true", default=False, help="Use Mixup augmentation") # Mixup not implemented here
    # Removed lambda_consistency argument
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="Freeze backbone layers during training")
    parser.add_argument("--checkpoint_dir", type=str, default="chckpt/finetune_simplified/run_1", help="Directory to save checkpoints")
    parser.add_argument("--wandb_project", type=str, default="dr_finetune_simplified", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (optional, defaults to generated)")


    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler("simplified_finetune.log"), logging.StreamHandler()]
    )

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.info(f"Checkpoints will be saved in: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Using Automatic Mixed Precision: {args.use_amp}")
    logging.info(f"Freezing Backbone: {args.freeze_backbone}")

    run_name = args.wandb_run_name if args.wandb_run_name else f"simplified_{args.img_size}_lr{args.lr}_bs{args.batch_size}"
    wandb_run = wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=run_name
    )

    model = EnhancedDRClassifier(
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone # Pass freeze argument
    ).to(device)


    train_transform = data_aug.MoCoSingleAug(img_size=args.img_size) # Verify this augmentation pipeline
    val_transform = data_aug.MoCoSingleAug(img_size=args.img_size) # Assuming validation needs basic resize/normalize


    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"] # Example datasets
    
    try:
        train_dataset_provider = data_set.UniformTrainDataloader(
            dataset_names=dataset_names,
            transformation=train_transform,
            batch_size=args.batch_size,
            num_workers=5, # Adjust based on system capabilities
            sampler=True # Assuming weighted sampling is desired
        )
        train_loader = train_dataset_provider.get_loader()

        val_dataset_provider = data_set.UniformValidDataloader(
            dataset_names=dataset_names, # Use same datasets for validation? Or a separate validation set?
            transformation=val_transform,
            batch_size=args.batch_size, # Can use larger batch size for validation if memory allows
            num_workers=3, # Adjust based on system capabilities
            sampler=False # Typically no sampling needed for validation
        )
        val_loader = val_dataset_provider.get_loader()
    except Exception as e:
         logging.error(f"Failed to create DataLoaders: {e}")
         wandb.finish()
         return # Exit if data loading fails

    logging.info(f"Training dataset size: {len(train_loader.dataset)} (estimated, depends on sampler)")
    logging.info(f"Validation dataset size: {len(val_loader.dataset)} (estimated)")


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define Loss Function
    criterion = nn.CrossEntropyLoss()

    # Define Learning Rate Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    logging.info(f"Total training steps: {total_steps}")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,          # Percentage of steps for warm-up
        div_factor=10,          # Initial LR = max_lr / div_factor
        final_div_factor=1000,  # Min LR = max_lr / final_div_factor
        anneal_strategy='cos'   # Cosine annealing
    )

    # Initialize GradScaler for AMP
    scaler = GradScaler() if args.use_amp and torch.cuda.is_available() else None

    # Tracking best metrics
    best_val_metrics = {
        "loss": float('inf'), "accuracy": 0, "f1": 0, 
        "sensitivity": 0, "specificity": 0, "qwk": -1, "auc": 0 # QWK can be negative
    }
    best_combined_metric = 0 
    patience_counter = 0 
    PATIENCE_LIMIT = 20 
    logging.info("Starting training...")
    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, wandb_run,
            scaler=scaler, scheduler=scheduler 
        )
        
        val_loss, val_acc, val_f1, val_sensitivity, val_specificity, val_qwk, val_auc = validate(
            model, val_loader, criterion, device, epoch, wandb_run
        )
        
        combined_metric = 0.4 * val_qwk + 0.2 * val_sensitivity + 0.2 * val_specificity + 0.2 * val_auc 
        
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc, 'val_f1': val_f1,
            'val_sensitivity': val_sensitivity, 'val_specificity': val_specificity,
            'val_qwk': val_qwk, 'val_auc': val_auc,
            'config': vars(args) # Save config used for this run
        }
        
        if (epoch + 1) % 10 == 0: # Save every 10 epochs
            save_checkpoint(checkpoint_state, checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        
        if val_loss < best_val_metrics["loss"]:
            logging.info(f"New best validation loss: {val_loss:.4f}")
            best_val_metrics["loss"] = val_loss
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_loss_checkpoint.pth")

        if val_qwk > best_val_metrics["qwk"]:
            logging.info(f"New best validation QWK: {val_qwk:.4f}")
            best_val_metrics["qwk"] = val_qwk
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_qwk_checkpoint.pth")
            # Update other best metrics when QWK improves (or track them independently)
            best_val_metrics["accuracy"] = val_acc
            best_val_metrics["f1"] = val_f1
            best_val_metrics["sensitivity"] = val_sensitivity
            best_val_metrics["specificity"] = val_specificity
            best_val_metrics["auc"] = val_auc


        if combined_metric > best_combined_metric:
            logging.info(f"New best combined clinical metric: {combined_metric:.4f}")
            best_combined_metric = combined_metric
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_clinical_checkpoint.pth")
            patience_counter = 0 # Reset patience since we improved
        else:
            patience_counter += 1
            logging.info(f"Combined metric did not improve. Patience: {patience_counter}/{PATIENCE_LIMIT}")


        # Log best metrics found so far to wandb
        wandb_run.log({
            "best_val_loss": best_val_metrics["loss"],          
            "best_val_accuracy": best_val_metrics["accuracy"], # Usually tracked alongside the primary metric
            "best_val_f1": best_val_metrics["f1"],             # Usually tracked alongside the primary metric
            "best_val_sensitivity": best_val_metrics["sensitivity"],
            "best_val_specificity": best_val_metrics["specificity"],
            "best_val_qwk": best_val_metrics["qwk"],
            "best_val_auc": best_val_metrics["auc"],
            "best_combined_metric": best_combined_metric,
            "epoch": epoch + 1 # Ensure epoch aligns correctly
        })

        # Early Stopping Check
        if patience_counter >= PATIENCE_LIMIT:
             logging.info(f"Early stopping triggered at epoch {epoch+1} due to lack of improvement in combined metric.")
             break

    # --- End of Training ---
    logging.info("Training complete!")
    logging.info(f"Best validation loss: {best_val_metrics['loss']:.4f}")
    logging.info(f"Best validation QWK: {best_val_metrics['qwk']:.4f} (achieved at epoch {checkpoint_state['epoch'] if 'best_qwk_checkpoint.pth' in os.listdir(checkpoint_dir) else 'N/A'})") # Indicate epoch if possible
    logging.info(f"Best validation accuracy (at best QWK): {best_val_metrics['accuracy']:.4f}")
    logging.info(f"Best validation F1 (at best QWK): {best_val_metrics['f1']:.4f}")
    logging.info(f"Best validation sensitivity (at best QWK): {best_val_metrics['sensitivity']:.4f}")
    logging.info(f"Best validation specificity (at best QWK): {best_val_metrics['specificity']:.4f}")
    logging.info(f"Best validation AUC (at best QWK): {best_val_metrics['auc']:.4f}")
    logging.info(f"Best combined clinical metric: {best_combined_metric:.4f}")

    wandb_run.finish() # Ensure wandb run is closed properly

if __name__ == "__main__":
    main()