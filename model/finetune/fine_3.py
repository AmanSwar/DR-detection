import os
import logging
import argparse
import numpy as np
import random
import time

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, cohen_kappa_score
)
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_

import timm
import wandb
from data_pipeline import data_aug, data_set

# --- Seeding ---
SEED = 13102021
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Model Components ---

# Based on CBAM
class LesionAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(LesionAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        # Spatial attention
        assert kernel_size % 2 == 1, "Kernel size must be odd for spatial attention"
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x_channel = x * channel_att

        # Spatial Attention
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_input))

        return x_channel * spatial_att

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Multiply gradient by -alpha, pass None for alpha's gradient
        return grad_output.neg() * ctx.alpha, None

class GradeConsistencyHead(nn.Module):
    def __init__(self, feature_dim, num_grades=5, dropout_rate=0.4):
        super(GradeConsistencyHead, self).__init__()
        self.grade_predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_grades)
        )
        # Ordinal regression part (predicts thresholds/cumulative logits)
        self.ordinal_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_grades - 1) # Predict K-1 thresholds for K classes
        )

    def forward(self, x):
        logits = self.grade_predictor(x)
        # Ensure ordinal thresholds are monotonically increasing (optional but good practice)
        # Here, we directly predict them. Can be post-processed if needed.
        ordinal_thresholds = self.ordinal_encoder(x)
        return logits, ordinal_thresholds

class EnhancedDRClassifier(nn.Module):
    def __init__(self,  num_classes=5, freeze_backbone=False, dropout_rate=0.5):
        super(EnhancedDRClassifier, self).__init__()
        
        # --- Load MoCo Backbone ---
        # try:
        #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
        #     moco_state_dict = checkpoint['model_state_dict']
        #     config = checkpoint['config']
        #     base_model_name = config.get('base_model') # Default if not found
        #     logging.info(f"Loading backbone: {base_model_name} from MoCo checkpoint.")
        #     self.backbone = timm.create_model(base_model_name, pretrained=False, num_classes=0) # num_classes=0 removes classifier head
        # except FileNotFoundError:
        #     logging.error(f"MoCo checkpoint not found at {checkpoint_path}. Exiting.")
        #     raise
        # except KeyError as e:
        #      logging.error(f"Key error loading checkpoint: {e}. Check checkpoint structure.")
        #      raise

        # # Extract state dict for the query encoder
        # backbone_state_dict = {}
        # for k, v in moco_state_dict.items():
        #     if k.startswith('query_encoder.'):
        #          # Remove the 'query_encoder.' prefix
        #         new_k = k.replace('query_encoder.', '', 1)
        #         backbone_state_dict[new_k] = v
        #     # Handle potential older checkpoints without 'query_encoder.' prefix
        #     elif not (k.startswith('key_encoder.') or k.startswith('queue') or k.startswith('queue_ptr')):
        #          # Assume it belongs to the backbone if not key encoder or queue
        #          backbone_state_dict[k] = v


        # # Load state dict (handle potential mismatches)
        # msg = self.backbone.load_state_dict(backbone_state_dict, strict=False)
       
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # --- Attention Module ---
        self.attention = LesionAttentionModule(self.feature_dim)

        # --- Main Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self.grade_head = GradeConsistencyHead(self.feature_dim, num_grades=num_classes, dropout_rate=dropout_rate - 0.1) # Slightly less dropout maybe
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True), # ReLU often used in DANN heads
            nn.Linear(256, 5) # Assuming 5 domains based on dataset_names
        )

        # --- Prototypes (Optional, for analysis or future loss terms) ---
        self.register_buffer('prototypes', torch.zeros(num_classes, self.feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))

        # --- Initialize Weights for New Layers ---
        self._initialize_weights()

    def _initialize_weights(self):
        logging.info("Initializing weights for classifier, grade_head, and domain_classifier.")
        for module in [self.classifier, self.grade_head, self.domain_classifier, self.attention]: # Initialize attention too
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # Kaiming init for linear layers often works well with ReLU/GELU
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d) and m.bias is not None: # For attention conv layers if they have bias
                     nn.init.constant_(m.bias, 0)


    def forward(self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None):
        # --- Feature Extraction ---
        if self.freeze_backbone:
             with torch.no_grad(): # Ensure no gradients calculated for frozen backbone
                 features = self.backbone.forward_features(x)
        else:
            features = self.backbone.forward_features(x) # Pass features directly from backbone

        # --- Attention ---
        attended_features = self.attention(features)
        # Global Average Pooling after attention
        h = torch.mean(attended_features, dim=(2, 3)) # Shape: (batch_size, feature_dim)

        # --- Main Classifier Output ---
        logits = self.classifier(h)

        # --- Auxiliary Outputs ---
        grade_outputs = self.grade_head(h) # Returns (logits, ordinal_thresholds)

        # --- Domain Classifier Output (with Gradient Reversal) ---
        domain_logits = None
        if alpha > 0:
            reversed_features = GradientReversal.apply(h, alpha)
            domain_logits = self.domain_classifier(reversed_features)

        # --- Prototype Update (Non-differentiable) ---
        if update_prototypes and labels is not None and self.training: # Only update during training
            with torch.no_grad():
                for i, label in enumerate(labels):
                    # Ensure label is within bounds
                    if 0 <= label < self.prototypes.size(0):
                        current_count = self.prototype_counts[label].item()
                        # Use momentum update for stability (e.g., momentum=0.99)
                        momentum = 0.99
                        if current_count == 0: # First sample for this class
                             self.prototypes[label] = h[i].clone()
                        else:
                            self.prototypes[label] = momentum * self.prototypes[label] + (1 - momentum) * h[i]
                        # Increment count (using item() and direct assignment avoids in-place issues)
                        self.prototype_counts[label] = current_count + 1
                    else:
                        logging.warning(f"Label {label.item()} out of bounds for prototype update.")

        if get_attention:
            return logits, grade_outputs, domain_logits, h, attended_features
        else:
            return logits, grade_outputs, domain_logits

    def unfreeze_backbone(self):
        if self.freeze_backbone:
            logging.info("Unfreezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze_backbone = False # Update state


# --- Loss Function ---

def OrdinalDomainLoss(outputs, labels, grade_outputs=None, domain_logits=None, domain_labels=None,
                      lambda_consistency=0.1, lambda_domain=0.05, ordinal_weight=0.3, num_classes=5):
    """
    Computes the combined loss: Main CE + Grade Consistency (Ordinal) + Domain Adversarial.

    Args:
        outputs: Main classifier logits (batch_size, num_classes).
        labels: Ground truth labels (batch_size).
        grade_outputs: Tuple (grade_logits, ordinal_thresholds) from GradeConsistencyHead.
        domain_logits: Domain classifier logits (batch_size, num_domains).
        domain_labels: Ground truth domain labels (batch_size).
        lambda_consistency: Weight for the grade consistency loss.
        lambda_domain: Weight for the domain adversarial loss.
        ordinal_weight: Weight for the ordinal threshold loss within consistency loss.
        num_classes: Number of target classes.

    Returns:
        Total combined loss (scalar tensor).
    """
    # --- Main Classification Loss ---
    main_criterion = nn.CrossEntropyLoss()
    main_loss = main_criterion(outputs, labels)
    loss = main_loss
    consistency_loss_val = 0.0 # For logging
    domain_loss_val = 0.0 # For logging


    # --- Grade Consistency Loss ---
    if grade_outputs is not None and lambda_consistency > 0:
        grade_logits, ordinal_thresholds = grade_outputs
        batch_size = labels.size(0)

        # 1. Standard CE loss on the grade predictor (Treat as multi-class)
        # grade_ce_loss = main_criterion(grade_logits, labels) # Option 1: Simple CE

        # 2. Binary Cross Entropy for cumulative probabilities (Ordinal approach 1)
        targets_cumulative = torch.zeros_like(grade_logits)
        for i in range(batch_size):
            if 0 <= labels[i] < num_classes: # Ensure label is valid
                targets_cumulative[i, :labels[i]+1] = 1.0 # Target is 1 for classes <= true label
        # Use BCEWithLogitsLoss for numerical stability
        consistency_loss_bce = F.binary_cross_entropy_with_logits(grade_logits, targets_cumulative, reduction='mean')

        consistency_loss = consistency_loss_bce # Start with BCE part

        # 3. Ordinal Threshold Loss (Ordinal approach 2 - CORAL-like or similar)
        # Predict P(y > k) using thresholds. Target is 1 if true_label > k, else 0.
        if ordinal_thresholds is not None and ordinal_weight > 0:
            ordinal_targets = torch.zeros_like(ordinal_thresholds) # Shape (batch_size, num_classes - 1)
            for i in range(batch_size):
                 for k in range(num_classes - 1):
                     if labels[i] > k:
                         ordinal_targets[i, k] = 1.0
            # Use BCEWithLogitsLoss for the thresholds as well
            ordinal_loss_bce = F.binary_cross_entropy_with_logits(ordinal_thresholds, ordinal_targets, reduction='mean')

            # Combine the two ordinal approaches
            consistency_loss = (1.0 - ordinal_weight) * consistency_loss_bce + ordinal_weight * ordinal_loss_bce

        loss += lambda_consistency * consistency_loss
        consistency_loss_val = consistency_loss.item()


    # --- Domain Adversarial Loss ---
    if domain_logits is not None and domain_labels is not None and lambda_domain > 0:
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(domain_logits, domain_labels)
        loss += lambda_domain * domain_loss
        domain_loss_val = domain_loss.item()

    # Return total loss and optionally components for logging
    return loss # , main_loss.item(), consistency_loss_val, domain_loss_val


# --- Training & Validation Loops ---
def train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs, wandb_run, scaler=None,
                    lambda_consistency=0.1, lambda_domain=0.05, ordinal_weight=0.3,
                    domain_adaptation=True, max_grad_norm=1.0, num_classes=5, scheduler=None):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    total_steps = len(dataloader)

    p = float(epoch) / num_epochs
    alpha = 2. / (1. + np.exp(-10. * p)) - 1 if domain_adaptation else 0.0

    start_time = time.time()

    for i, batch_data in enumerate(dataloader):
        if len(batch_data) == 3:
            images, labels, domain_labels = batch_data
        else:
            images, labels = batch_data
            domain_labels = None

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if domain_labels is not None:
            domain_labels = domain_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits, grade_outputs, domain_logits = model(images, alpha=alpha, update_prototypes=True, labels=labels)
                loss = OrdinalDomainLoss(
                    logits, labels,
                    grade_outputs=grade_outputs,
                    domain_logits=domain_logits,
                    domain_labels=domain_labels,
                    lambda_consistency=lambda_consistency,
                    lambda_domain=lambda_domain,
                    ordinal_weight=ordinal_weight,
                    num_classes=num_classes
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, grade_outputs, domain_logits = model(images, alpha=alpha, update_prototypes=True, labels=labels)
            loss = OrdinalDomainLoss(
                logits, labels,
                grade_outputs=grade_outputs,
                domain_logits=domain_logits,
                domain_labels=domain_labels,
                lambda_consistency=lambda_consistency,
                lambda_domain=lambda_domain,
                ordinal_weight=ordinal_weight,
                num_classes=num_classes
            )
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Step the scheduler after each batch for OneCycleLR
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        if (i + 1) % 20 == 0 or i == total_steps - 1:
            current_lr = optimizer.param_groups[0]['lr']
            batch_time = time.time() - start_time
            eta_seconds = batch_time / (i + 1) * (total_steps - (i + 1))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{total_steps}] Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.6f} | GradNorm: {grad_norm:.4f} | ETA: {eta_str}"
            )
            if wandb_run:
                wandb_run.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/grad_norm_clipped": grad_norm,
                    "train/domain_alpha": alpha,
                    "step": i + epoch * total_steps
                })
            start_time = time.time()

    avg_loss = running_loss / total_steps
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    logging.info(f"Epoch [{epoch+1}/{num_epochs}] Train Summary - Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, QWK: {qwk:.4f}")
    if wandb_run:
        wandb_run.log({
            "train/epoch_loss": avg_loss,
            "train/accuracy": acc,
            "train/f1_weighted": f1,
            "train/qwk": qwk,
            "epoch": epoch + 1
        })

    return avg_loss, acc, f1, qwk


def validate(model, dataloader, device, epoch, num_epochs, wandb_run,
             lambda_consistency=0.1, ordinal_weight=0.3, num_classes=5):
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = [] # Store probabilities for AUC calculation

    with torch.no_grad(): # Disable gradient calculations
        for batch_data in dataloader:
             # Adjust based on your specific dataloader structure
            if len(batch_data) == 3: # Might still have domain label placeholder
                images, labels, _ = batch_data
            else:
                 images, labels = batch_data

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass (no alpha, no prototype update in validation)
            logits, grade_outputs, _ = model(images, alpha=0.0, update_prototypes=False)

            # Calculate validation loss (optional, but good for monitoring)
            # Note: Domain loss is usually not included in validation loss.
            loss = OrdinalDomainLoss(
                logits, labels,
                grade_outputs=grade_outputs,
                domain_logits=None, domain_labels=None, # No domain loss in validation
                lambda_consistency=lambda_consistency,
                lambda_domain=0.0, # Ensure domain loss weight is 0
                ordinal_weight=ordinal_weight,
                num_classes=num_classes
            )

            running_loss += loss.item()

            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Calculate Metrics ---
    avg_loss = running_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    try:
        cm = confusion_matrix(all_labels, all_preds)
        # Calculate sensitivity (recall) and specificity per class
        sensitivity = []
        specificity = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fp - fn

            sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        avg_sensitivity = np.mean(sensitivity)
        avg_specificity = np.mean(specificity)
    except Exception as e:
        logging.warning(f"Could not calculate confusion matrix based metrics: {e}")
        cm = None
        avg_sensitivity = 0.0
        avg_specificity = 0.0
        sensitivity = [0.0] * num_classes
        specificity = [0.0] * num_classes


    # Calculate AUC (Macro average One-vs-Rest)
    try:
        # Ensure all classes are present, otherwise AUC might fail or be misleading
        present_classes = np.unique(all_labels)
        if len(present_classes) == num_classes and all_probs.shape[1] == num_classes:
             auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        elif len(present_classes) > 1 and all_probs.shape[1] == num_classes:
             logging.warning(f"Only {len(present_classes)}/{num_classes} classes present in validation batch. AUC might be unreliable.")
             # Calculate OvR AUC only for present classes if possible, or report 0
             try:
                auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro', labels=present_classes)
             except ValueError: # Handle cases where AUC is undefined for a class
                auc_macro_ovr = 0.0
        else:
             logging.warning("Not enough classes present or probability shape mismatch for AUC calculation.")
             auc_macro_ovr = 0.0
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        auc_macro_ovr = 0.0


    # --- Logging ---
    logging.info(
        f"Validation - Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, "
        f"F1(W): {f1_weighted:.4f}, QWK: {qwk:.4f}"
    )
    logging.info(
        f"Validation - Avg Sensitivity: {avg_sensitivity:.4f}, Avg Specificity: {avg_specificity:.4f}, AUC(Macro-OvR): {auc_macro_ovr:.4f}"
     )

    if wandb_run:
        log_dict = {
            "val/epoch_loss": avg_loss,
            "val/accuracy": acc,
            "val/f1_weighted": f1_weighted,
            "val/qwk": qwk,
            "val/avg_sensitivity": avg_sensitivity,
            "val/avg_specificity": avg_specificity,
            "val/auc_macro_ovr": auc_macro_ovr,
            "epoch": epoch + 1
        }
        # Log per-class metrics
        report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        for i in range(num_classes):
             log_dict[f"val/sensitivity_class{i}"] = sensitivity[i]
             log_dict[f"val/specificity_class{i}"] = specificity[i]
             cls_report = report_dict.get(str(i), {})
             log_dict[f"val/f1_class{i}"] = cls_report.get('f1-score', 0.0)
             log_dict[f"val/precision_class{i}"] = cls_report.get('precision', 0.0)
             log_dict[f"val/recall_class{i}"] = cls_report.get('recall', 0.0) # Recall is sensitivity

        wandb_run.log(log_dict)

        # Log confusion matrix as image (optional)
        # try:
        #     if cm is not None:
        #         wandb.log({"val/confusion_matrix": wandb.Image(cm_figure)}) # Need function to plot cm
        # except Exception as e:
        #     logging.warning(f"Could not log confusion matrix to W&B: {e}")


    return avg_loss, acc, f1_weighted, avg_sensitivity, avg_specificity, qwk, auc_macro_ovr


# --- Utility Functions ---

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth", is_best=False):
    """Saves checkpoint."""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    status = "Best" if is_best else "Regular"
    logging.info(f"Saved {status} checkpoint: {filepath}")
    # Optionally, save the best checkpoint with a fixed name like 'best_model.pth'
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, f"best_{filename.split('_checkpoint')[0]}_model.pth") # e.g., best_qwk_model.pth
        torch.save(state, best_filepath)
        logging.info(f"Saved as overall best model: {best_filepath}")


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Fine-tune MoCo model for Diabetic Retinopathy Classification")
    # Existing arguments...
    parser.add_argument("--output_dir", type=str, default="chckpt/finetune_nofreeze/fine_3", help="Directory to save checkpoints and logs")
    parser.add_argument("--dataset_names", nargs='+', default=["eyepacs", "aptos", "ddr", "idrid", "messdr"], help="List of dataset names to use")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for training and validation")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Number of epochs to train only the heads before unfreezing backbone")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak Learning rate for OneCycleLR")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of DR classes")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for classifier heads")
    parser.add_argument("--lambda_consistency", type=float, default=0.1, help="Weight for grade consistency loss")
    parser.add_argument("--ordinal_weight", type=float, default=0.3, help="Weight for ordinal threshold loss")
    parser.add_argument("--domain_adaptation", action="store_true", default=True, help="Use domain adaptation (DANN)")
    parser.add_argument("--lambda_domain", type=float, default=0.05, help="Weight for domain adversarial loss")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True, help="Use automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--wandb_project", type=str, default="DR_FineTuning", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")

    # New argument for resuming
    parser.add_argument("--resume", type=str, default=None, help="chckpt/finetune_nofreeze/fine_3/best_loss_checkpoint.pth")

    args = parser.parse_args()

    # --- Setup Output Directory and Logging ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "finetune.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info("Starting Enhanced DR Fine-tuning Script")
    logging.info(f"Arguments: {vars(args)}")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- WandB Setup ---
    run_name = args.wandb_run_name or f"ft_{args.img_size}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}" + \
               (f"_freeze{args.freeze_epochs}" if args.freeze_epochs > 0 else "") + \
               ("_da" if args.domain_adaptation else "")
    try:
        wandb_run = wandb.init(project=args.wandb_project, config=vars(args), name=run_name)
        logging.info(f"WandB run initialized: {run_name}")
    except Exception as e:
        logging.warning(f"Could not initialize WandB: {e}. Proceeding without W&B logging.")
        wandb_run = None

    # --- Model Initialization ---
    model = EnhancedDRClassifier(
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_epochs > 0,
        dropout_rate=args.dropout_rate
    ).to(device)

    # --- Data Loaders ---
    train_transform = data_aug.MoCoSingleAug(img_size=args.img_size)
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = data_set.UniformTrainDataloader(
        dataset_names=args.dataset_names,
        transformation=train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=True
    ).get_loader()
    val_loader = data_set.UniformValidDataloader(
        dataset_names=args.dataset_names,
        transformation=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=True
    ).get_loader()
    logging.info(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # --- Optimizer ---
    backbone_params = model.backbone.parameters()
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.') and p.requires_grad]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr / 10.0},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    # --- Scheduler ---
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    if args.freeze_epochs > 0:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=steps_per_epoch * args.freeze_epochs,
            pct_start=0.1, div_factor=10, final_div_factor=1000, anneal_strategy='cos'
        )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[args.lr / 10.0, args.lr],
            total_steps=total_steps,
            pct_start=0.1, div_factor=10, final_div_factor=1000, anneal_strategy='cos'
        )

    # --- AMP Scaler ---
    scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
    if scaler:
        logging.info("Using Automatic Mixed Precision (AMP).")

    # --- Resume from Checkpoint ---

    print("\n")
    print("loading checkpoint...")
    resume_chkpt = "chckpt/finetune_nofreeze/fine_3/best_loss_checkpoint.pth"
    start_epoch = 0
    logging.info(f"Resuming from checkpoint: {resume_chkpt}")
    checkpoint = torch.load(resume_chkpt, map_location=device , weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']

    logging.info(f"Resuming training from epoch {start_epoch}")
    print("\n")
    # --- Training Loop ---
    best_val_metrics = {
        "loss": float('inf'), "accuracy": 0, "f1": 0, "sensitivity": 0,
        "specificity": 0, "qwk": -1.0, "auc": 0
    }
    patience_counter = 0
    best_metric_combined = 0.0
    PATIENCE_LIMIT = 20

    logging.info("--- Starting Training Loop ---")
    start_training_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            logging.info(f"--- Unfreezing backbone at epoch {epoch+1} ---")
            model.unfreeze_backbone()
            backbone_params = model.backbone.parameters()
            head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.') and p.requires_grad]
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': args.lr / 10.0},
                {'params': head_params, 'lr': args.lr}
            ], weight_decay=args.weight_decay)
            remaining_steps = steps_per_epoch * (args.epochs - args.freeze_epochs)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[args.lr / 10.0, args.lr],
                total_steps=remaining_steps,
                pct_start=0.1, div_factor=10, final_div_factor=1000, anneal_strategy='cos'
            )
            logging.info("Re-initialized optimizer and scheduler for full model fine-tuning.")

        train_loss, train_acc, train_f1, train_qwk = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs, wandb_run,
            scaler=scaler, lambda_consistency=args.lambda_consistency,
            lambda_domain=args.lambda_domain, ordinal_weight=args.ordinal_weight,
            domain_adaptation=args.domain_adaptation, max_grad_norm=args.max_grad_norm,
            num_classes=args.num_classes, scheduler=scheduler
        )

        val_loss, val_acc, val_f1, val_sensitivity, val_specificity, val_qwk, val_auc = validate(
            model, val_loader, device, epoch, args.epochs, wandb_run,
            lambda_consistency=args.lambda_consistency, ordinal_weight=args.ordinal_weight,
            num_classes=args.num_classes
        )

        # --- Checkpoint Saving ---
        combined_metric = 0.3 * val_acc + 0.4 * val_sensitivity + 0.3 * val_specificity
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss, 'val_accuracy': val_acc, 'val_f1': val_f1, 'val_qwk': val_qwk, 'val_auc': val_auc,
            'config': vars(args)
        }

        if (epoch + 1) % 10 == 0:
            save_checkpoint(checkpoint_state, output_dir, f"checkpoint_epoch_{epoch+1}.pth")

        is_best_loss = val_loss < best_val_metrics["loss"]
        is_best_qwk = val_qwk > best_val_metrics["qwk"]
        is_best_combined = combined_metric > best_metric_combined

        if is_best_loss:
            best_val_metrics["loss"] = val_loss
            save_checkpoint(checkpoint_state, output_dir, "best_loss_checkpoint.pth", is_best=True)
        if is_best_qwk:
            best_val_metrics["qwk"] = val_qwk
            save_checkpoint(checkpoint_state, output_dir, "best_qwk_checkpoint.pth", is_best=True)
        if is_best_combined:
            logging.info(f"*** New best combined metric: {combined_metric:.4f} (previous: {best_metric_combined:.4f})")
            best_metric_combined = combined_metric
            save_checkpoint(checkpoint_state, output_dir, "best_clinical_checkpoint.pth", is_best=True)
            patience_counter = 0
        else:
            patience_counter += 1

        best_val_metrics["accuracy"] = max(best_val_metrics["accuracy"], val_acc)
        best_val_metrics["f1"] = max(best_val_metrics["f1"], val_f1)
        best_val_metrics["sensitivity"] = max(best_val_metrics["sensitivity"], val_sensitivity)
        best_val_metrics["specificity"] = max(best_val_metrics["specificity"], val_specificity)
        best_val_metrics["auc"] = max(best_val_metrics["auc"], val_auc)

        if wandb_run:
            wandb_run.log({
                "val/best_loss_so_far": best_val_metrics["loss"],
                "val/best_qwk_so_far": best_val_metrics["qwk"],
                "val/best_combined_metric_so_far": best_metric_combined,
                "val/patience_counter": patience_counter,
                "epoch": epoch + 1
            })

    # --- End of Training ---
    total_training_time = time.time() - start_training_time
    logging.info(f"Total Training Time: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")
    logging.info("--- Best Validation Metrics Achieved ---")
    for metric, value in best_val_metrics.items():
        logging.info(f"Best {metric.capitalize()}: {value:.4f}")
    logging.info(f"Best Combined Clinical Metric: {best_metric_combined:.4f}")

    save_checkpoint(checkpoint_state, output_dir, "final_checkpoint.pth")
    if wandb_run:
        wandb_run.summary.update({
            f"best_val_{k}": v for k, v in best_val_metrics.items()
        })
        wandb_run.summary["best_combined_clinical_metric"] = best_metric_combined
        wandb_run.summary["total_training_time_seconds"] = total_training_time
        wandb_run.finish()


if __name__ == "__main__":
    main()