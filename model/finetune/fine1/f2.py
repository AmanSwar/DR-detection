import os
import logging
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn # Import update_bn

import torchvision
import timm
import wandb

# Assuming data_pipeline, data_aug, data_set modules are in the correct path
from data_pipeline import data_aug, data_set

# --- Model Definitions ---

class EnhancedLesionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedLesionAttentionModule, self).__init__()
        # Channel attention path
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Reduced complexity slightly in FC layers for stability
        hidden_dim = max(16, in_channels // 16)
        self.fc1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)

        # Spatial attention path
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Original input shape check removed for flexibility, assuming 4D input
        # Channel attention
        avg_pool_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_pool_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid_channel(avg_pool_out + max_pool_out)
        x_channel = x * channel_att

        # Spatial attention
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_input))

        return x_channel * spatial_att


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Clamp gradients to prevent excessively large negative gradients
        # grad_clamped = torch.clamp(grad_output, -1.0, 1.0) # Optional: clamp gradient magnitude
        # return grad_clamped.neg() * ctx.alpha, None
        return grad_output.neg() * ctx.alpha, None

class EnhancedGradeConsistencyHead(nn.Module):
    def __init__(self, feature_dim, num_grades=5):
        super(EnhancedGradeConsistencyHead, self).__init__()
        self.grade_predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_grades)
        )
        # Ordinal relationship encoder (Optional, can be removed if causing instability)
        self.ordinal_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_grades - 1)
        )

    def forward(self, x):
        logits = self.grade_predictor(x)
        ordinal_thresholds = self.ordinal_encoder(x)
        return logits, ordinal_thresholds


class EnhancedDRClassifier(nn.Module):
    def __init__(self, checkpoint_path, num_classes=5, freeze_backbone=True):
        super(EnhancedDRClassifier, self).__init__()
        logging.info(f"Loading MoCo checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        moco_state_dict = checkpoint['model_state_dict']
        config = checkpoint['config']

        logging.info(f"Creating backbone: {config['base_model']}")
        self.backbone = timm.create_model(config['base_model'], pretrained=False, num_classes=0) # num_classes=0 removes head

        backbone_state_dict = {}
        for k, v in moco_state_dict.items():
            if k.startswith('query_encoder.'):
                backbone_state_dict[k.replace('query_encoder.', '')] = v
        missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        logging.info(f"Backbone loaded. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        if unexpected_keys: logging.warning(f"Unexpected keys in backbone load: {unexpected_keys[:5]}...")


        if freeze_backbone:
            logging.info("Freezing backbone parameters initially.")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.feature_dim = self.backbone.num_features
        logging.info(f"Backbone feature dimension: {self.feature_dim}")

        self.attention = EnhancedLesionAttentionModule(self.feature_dim)

        # Main classifier head
        self.classifier = nn.Sequential(
            # Consider adding BatchNorm before activation
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Enhanced grade consistency head
        self.grade_head = EnhancedGradeConsistencyHead(self.feature_dim, num_grades=num_classes)

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256), # Add BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), # Add BatchNorm
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)  # Num datasets
        )

        # Prototypes (Optional, can be removed if not crucial/stable)
        self.register_buffer('prototypes', torch.zeros(num_classes, self.feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))

        self._initialize_weights()

    def _initialize_weights(self):
        logging.info("Initializing weights for classifier heads.")
        for module in [self.classifier, self.grade_head.grade_predictor,
                      self.grade_head.ordinal_encoder, self.domain_classifier, self.attention]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                     # Initialize Conv2d layers in attention properly
                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                     if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    def forward(self, x, alpha=0.0, update_prototypes=False, labels=None):
        features = self.backbone.forward_features(x) # Shape: (batch, channels, H, W)
        
        # Global Average Pooling before attention can sometimes stabilize
        # features_pooled = torch.mean(features, dim=(2, 3)) # Shape: (batch, channels)
        # attended_features = self.attention(features_pooled.unsqueeze(-1).unsqueeze(-1)) # Apply attention after pooling
        
        attended_features = self.attention(features) # Apply attention on feature maps

        # Pooling *after* attention
        h = torch.mean(attended_features, dim=(2, 3)) # Shape: (batch, channels)

        # Main class prediction
        logits = self.classifier(h)

        # Grade consistency prediction
        grade_outputs = self.grade_head(h) # Returns (grade_logits, ordinal_thresholds)

        # Prototype update (Optional)
        if update_prototypes and labels is not None and hasattr(self, 'prototypes'):
             with torch.no_grad():
                 for i, label in enumerate(labels):
                     label_idx = label.item() # Ensure label is an integer index
                     current_count = self.prototype_counts[label_idx].item()
                     new_count = current_count + 1
                     # Numerically stable update
                     self.prototypes[label_idx] = (self.prototypes[label_idx] * (current_count / new_count) +
                                                  h[i] * (1 / new_count))
                     self.prototype_counts[label_idx] += 1

        domain_logits = None
        if alpha > 0:
            reversed_features = GradientReversal.apply(h, alpha)
            domain_logits = self.domain_classifier(reversed_features)

        return logits, grade_outputs, domain_logits


    def unfreeze_backbone(self):
        logging.info("Unfreezing all backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = True

# --- Augmentations & Loss ---

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if batch_size <= 1: # Cannot mixup with batch size 1
        return x, y, y, 1.0

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Optional: Label Smoothing Cross Entropy
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def combined_loss(outputs, labels, grade_outputs=None, domain_logits=None, domain_labels=None,
                  lambda_consistency=0.3, lambda_domain=0.1, lambda_ordinal=0.1,
                  use_label_smoothing=False, smoothing=0.1):

    if use_label_smoothing:
        main_criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        main_criterion = nn.CrossEntropyLoss()

    main_loss = main_criterion(outputs, labels)
    total_loss = main_loss
    loss_breakdown = {'main_loss': main_loss.item()}

    # Enhanced grade consistency loss
    if grade_outputs is not None:
        grade_logits, ordinal_thresholds = grade_outputs
        batch_size = labels.size(0)

        # BCE loss for grade prediction (treat as multi-label)
        targets = torch.zeros_like(grade_logits).to(labels.device)
        for i in range(batch_size):
            targets[i, :labels[i]+1] = 1.0 # All grades up to the true label are 'active'

        consistency_loss_bce = F.binary_cross_entropy_with_logits(grade_logits, targets)
        loss_breakdown['consistency_loss_bce'] = consistency_loss_bce.item()
        total_loss += lambda_consistency * consistency_loss_bce

        # Ordinal regression loss for thresholds (Optional but recommended)
        if ordinal_thresholds is not None:
            ordinal_targets = torch.zeros_like(ordinal_thresholds).to(labels.device)
            for i in range(batch_size):
                # Target is 1 if true label > threshold index k
                for k in range(ordinal_thresholds.size(1)): # num_classes - 1 thresholds
                    if labels[i] > k:
                        ordinal_targets[i, k] = 1.0

            ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_thresholds, ordinal_targets)
            loss_breakdown['ordinal_loss'] = ordinal_loss.item()
            total_loss += lambda_ordinal * ordinal_loss # Use separate lambda

    # Domain classification loss
    if domain_logits is not None and domain_labels is not None:
        # Ensure domain_labels are long type
        if domain_labels.dtype != torch.long:
             domain_labels = domain_labels.long()
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(domain_logits, domain_labels)
        loss_breakdown['domain_loss'] = domain_loss.item()
        total_loss += lambda_domain * domain_loss

    loss_breakdown['total_loss'] = total_loss.item()
    return total_loss, loss_breakdown


# --- Training & Validation ---

def train_one_epoch(model, dataloader, optimizer, device, epoch, wandb_run, scaler=None,
                   lambda_consistency=0.3, lambda_domain=0.1, lambda_ordinal=0.1,
                   domain_adaptation=True, domain_alpha_max=1.0, domain_alpha_ramp_epochs=20, # Control alpha ramp
                   use_mixup=True, mixup_alpha=0.4, grad_clip_norm=1.0,
                   use_label_smoothing=False, smoothing=0.1):
    model.train()
    running_loss = 0.0
    total_main_loss = 0.0
    total_consistency_loss = 0.0
    total_ordinal_loss = 0.0
    total_domain_loss = 0.0
    num_samples = 0

    all_labels = []
    all_preds = []

    # Calculate domain adaptation alpha: linear ramp-up
    if domain_adaptation:
        alpha = min(domain_alpha_max, (domain_alpha_max / domain_alpha_ramp_epochs) * epoch)
    else:
        alpha = 0.0

    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 10) # Log ~10 times per epoch

    for i, batch_data in enumerate(dataloader):
        # Adjust based on your dataloader structure
        if len(batch_data) == 3:
            images, labels, domain_labels = batch_data
            domain_labels = domain_labels.to(device)
        else: # Assume no domain labels if not provided
            images, labels = batch_data
            domain_labels = None
            if domain_adaptation:
                logging.warning("Domain adaptation enabled but no domain labels found in batch.")

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True) # More efficient zeroing

        # Apply mixup if enabled
        apply_mixup_this_batch = use_mixup and np.random.rand() < 0.5 # 50% chance
        if apply_mixup_this_batch and images.size(0) > 1:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            mixed_target = True
        else:
            mixed_target = False

        autocast_enabled = scaler is not None
        with autocast(enabled=autocast_enabled):
            # Pass labels only if not mixup for prototype update
            proto_labels = labels if not mixed_target else None
            logits, grade_outputs, domain_logits = model(images, alpha=alpha, update_prototypes=not mixed_target, labels=proto_labels)

            if mixed_target:
                # Handle mixup loss calculation
                if use_label_smoothing:
                    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
                else:
                    criterion = nn.CrossEntropyLoss()

                loss_val = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                # For simplicity, maybe skip auxiliary losses with mixup, or apply them carefully
                # e.g., apply domain loss to the non-mixed domain labels if available
                if domain_logits is not None and domain_labels is not None:
                     if domain_labels.dtype != torch.long: domain_labels = domain_labels.long()
                     domain_criterion = nn.CrossEntropyLoss()
                     loss_val += lambda_domain * domain_criterion(domain_logits, domain_labels) # Use original domain labels
                loss_breakdown = {'total_loss': loss_val.item()} # Simplified breakdown for mixup
            else:
                # Regular loss calculation
                loss_val, loss_breakdown = combined_loss(
                    logits, labels,
                    grade_outputs=grade_outputs,
                    domain_logits=domain_logits,
                    domain_labels=domain_labels,
                    lambda_consistency=lambda_consistency,
                    lambda_domain=lambda_domain,
                    lambda_ordinal=lambda_ordinal, # Pass ordinal lambda
                    use_label_smoothing=use_label_smoothing,
                    smoothing=smoothing
                )

        if autocast_enabled:
            scaler.scale(loss_val).backward()
            # Unscale before clipping
            scaler.unscale_(optimizer)
            # Gradient Clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_val.backward()
            # Gradient Clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss_val.item() * batch_size # Accumulate total loss for epoch avg
        num_samples += batch_size

        # Accumulate breakdown losses (use .get with default 0)
        total_main_loss += loss_breakdown.get('main_loss', 0) * batch_size
        total_consistency_loss += loss_breakdown.get('consistency_loss_bce', 0) * batch_size
        total_ordinal_loss += loss_breakdown.get('ordinal_loss', 0) * batch_size
        total_domain_loss += loss_breakdown.get('domain_loss', 0) * batch_size


        # Collect metrics only for non-mixup batches for accuracy/f1
        if not mixed_target:
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        if i % log_interval == 0 or i == num_batches - 1:
            current_lr = optimizer.param_groups[0]['lr'] # Get LR from first param group
            logging.info(f"Epoch [{epoch+1}/{wandb_run.config.epochs}] Step [{i+1}/{num_batches}] Batch Loss: {loss_val.item():.4f} LR: {current_lr:.6f} Alpha: {alpha:.3f}")
            wandb_run.log({
                "train/batch_loss": loss_val.item(),
                "train/learning_rate": current_lr,
                "train/step": i + epoch * num_batches, # Global step
                "train/domain_alpha": alpha
            })

    avg_loss = running_loss / num_samples if num_samples > 0 else 0
    avg_main_loss = total_main_loss / num_samples if num_samples > 0 else 0
    avg_consistency_loss = total_consistency_loss / num_samples if num_samples > 0 else 0
    avg_ordinal_loss = total_ordinal_loss / num_samples if num_samples > 0 else 0
    avg_domain_loss = total_domain_loss / num_samples if num_samples > 0 else 0

    metrics = {"train/epoch_loss": avg_loss, "epoch": epoch + 1}
    if avg_main_loss > 0: metrics["train/main_loss"] = avg_main_loss
    if avg_consistency_loss > 0: metrics["train/consistency_loss"] = avg_consistency_loss
    if avg_ordinal_loss > 0: metrics["train/ordinal_loss"] = avg_ordinal_loss
    if avg_domain_loss > 0: metrics["train/domain_loss"] = avg_domain_loss


    # Calculate epoch metrics based on non-mixup samples
    if len(all_preds) > 0:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted') # Use weighted for imbalance
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        metrics.update({
            "train/accuracy": acc,
            "train/f1_weighted": f1,
            "train/qwk": qwk
        })
        logging.info(f"Epoch [{epoch+1}] Train Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, QWK: {qwk:.4f}")
    else:
        logging.info(f"Epoch [{epoch+1}] Train Avg Loss: {avg_loss:.4f} (No accuracy metrics due to mixup/batch size)")

    wandb_run.log(metrics)
    return avg_loss, metrics.get("train/accuracy", 0), metrics.get("train/f1_weighted", 0)


def validate(model, dataloader, device, epoch, wandb_run, lambda_consistency=0.3, lambda_ordinal=0.1,
             use_label_smoothing=False, smoothing=0.1, phase="val"):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    num_samples = 0

    with torch.no_grad():
        for batch_data in dataloader:
            # Adjust based on dataloader structure
            if len(batch_data) == 3:
                images, labels, _ = batch_data # Ignore domain labels during validation
            else:
                images, labels = batch_data

            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            logits, grade_outputs, _ = model(images, alpha=0.0) # No domain reversal or prototype update

            # Calculate validation loss (without domain loss)
            loss_val, loss_breakdown = combined_loss(
                logits, labels,
                grade_outputs=grade_outputs,
                domain_logits=None, # No domain loss in validation
                domain_labels=None,
                lambda_consistency=lambda_consistency,
                lambda_domain=0, # Explicitly zero
                lambda_ordinal=lambda_ordinal,
                use_label_smoothing=use_label_smoothing,
                smoothing=smoothing
            )

            running_loss += loss_val.item() * batch_size
            num_samples += batch_size

            # For metrics calculation
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / num_samples if num_samples > 0 else 0
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # --- Calculate Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate AUC (handle cases with missing classes)
    mean_auc = 0.0
    auc_scores = []
    num_classes = all_probs.shape[1]
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 1: # Need at least 2 classes for AUC
        try:
            for i in range(num_classes):
                if i in unique_labels:
                    y_true_binary = (all_labels == i).astype(int)
                    # Check if there are positive samples for this class
                    if np.sum(y_true_binary) > 0 and np.sum(y_true_binary) < len(y_true_binary):
                        y_prob_binary = all_probs[:, i]
                        auc = roc_auc_score(y_true_binary, y_prob_binary)
                        auc_scores.append(auc)
            if auc_scores:
                mean_auc = np.mean(auc_scores)
        except Exception as e:
            logging.warning(f"Could not calculate {phase} AUC: {e}")
            mean_auc = 0.0

    # Calculate Sensitivity (Recall) and Specificity per class
    sensitivity = []
    specificity = []
    eps = 1e-6 # Avoid division by zero
    for i in range(num_classes):
         if i < len(cm): # Check if class index is within confusion matrix bounds
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fp - fn

            sensitivity.append(tp / (tp + fn + eps))
            specificity.append(tn / (tn + fp + eps))
         else:
            sensitivity.append(0.0)
            specificity.append(0.0)

    avg_sensitivity = np.mean(sensitivity) if sensitivity else 0.0
    avg_specificity = np.mean(specificity) if specificity else 0.0

    logging.info(f"{phase.upper()} - Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1-W: {f1_weighted:.4f}, QWK: {qwk:.4f}")
    logging.info(f"{phase.upper()} - Sens: {avg_sensitivity:.4f}, Spec: {avg_specificity:.4f}, AUC: {mean_auc:.4f}")

    # Log metrics to wandb, prefixing with phase (val or test)
    metrics_to_log = {
        f"{phase}/epoch_loss": avg_loss,
        f"{phase}/accuracy": acc,
        f"{phase}/f1_weighted": f1_weighted,
        f"{phase}/f1_macro": f1_macro,
        f"{phase}/qwk": qwk,
        f"{phase}/sensitivity_avg": avg_sensitivity,
        f"{phase}/specificity_avg": avg_specificity,
        f"{phase}/auc_mean": mean_auc,
        "epoch": epoch + 1
    }

    # Log class-wise metrics
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    for i in range(num_classes):
        metrics_to_log[f"{phase}/sensitivity_class{i}"] = sensitivity[i]
        metrics_to_log[f"{phase}/specificity_class{i}"] = specificity[i]
        # Get f1 from classification report safely
        f1_class = class_report.get(str(i), {}).get('f1-score', 0.0)
        metrics_to_log[f"{phase}/f1_class{i}"] = f1_class

    wandb_run.log(metrics_to_log)

    # Return dict of all metrics for early stopping use
    all_metrics = {
        "loss": avg_loss, "accuracy": acc, "f1_weighted": f1_weighted, "f1_macro": f1_macro,
        "sensitivity": avg_sensitivity, "specificity": avg_specificity, "qwk": qwk, "auc": mean_auc
    }
    return all_metrics


# --- Checkpoint & Logging Helpers ---

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint: {filepath}")

def log_final_metrics(metrics_dict, phase="Best Validation"):
    logging.info(f"--- {phase} Metrics ---")
    for key, value in metrics_dict.items():
        # Format key for better readability
        formatted_key = key.replace('_', ' ').replace('val/', '').capitalize()
        logging.info(f"{formatted_key}: {value:.4f}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Enhanced Fine-tuning for DR Classification")
    # Paths and Data
    parser.add_argument("--checkpoint", type=str, default="model/new/chckpt/moco/new/best_checkpoint.pth", help="Path to MoCo checkpoint")
    parser.add_argument("--output_dir", type=str, default="model/new/chckpt/enhanced_finetune_stable", help="Directory to save checkpoints and logs")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of DR classes")
    parser.add_argument("--dataset_names", nargs='+', default=["eyepacs", "aptos", "ddr", "idrid", "messdr"], help="List of dataset names to use")
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=10, help="Epochs to train with frozen backbone") # Increased default
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (reduced for stability)") # Reduced default
    parser.add_argument("--lr", type=float, default=5e-5, help="Base learning rate (reduced default)") # Reduced default
    parser.add_argument("--lr_backbone_unfrozen", type=float, default=5e-6, help="LR for backbone after unfreezing") # Explicit LR for unfrozen backbone
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    # Regularization & Augmentation
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp", help="Disable Automatic Mixed Precision")
    parser.add_argument("--use_mixup", action="store_true", default=True, help="Use Mixup augmentation")
    parser.add_argument("--no_mixup", action="store_false", dest="use_mixup", help="Disable Mixup")
    parser.add_argument("--mixup_alpha", type=float, default=0.4, help="Mixup alpha value")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--use_label_smoothing", action="store_true", default=False, help="Use Label Smoothing") # Default off
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    # Loss Weights & Domain Adaptation
    parser.add_argument("--lambda_consistency", type=float, default=0.2, help="Weight for grade consistency BCE loss") # Reduced default
    parser.add_argument("--lambda_ordinal", type=float, default=0.1, help="Weight for ordinal threshold loss") # Added lambda
    parser.add_argument("--lambda_domain", type=float, default=0.1, help="Weight for domain adaptation loss") # Reduced default
    parser.add_argument("--domain_adaptation", action="store_true", default=True, help="Use domain adaptation GRL")
    parser.add_argument("--no_domain_adaptation", action="store_false", dest="domain_adaptation", help="Disable domain adaptation")
    parser.add_argument("--domain_alpha_max", type=float, default=0.5, help="Maximum alpha for GRL") # Reduced default
    parser.add_argument("--domain_alpha_ramp_epochs", type=int, default=25, help="Epochs to ramp up GRL alpha") # Increased ramp
    # Early Stopping & SWA
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--early_stopping_metric", type=str, default="val/qwk", # Use wandb metric name
                        choices=["val/loss", "val/accuracy", "val/f1_weighted", "val/sensitivity_avg", "val/specificity_avg", "val/qwk", "val/auc_mean"],
                        help="Metric for early stopping (use wandb key)")
    parser.add_argument("--use_swa", action="store_true", default=True, help="Use Stochastic Weight Averaging") # Default on
    parser.add_argument("--no_swa", action="store_false", dest="use_swa", help="Disable SWA")
    parser.add_argument("--swa_start_epoch_frac", type=float, default=0.75, help="Fraction of epochs to start SWA")
    parser.add_argument("--swa_lr", type=float, default=1e-6, help="Learning rate for SWALR scheduler") # Low SWA LR
    # Logging & Misc
    parser.add_argument("--wandb_project", type=str, default="DR-Finetuning-Stable", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (generated if None)")

    args = parser.parse_args()

    # --- Setup Output Dir & Logging ---
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "finetune.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting Enhanced DR Finetuning Script")
    logging.info(f"Arguments: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU. This will be very slow.")
        args.use_amp = False # AMP requires CUDA

    # --- Initialize WandB ---
    if args.wandb_run_name is None:
         run_name = f"enhanced_ft_{args.img_size}_lr{args.lr}_bs{args.batch_size}"
         if args.use_swa: run_name += "_swa"
         if args.domain_adaptation: run_name += "_da"
         if args.use_mixup: run_name += "_mixup"
    else:
        run_name = args.wandb_run_name

    wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
            dir=args.output_dir # Save wandb logs locally as well
        )
    logging.info(f"WandB Run URL: {wandb_run.get_url()}")

    # --- Initialize Model ---
    model = EnhancedDRClassifier(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        freeze_backbone=True # Start frozen
    ).to(device)
    wandb.watch(model, log='gradients', log_freq=500) # Log gradients periodically

    # --- Define Transformations & Datasets ---
    # Use simpler transforms initially for stability
    train_transform = data_aug.MoCoSingleAug(img_size=args.img_size) # Use basic train aug
    val_transform = data_aug.MoCoSingleAug(img_size=args.img_size) # Use basic val aug

    logging.info(f"Using datasets: {args.dataset_names}")
    train_loader = data_set.UniformTrainDataloader(
        dataset_names=args.dataset_names,
        transformation=train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=True # Use weighted sampler
    ).get_loader()

    val_loader = data_set.UniformValidDataloader(
        dataset_names=args.dataset_names, # Validate on all datasets combined
        transformation=val_transform,
        batch_size=args.batch_size, # Can often use larger BS for validation
        num_workers=args.num_workers,
    ).get_loader()

    # --- Optimizer & Scheduler (Initial Frozen State) ---
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': args.lr},
        {'params': model.grade_head.parameters(), 'lr': args.lr},
        {'params': model.domain_classifier.parameters(), 'lr': args.lr},
        {'params': model.attention.parameters(), 'lr': args.lr }, # Same LR for attention initially
        # Backbone params are not included yet as they are frozen
    ], weight_decay=args.weight_decay, eps=1e-8) # AdamW epsilon for stability

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    # Scheduler for frozen phase (only affects non-backbone params)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[pg['lr'] for pg in optimizer.param_groups], # Get max_lr for each group
        total_steps=total_steps, # Schedule over all epochs initially
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    logging.info("Optimizer and Scheduler initialized for frozen backbone phase.")

    # --- AMP Scaler ---
    scaler = GradScaler(enabled=args.use_amp)
    logging.info(f"Automatic Mixed Precision {'enabled' if args.use_amp else 'disabled'}.")

    # --- Early Stopping & Best Model Tracking ---
    early_stopping_metric = args.early_stopping_metric
    lower_is_better = "loss" in early_stopping_metric # Check if loss is the metric
    best_metric_val = float('inf') if lower_is_better else -float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    best_val_metrics_all = {} # Store all metrics from the best epoch
    logging.info(f"Early stopping enabled: Metric='{early_stopping_metric}', Patience={args.early_stopping_patience}, LowerIsBetter={lower_is_better}")


    # --- SWA Setup ---
    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        swa_model = AveragedModel(model)
        # SWA scheduler steps *after* the main scheduler
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
        swa_start_epoch = int(args.epochs * args.swa_start_epoch_frac)
        logging.info(f"SWA enabled: Start epoch ~{swa_start_epoch}, SWA LR={args.swa_lr}")
    else:
        swa_start_epoch = args.epochs + 1 # Ensure SWA doesn't start if disabled
        logging.info("SWA disabled.")

    # --- Training Loop ---
    logging.info("Starting training loop...")
    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")

        # --- Unfreeze Logic ---
        if epoch == args.freeze_epochs:
            logging.info(f"Epoch {epoch+1}: Unfreezing backbone.")
            model.unfreeze_backbone()

            # Recreate optimizer with backbone parameters included
            logging.info(f"Recreating optimizer. Head LR: {args.lr}, Backbone LR: {args.lr_backbone_unfrozen}")
            optimizer = optim.AdamW([
                {'params': model.classifier.parameters(), 'lr': args.lr / 2}, # Optionally reduce head LR slightly
                {'params': model.grade_head.parameters(), 'lr': args.lr / 2},
                {'params': model.domain_classifier.parameters(), 'lr': args.lr / 2},
                {'params': model.attention.parameters(), 'lr': args.lr / 2},
                {'params': model.backbone.parameters(), 'lr': args.lr_backbone_unfrozen} # Add backbone params with low LR
            ], weight_decay=args.weight_decay, eps=1e-8)

            # Recreate scheduler for the remaining epochs
            remaining_epochs = args.epochs - epoch
            total_steps_remaining = steps_per_epoch * remaining_epochs
            logging.info(f"Recreating scheduler for {remaining_epochs} remaining epochs.")

            scheduler = OneCycleLR(
                optimizer,
                max_lr=[pg['lr'] for pg in optimizer.param_groups], # Get new max LRs
                total_steps=total_steps_remaining,
                pct_start=0.05, # Shorter warmup after unfreezing
                div_factor=10, # Less aggressive initial division
                final_div_factor=100,
                anneal_strategy='cos'
            )

            # Re-initialize SWA scheduler with the new optimizer if SWA is active
            if args.use_swa:
                swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr) # Keep SWA LR constant

            # Reset AMP scaler state if needed (optional, usually okay)
            # scaler = GradScaler(enabled=args.use_amp)

        # --- Train ---
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device, epoch, wandb_run, scaler,
            lambda_consistency=args.lambda_consistency,
            lambda_domain=args.lambda_domain,
            lambda_ordinal=args.lambda_ordinal, # Pass ordinal lambda
            domain_adaptation=args.domain_adaptation,
            domain_alpha_max=args.domain_alpha_max,
            domain_alpha_ramp_epochs=args.domain_alpha_ramp_epochs,
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha,
            grad_clip_norm=args.grad_clip_norm,
            use_label_smoothing=args.use_label_smoothing,
            smoothing=args.label_smoothing
        )

        # --- Validate ---
        current_val_metrics = validate(
            model, val_loader, device, epoch, wandb_run,
            lambda_consistency=args.lambda_consistency,
            lambda_ordinal=args.lambda_ordinal,
            use_label_smoothing=args.use_label_smoothing,
            smoothing=args.label_smoothing,
            phase="val" # Specify validation phase
        )

        # --- Scheduler & SWA Step ---
        # Step main scheduler first
        scheduler.step()
        # Then, step SWA scheduler and update SWA model if in SWA phase
        if args.use_swa and epoch >= swa_start_epoch:
            # Check if swa_model exists (safety)
            if swa_model is not None and swa_scheduler is not None:
                 swa_model.update()
                 swa_scheduler.step()
                 logging.debug(f"Epoch {epoch+1}: Stepped SWA scheduler.")
            else:
                 logging.warning(f"Epoch {epoch+1}: Tried to step SWA, but model/scheduler is None.")


        # --- Checkpoint Saving & Early Stopping ---
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(), # Standard model state
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'config': vars(args),
            'val_metrics': current_val_metrics # Store all validation metrics for this epoch
        }

        # Save latest checkpoint
        save_checkpoint(checkpoint_state, args.output_dir, "latest_checkpoint.pth")
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
             save_checkpoint(checkpoint_state, args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")


        # Early Stopping Check
        current_metric_val = current_val_metrics.get(early_stopping_metric.replace("val/", ""), None) # Get metric value matching key format in dict
        if current_metric_val is None:
            logging.error(f"Early stopping metric '{early_stopping_metric}' not found in validation results dict. Available keys: {current_val_metrics.keys()}. Stopping check.")
            # Optionally break here or default to a different behavior
        else:
             improved = (current_metric_val < best_metric_val) if lower_is_better else (current_metric_val > best_metric_val)
             if improved:
                 logging.info(f"Epoch {epoch+1}: {early_stopping_metric} improved from {best_metric_val:.4f} to {current_metric_val:.4f}. Saving best model.")
                 best_metric_val = current_metric_val
                 best_epoch = epoch + 1
                 best_val_metrics_all = current_val_metrics # Save all metrics from this best epoch
                 epochs_no_improve = 0
                 # Save the best model based on the chosen metric
                 save_checkpoint(checkpoint_state, args.output_dir, f"best_{early_stopping_metric.replace('/', '_')}_checkpoint.pth")
             else:
                 epochs_no_improve += 1
                 logging.info(f"Epoch {epoch+1}: {early_stopping_metric} did not improve for {epochs_no_improve} epochs. Best: {best_metric_val:.4f} (Epoch {best_epoch})")

             if epochs_no_improve >= args.early_stopping_patience:
                 logging.warning(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement on {early_stopping_metric}.")
                 break # Exit the training loop

    # --- End of Training Loop ---
    logging.info("Training loop finished.")
    logging.info(f"Best checkpoint saved based on '{early_stopping_metric}' from Epoch {best_epoch}")
    log_final_metrics(best_val_metrics_all, phase="Best Validation") # Log the best metrics found

    # Log best metrics to WandB summary
    for key, value in best_val_metrics_all.items():
        wandb_run.summary[f"best_val/{key}"] = value
    wandb_run.summary["best_epoch"] = best_epoch


    # --- SWA Finalization ---
    if args.use_swa and best_epoch >= swa_start_epoch and swa_model is not None:
        logging.info("Performing SWA Batch Norm update...")
        try:
            # Ensure model is on the correct device for BN update
            swa_model = swa_model.to(device)
            swa_model.eval() # Set to eval mode for BN update
            # Pass the underlying module if DataParallel was used (unlikely here but good practice)
            bn_update_model = swa_model.module if isinstance(swa_model, nn.DataParallel) else swa_model
            update_bn(train_loader, bn_update_model, device=device)
            logging.info("SWA Batch Norm update complete.")

            # Save the final SWA model state
            swa_state = {
                'epoch': best_epoch, # Or args.epochs if no early stopping
                'model_state_dict': bn_update_model.state_dict(), # Use the state_dict after BN update
                'config': vars(args),
                'val_metrics_at_best_epoch': best_val_metrics_all # Include metrics from standard model's best epoch
            }
            save_checkpoint(swa_state, args.output_dir, "swa_final_checkpoint.pth")
            logging.info("Saved final SWA model checkpoint.")

            # Optional: Evaluate the SWA model on the validation set
            logging.info("Evaluating final SWA model...")
            # Load SWA state into a fresh instance or the current swa_model
            # model.load_state_dict(swa_state['model_state_dict']) # Load into base model structure
            swa_val_metrics = validate(
                bn_update_model, val_loader, device, args.epochs, wandb_run, # Use updated swa_model
                lambda_consistency=args.lambda_consistency,
                lambda_ordinal=args.lambda_ordinal,
                use_label_smoothing=args.use_label_smoothing,
                smoothing=args.label_smoothing,
                phase="swa_val" # Specify SWA validation phase
                )
            log_final_metrics(swa_val_metrics, phase="SWA Final Validation")
            for key, value in swa_val_metrics.items():
                 wandb_run.summary[f"swa_val/{key}"] = value

        except Exception as e:
            logging.error(f"Error during SWA finalization/evaluation: {e}", exc_info=True)
            logging.warning("Skipping SWA model saving/evaluation due to error.")
    elif args.use_swa:
         logging.warning("SWA was enabled, but training stopped before SWA start epoch or SWA model is None. Skipping SWA finalization.")


    wandb_run.finish()
    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()