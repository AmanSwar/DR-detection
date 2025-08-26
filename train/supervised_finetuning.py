import os
import logging
import argparse
import numpy as np
import random
import time  # For potential timestamping

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
)
from torchvision import transforms  # Added for validation transform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    OneCycleLR,
)  # , CosineAnnealingLR, ReduceLROnPlateau
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
    # Might make things non-deterministic, but can improve performance
    # torch.backends.cudnn.benchmark = True
    # Use deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable benchmark for full determinism


# Based on CBAM
class LesionAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(LesionAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False
            ),
        )

        # Spatial attention
        assert kernel_size % 2 == 1, "Kernel size must be odd for spatial attention"
        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg -> shared mlp -> avg out (bs , c )
        avg_out = self.shared_mlp(self.avg_pool(x))
        # max -> shared mlp -> max out (bs ,c )
        max_out = self.shared_mlp(self.max_pool(x))
        # channel attn => sum(avg + max) -> sigmoid -> out
        channel_att = self.sigmoid(avg_out + max_out)
        # attn into main channel
        x_channel = x * channel_att

        # spatial attn
        # avg
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        # max
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        # concatinate max and avg
        spatial_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        # conv layer(spatial input) -> sigmoid -> outs
        spatial_att = self.sigmoid(self.conv_spatial(spatial_input))

        # multiply it by main x_channel
        return x_channel * spatial_att


# Gradient Reversal


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        """do nothing"""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Multiply gradient by -alpha, pass None for alpha's gradient
        return grad_output.neg() * ctx.alpha, None


class GradeConsistencyHead(nn.Module):
    def __init__(self, feature_dim, num_grades=5, dropout_rate=0.4):
        super(GradeConsistencyHead, self).__init__()
        # grade predictor
        self.grade_predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_grades),
        )
        # Ordinal regression part (predicts thresholds/cumulative logits)
        self.ordinal_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_grades - 1),  # Predict K-1 thresholds for K classes
        )

    def forward(self, x):
        logits = self.grade_predictor(x)

        ordinal_thresholds = self.ordinal_encoder(x)
        return logits, ordinal_thresholds


class EnhancedDRClassifier(nn.Module):
    def __init__(
        self, checkpoint_path, num_classes=5, freeze_backbone=True, dropout_rate=0.5
    ):
        super(EnhancedDRClassifier, self).__init__()

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            moco_state_dict = checkpoint["model_state_dict"]
            config = checkpoint["config"]
            base_model_name = config.get("base_model", "resnet50")
            logging.info(f"Loading backbone: {base_model_name} from MoCo checkpoint.")
            self.backbone = timm.create_model(
                base_model_name, pretrained=False, num_classes=0
            )  # num_classes=0 removes classifier head
        except FileNotFoundError:
            logging.error(f"MoCo checkpoint not found at {checkpoint_path}. Exiting.")
            raise
        except KeyError as e:
            logging.error(
                f"Key error loading checkpoint: {e}. Check checkpoint structure."
            )
            raise

        # Extract state dict for the query encoder
        backbone_state_dict = {}
        for k, v in moco_state_dict.items():
            if k.startswith("query_encoder."):
                # Remove the 'query_encoder.' prefix
                new_k = k.replace("query_encoder.", "", 1)
                backbone_state_dict[new_k] = v
            # Handle potential older checkpoints without 'query_encoder.' prefix
            elif not (
                k.startswith("key_encoder.")
                or k.startswith("queue")
                or k.startswith("queue_ptr")
            ):
                # Assume it belongs to the backbone if not key encoder or queue
                backbone_state_dict[k] = v

        # Load state dict (handle potential mismatches)
        msg = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        logging.info(f"Backbone loading message: {msg}")
        if msg.missing_keys:
            logging.warning(f"Missing keys when loading backbone: {msg.missing_keys}")
        if msg.unexpected_keys:
            logging.warning(
                f"Unexpected keys when loading backbone: {msg.unexpected_keys}"
            )

        # --- Freeze Backbone ---
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            logging.info("Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            logging.info("Backbone parameters are trainable.")

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
            nn.Linear(512, num_classes),
        )

        # --- Auxiliary Heads ---
        self.grade_head = GradeConsistencyHead(
            self.feature_dim, num_grades=num_classes, dropout_rate=dropout_rate - 0.1
        )  # Slightly less dropout maybe
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),  # ReLU often used in DANN heads
            nn.Linear(256, 5),  # Assuming 5 domains based on dataset_names
        )

        # --- Initialize Weights for New Layers ---
        self._initialize_weights()

    def _initialize_weights(self):
        logging.info(
            "Initializing weights for classifier, grade_head, and domain_classifier."
        )
        for module in [
            self.classifier,
            self.grade_head,
            self.domain_classifier,
            self.attention,
        ]:  # Initialize attention too
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # Kaiming init for linear layers often works well with ReLU/GELU
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif (
                    isinstance(m, nn.Conv2d) and m.bias is not None
                ):  # For attention conv layers if they have bias
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None
    ):
        # --- Feature Extraction ---
        if self.freeze_backbone:
            with torch.no_grad():  # Ensure no gradients calculated for frozen backbone
                features = self.backbone.forward_features(x)
        else:
            features = self.backbone.forward_features(
                x
            )  # Pass features directly from backbone

        # --- Attention ---
        attended_features = self.attention(features)
        h = torch.mean(attended_features, dim=(2, 3))  # Shape: (batch_size, 768)

        # --- Main Classifier Output ---
        logits = self.classifier(h)

        # --- Auxiliary Outputs ---
        grade_outputs = self.grade_head(h)  # Returns (logits, ordinal_thresholds)

        # --- Domain Classifier Output (with Gradient Reversal) ---
        domain_logits = None
        if alpha > 0:
            reversed_features = GradientReversal.apply(h, alpha)
            domain_logits = self.domain_classifier(reversed_features)

        if get_attention:
            return logits, grade_outputs, domain_logits, h, attended_features
        else:
            return logits, grade_outputs, domain_logits

    def unfreeze_backbone(self):
        if self.freeze_backbone:
            logging.info("Unfreezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze_backbone = False  # Update state


# --- Loss Function ---


def OrdinalDomainLoss(
    outputs,
    labels,
    grade_outputs=None,
    domain_logits=None,
    domain_labels=None,
    lambda_consistency=0.1,
    lambda_domain=0.05,
    ordinal_weight=0.3,
    num_classes=5,
):
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
    # global loss => loss
    loss = main_loss

    # --- Grade Consistency Loss ---
    if grade_outputs is not None and lambda_consistency > 0:
        grade_logits, ordinal_thresholds = grade_outputs
        batch_size = labels.size(0)

        targets_cumulative = torch.zeros_like(grade_logits)
        for i in range(batch_size):
            if 0 <= labels[i] < num_classes:  # Ensure label is valid
                targets_cumulative[i, : labels[i] + 1] = (
                    1.0  # Target is 1 for classes <= true label
                )
        # Use BCEWithLogitsLoss for numerical stability
        consistency_loss_bce = F.binary_cross_entropy_with_logits(
            grade_logits, targets_cumulative, reduction="mean"
        )

        consistency_loss = consistency_loss_bce  # Start with BCE part

        # 3. Ordinal Threshold Loss (Ordinal approach 2 - CORAL-like or similar)
        # Predict P(y > k) using thresholds. Target is 1 if true_label > k, else 0.
        if ordinal_thresholds is not None and ordinal_weight > 0:
            ordinal_targets = torch.zeros_like(
                ordinal_thresholds
            )  # Shape (batch_size, num_classes - 1)
            for i in range(batch_size):
                for k in range(num_classes - 1):
                    if labels[i] > k:
                        ordinal_targets[i, k] = 1.0
            # Use BCEWithLogitsLoss for the thresholds as well
            ordinal_loss_bce = F.binary_cross_entropy_with_logits(
                ordinal_thresholds, ordinal_targets, reduction="mean"
            )

            # Combine the two ordinal approaches
            consistency_loss = (
                1.0 - ordinal_weight
            ) * consistency_loss_bce + ordinal_weight * ordinal_loss_bce

        loss += lambda_consistency * consistency_loss
        consistency_loss_val = consistency_loss.item()

    # --- Domain Adversarial Loss ---
    if domain_logits is not None and domain_labels is not None and lambda_domain > 0:
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(domain_logits, domain_labels)
        loss += lambda_domain * domain_loss
        domain_loss_val = domain_loss.item()

    # Return total loss and optionally components for logging
    return loss  # , main_loss.item(), consistency_loss_val, domain_loss_val


# --- Training & Validation Loops ---


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    num_epochs,
    wandb_run,
    scaler=None,
    lambda_consistency=0.1,
    lambda_domain=0.05,
    ordinal_weight=0.3,
    domain_adaptation=True,
    max_grad_norm=1.0,
    num_classes=5,
):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    total_steps = len(dataloader)

    # Calculate DANN alpha (gradient reversal multiplier) based on epoch progress
    # Common schedule: smoothly increase alpha from 0 to 1
    p = float(epoch) / num_epochs
    alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1 if domain_adaptation else 0.0
    # Alternative linear schedule: alpha = min(1.0, 2.0 * p) if domain_adaptation else 0.0

    start_time = time.time()

    for i, batch_data in enumerate(dataloader):
        # Adjust based on your specific dataloader structure
        if len(batch_data) == 3:
            images, labels, domain_labels = batch_data
        else:  # Assuming validation might not have domain labels
            images, labels = batch_data
            domain_labels = (
                None  # Or handle appropriately if domain loss is still calculated
            )

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if domain_labels is not None:
            domain_labels = domain_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More memory efficient

        if scaler is not None:  # Using AMP
            with autocast():
                # Pass alpha for gradient reversal, enable prototype update
                logits, grade_outputs, domain_logits = model(
                    images, alpha=alpha, update_prototypes=True, labels=labels
                )
                loss = OrdinalDomainLoss(
                    logits,
                    labels,
                    grade_outputs=grade_outputs,
                    domain_logits=domain_logits,
                    domain_labels=domain_labels,
                    lambda_consistency=lambda_consistency,
                    lambda_domain=lambda_domain,
                    ordinal_weight=ordinal_weight,
                    num_classes=num_classes,
                )

            # Scale loss, backward pass, unscale gradients, clip gradients, optimizer step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:  # Not using AMP
            logits, grade_outputs, domain_logits = model(
                images, alpha=alpha, update_prototypes=True, labels=labels
            )
            loss = OrdinalDomainLoss(
                logits,
                labels,
                grade_outputs=grade_outputs,
                domain_logits=domain_logits,
                domain_labels=domain_labels,
                lambda_consistency=lambda_consistency,
                lambda_domain=lambda_domain,
                ordinal_weight=ordinal_weight,
                num_classes=num_classes,
            )
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        running_loss += loss.item()

        # --- Track predictions and labels for epoch metrics ---
        _, predicted = torch.max(logits.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        # --- Logging ---
        if (
            i + 1
        ) % 20 == 0 or i == total_steps - 1:  # Log every 20 steps and at the end
            current_lr = optimizer.param_groups[0][
                "lr"
            ]  # Get LR from first param group
            batch_time = time.time() - start_time
            eta_seconds = batch_time / (i + 1) * (total_steps - (i + 1))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{total_steps}] Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.6f} | GradNorm: {grad_norm:.4f} | ETA: {eta_str}"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "train/grad_norm_clipped": grad_norm,  # Name clarifies it's post-clipping
                        "train/domain_alpha": alpha,
                        "step": i + epoch * total_steps,  # Global step
                    }
                )
            start_time = time.time()  # Reset timer for next batch group

    # --- Epoch Metrics ---
    avg_loss = running_loss / total_steps
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(
        all_labels, all_preds, average="weighted"
    )  # Use weighted F1 for imbalance
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    logging.info(
        f"Epoch [{epoch+1}/{num_epochs}] Train Summary - Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, QWK: {qwk:.4f}"
    )

    if wandb_run:
        wandb_run.log(
            {
                "train/epoch_loss": avg_loss,
                "train/accuracy": acc,
                "train/f1_weighted": f1,
                "train/qwk": qwk,
                "epoch": epoch + 1,
            }
        )

    return avg_loss, acc, f1, qwk  # Return QWK as well


def validate(
    model,
    dataloader,
    device,
    epoch,
    num_epochs,
    wandb_run,
    lambda_consistency=0.1,
    ordinal_weight=0.3,
    num_classes=5,
):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []  # Store probabilities for AUC calculation

    with torch.no_grad():  # Disable gradient calculations
        for batch_data in dataloader:
            # Adjust based on your specific dataloader structure
            if len(batch_data) == 3:  # Might still have domain label placeholder
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
                logits,
                labels,
                grade_outputs=grade_outputs,
                domain_logits=None,
                domain_labels=None,  # No domain loss in validation
                lambda_consistency=lambda_consistency,
                lambda_domain=0.0,  # Ensure domain loss weight is 0
                ordinal_weight=ordinal_weight,
                num_classes=num_classes,
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
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

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
            auc_macro_ovr = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro"
            )
        elif len(present_classes) > 1 and all_probs.shape[1] == num_classes:
            logging.warning(
                f"Only {len(present_classes)}/{num_classes} classes present in validation batch. AUC might be unreliable."
            )
            # Calculate OvR AUC only for present classes if possible, or report 0
            try:
                auc_macro_ovr = roc_auc_score(
                    all_labels,
                    all_probs,
                    multi_class="ovr",
                    average="macro",
                    labels=present_classes,
                )
            except ValueError:  # Handle cases where AUC is undefined for a class
                auc_macro_ovr = 0.0
        else:
            logging.warning(
                "Not enough classes present or probability shape mismatch for AUC calculation."
            )
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
            "epoch": epoch + 1,
        }
        # Log per-class metrics
        report_dict = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )
        for i in range(num_classes):
            log_dict[f"val/sensitivity_class{i}"] = sensitivity[i]
            log_dict[f"val/specificity_class{i}"] = specificity[i]
            cls_report = report_dict.get(str(i), {})
            log_dict[f"val/f1_class{i}"] = cls_report.get("f1-score", 0.0)
            log_dict[f"val/precision_class{i}"] = cls_report.get("precision", 0.0)
            log_dict[f"val/recall_class{i}"] = cls_report.get(
                "recall", 0.0
            )  # Recall is sensitivity

        wandb_run.log(log_dict)

        # Log confusion matrix as image (optional)
        # try:
        #     if cm is not None:
        #         wandb.log({"val/confusion_matrix": wandb.Image(cm_figure)}) # Need function to plot cm
        # except Exception as e:
        #     logging.warning(f"Could not log confusion matrix to W&B: {e}")

    return (
        avg_loss,
        acc,
        f1_weighted,
        avg_sensitivity,
        avg_specificity,
        qwk,
        auc_macro_ovr,
    )


# --- Utility Functions ---


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth", is_best=False):
    """Saves checkpoint."""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    status = "Best" if is_best else "Regular"
    logging.info(f"Saved {status} checkpoint: {filepath}")
    # Optionally, save the best checkpoint with a fixed name like 'best_model.pth'
    if is_best:
        best_filepath = os.path.join(
            checkpoint_dir, f"best_{filename.split('_checkpoint')[0]}_model.pth"
        )  # e.g., best_qwk_model.pth
        torch.save(state, best_filepath)
        logging.info(f"Saved as overall best model: {best_filepath}")


# --- Main Function ---


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MoCo model for Diabetic Retinopathy Classification"
    )
    # --- Paths and Data ---
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/new/chckpt/moco/new/best_checkpoint.pth",
        help="Path to MoCo pre-trained backbone checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chckpt/finetune_nofreeze/fine_3",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=["eyepacs", "aptos", "ddr", "idrid", "messdr"],
        help="List of dataset names to use",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size for training and validation",
    )
    # --- Training Hyperparameters ---
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=5,
        help="Number of epochs to train only the heads before unfreezing backbone (0 for no freezing)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Peak Learning rate for OneCycleLR (for heads/new layers)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping",
    )
    # --- Model and Loss ---
    parser.add_argument(
        "--num_classes", type=int, default=5, help="Number of DR classes"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate for classifier heads",
    )
    parser.add_argument(
        "--lambda_consistency",
        type=float,
        default=0.1,
        help="Weight for grade consistency loss",
    )
    parser.add_argument(
        "--ordinal_weight",
        type=float,
        default=0.3,
        help="Weight for ordinal threshold loss within consistency loss",
    )
    # --- Domain Adaptation ---
    parser.add_argument(
        "--domain_adaptation",
        action="store_true",
        default=True,
        help="Use domain adaptation (DANN)",
    )
    parser.add_argument(
        "--lambda_domain",
        type=float,
        default=0.05,
        help="Weight for domain adversarial loss (if domain_adaptation is True)",
    )
    # --- Augmentation and Technical ---
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision (AMP)",
    )  # Use --use-amp or --no-use-amp
    # parser.add_argument("--use_mixup", action="store_true", default=False, help="Use Mixup/Cutmix augmentation (Not implemented in this version)")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    # --- W&B Logging ---
    parser.add_argument(
        "--wandb_project", type=str, default="DR_FineTuning", help="WandB project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name (generated if None)",
    )

    args = parser.parse_args()

    # --- Setup Output Directory and Logging ---
    # Add timestamp or unique identifier to output directory if needed
    # output_dir = os.path.join(args.output_dir, f"run_{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "finetune.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("Starting Enhanced DR Fine-tuning Script")
    logging.info(f"Arguments: {vars(args)}")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    # --- WandB Setup ---
    if args.wandb_run_name is None:
        # Generate a descriptive run name
        run_name = (
            f"ft_{args.img_size}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}"
        )
        if args.freeze_epochs > 0:
            run_name += f"_freeze{args.freeze_epochs}"
        if args.domain_adaptation:
            run_name += "_da"
    else:
        run_name = args.wandb_run_name

    try:
        wandb_run = wandb.init(
            project=args.wandb_project, config=vars(args), name=run_name
        )
        logging.info(f"WandB run initialized: {run_name}")
    except Exception as e:
        logging.warning(
            f"Could not initialize WandB: {e}. Proceeding without W&B logging."
        )
        wandb_run = None

    # --- Model Initialization ---
    model = EnhancedDRClassifier(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_epochs > 0,  # Start frozen if freeze_epochs > 0
        dropout_rate=args.dropout_rate,
    ).to(device)

    # --- Data Transformations ---
    # Use MoCo augmentation for training, simpler for validation
    train_transform = data_aug.MoCoSingleAug(
        img_size=args.img_size
    )  # Use your existing MoCo aug
    val_transform = transforms.Compose(
        [  # Standard validation transforms
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            # Use ImageNet stats or stats derived from your specific DR datasets if available
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    logging.info(f"Using Training Transform: {train_transform}")
    logging.info(f"Using Validation Transform: {val_transform}")

    # --- Data Loaders ---
    logging.info(f"Loading datasets: {args.dataset_names}")
    # Uses uniform sampling across specified datasets for training and validation
    train_loader = data_set.UniformTrainDataloader(
        dataset_names=args.dataset_names,
        transformation=train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=True,  # Ensures uniform sampling across datasets
    ).get_loader()

    val_loader = data_set.UniformValidDataloader(
        dataset_names=args.dataset_names,
        transformation=val_transform,
        batch_size=args.batch_size,  # Can often use larger batch size for validation
        num_workers=args.num_workers,
        sampler=True,  # Keep True if uniform validation sampling is desired, False for standard sequential
    ).get_loader()
    logging.info(
        f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}"
    )

    # --- Optimizer ---
    # Setup potentially different LRs for backbone vs heads
    if args.freeze_epochs > 0:
        logging.info(
            f"Optimizer setup: Training HEADS ONLY for first {args.freeze_epochs} epochs."
        )
        # Optimize only the parameters that require gradients (heads and attention)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,  # Use the main LR for heads
            weight_decay=args.weight_decay,
        )
    else:
        logging.info("Optimizer setup: Training FULL MODEL with differential LR.")
        # Separate parameters for backbone and new layers
        backbone_params = model.backbone.parameters()
        head_params = [
            p
            for name, p in model.named_parameters()
            if not name.startswith("backbone.") and p.requires_grad
        ]

        optimizer = optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": args.lr / 10.0,
                },  # Lower LR for pre-trained backbone
                {
                    "params": head_params,
                    "lr": args.lr,
                },  # Higher LR for randomly initialized layers
            ],
            weight_decay=args.weight_decay,
        )

    # --- Learning Rate Scheduler ---
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    if args.freeze_epochs > 0:
        # Scheduler for the head-only phase
        logging.info(
            f"Scheduler setup: OneCycleLR for head training ({args.freeze_epochs} epochs)"
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,  # Single max_lr as only heads are optimized
            total_steps=steps_per_epoch * args.freeze_epochs,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1000,
            anneal_strategy="cos",
        )
    else:
        # Scheduler for the full fine-tuning phase (adjust max_lr list for differential LR)
        logging.info("Scheduler setup: OneCycleLR for full model training")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[
                args.lr / 10.0,
                args.lr,
            ],  # Match max_lr to param groups [backbone, heads]
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1000,
            anneal_strategy="cos",
        )

    # --- Automatic Mixed Precision (AMP) Scaler ---
    scaler = GradScaler() if args.use_amp and device.type == "cuda" else None
    if scaler:
        logging.info("Using Automatic Mixed Precision (AMP).")

    # --- Training Loop ---
    best_val_metrics = {
        "loss": float("inf"),
        "accuracy": 0,
        "f1": 0,
        "sensitivity": 0,
        "specificity": 0,
        "qwk": -1.0,
        "auc": 0,  # Initialize QWK appropriately
    }
    patience_counter = 0
    best_metric_combined = 0.0  # Based on Acc, Sens, Spec
    PATIENCE_LIMIT = 20  # Example early stopping patience

    logging.info("--- Starting Training Loop ---")
    start_training_time = time.time()

    for epoch in range(args.epochs):

        # --- Handle Unfreezing ---
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            logging.info(f"--- Unfreezing backbone at epoch {epoch+1} ---")
            model.unfreeze_backbone()

            # Re-initialize optimizer for the whole model with differential LR
            backbone_params = model.backbone.parameters()
            head_params = [
                p
                for name, p in model.named_parameters()
                if not name.startswith("backbone.") and p.requires_grad
            ]
            optimizer = optim.AdamW(
                [
                    {
                        "params": backbone_params,
                        "lr": args.lr / 10.0,
                    },  # Start backbone LR low again
                    {"params": head_params, "lr": args.lr},
                ],
                weight_decay=args.weight_decay,
            )
            logging.info("Re-initialized optimizer for full model fine-tuning.")

            # Re-initialize scheduler for the remaining epochs
            remaining_steps = steps_per_epoch * (args.epochs - args.freeze_epochs)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[args.lr / 10.0, args.lr],  # Differential LR
                total_steps=remaining_steps,
                pct_start=0.1,
                div_factor=10,
                final_div_factor=1000,
                anneal_strategy="cos",
            )
            logging.info("Re-initialized scheduler for remaining epochs.")

        # --- Train for one epoch ---
        epoch_start_time = time.time()
        train_loss, train_acc, train_f1, train_qwk = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
            wandb_run,
            scaler=scaler,
            lambda_consistency=args.lambda_consistency,
            lambda_domain=args.lambda_domain,
            ordinal_weight=args.ordinal_weight,
            domain_adaptation=args.domain_adaptation,
            max_grad_norm=args.max_grad_norm,
            num_classes=args.num_classes,
        )

        # --- Validate ---
        (
            val_loss,
            val_acc,
            val_f1,
            val_sensitivity,
            val_specificity,
            val_qwk,
            val_auc,
        ) = validate(
            model,
            val_loader,
            device,
            epoch,
            args.epochs,
            wandb_run,
            lambda_consistency=args.lambda_consistency,
            ordinal_weight=args.ordinal_weight,
            num_classes=args.num_classes,
        )
        epoch_duration = time.time() - epoch_start_time
        logging.info(
            f"Epoch [{epoch+1}/{args.epochs}] Duration: {epoch_duration:.2f} seconds"
        )

        if not isinstance(scheduler, OneCycleLR):  # OneCycleLR usually steps per batch
            scheduler.step()  # Step other schedulers per epoch

        # --- Checkpoint Saving Logic ---
        # Define the combined metric (adjust weights as needed)
        combined_metric = 0.3 * val_acc + 0.4 * val_sensitivity + 0.3 * val_specificity

        checkpoint_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
            ),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_qwk": val_qwk,
            "val_auc": val_auc,  # Include key metrics
            "config": vars(args),
        }

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            save_checkpoint(
                checkpoint_state, output_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )

        # --- Save Best Checkpoints based on different metrics ---
        is_best_loss = val_loss < best_val_metrics["loss"]
        is_best_qwk = val_qwk > best_val_metrics["qwk"]
        is_best_combined = combined_metric > best_metric_combined

        if is_best_loss:
            best_val_metrics["loss"] = val_loss
            save_checkpoint(
                checkpoint_state, output_dir, "best_loss_checkpoint.pth", is_best=True
            )

        if is_best_qwk:
            best_val_metrics["qwk"] = val_qwk
            save_checkpoint(
                checkpoint_state, output_dir, "best_qwk_checkpoint.pth", is_best=True
            )

        if is_best_combined:
            logging.info(
                f"*** New best combined metric: {combined_metric:.4f} (previous: {best_metric_combined:.4f}) at epoch {epoch+1}"
            )
            best_metric_combined = combined_metric
            save_checkpoint(
                checkpoint_state,
                output_dir,
                "best_clinical_checkpoint.pth",
                is_best=True,
            )
            patience_counter = 0  # Reset patience because performance improved
        else:
            patience_counter += 1
            logging.info(
                f"Combined metric did not improve. Patience: {patience_counter}/{PATIENCE_LIMIT}"
            )

        # Update overall bests for other metrics (for final logging)
        best_val_metrics["accuracy"] = max(best_val_metrics["accuracy"], val_acc)
        best_val_metrics["f1"] = max(best_val_metrics["f1"], val_f1)
        best_val_metrics["sensitivity"] = max(
            best_val_metrics["sensitivity"], val_sensitivity
        )
        best_val_metrics["specificity"] = max(
            best_val_metrics["specificity"], val_specificity
        )
        best_val_metrics["auc"] = max(best_val_metrics["auc"], val_auc)

        # --- Log best metrics achieved so far to WandB ---
        if wandb_run:
            wandb_run.log(
                {
                    "val/best_loss_so_far": best_val_metrics["loss"],
                    "val/best_qwk_so_far": best_val_metrics["qwk"],
                    "val/best_combined_metric_so_far": best_metric_combined,
                    "val/patience_counter": patience_counter,
                    "epoch": epoch + 1,
                }
            )

        # --- Early Stopping ---
        if patience_counter >= PATIENCE_LIMIT:
            logging.info(f"--- Early stopping triggered at epoch {epoch+1} ---")
            break

    # --- End of Training ---
    total_training_time = time.time() - start_training_time
    logging.info(f"--- Training Complete ---")
    logging.info(
        f"Total Training Time: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}"
    )

    logging.info("--- Best Validation Metrics Achieved ---")
    logging.info(f"Best Loss: {best_val_metrics['loss']:.4f}")
    logging.info(f"Best Accuracy: {best_val_metrics['accuracy']:.4f}")
    logging.info(f"Best F1 (Weighted): {best_val_metrics['f1']:.4f}")
    logging.info(f"Best QWK: {best_val_metrics['qwk']:.4f}")
    logging.info(f"Best Avg Sensitivity: {best_val_metrics['sensitivity']:.4f}")
    logging.info(f"Best Avg Specificity: {best_val_metrics['specificity']:.4f}")
    logging.info(f"Best AUC (Macro-OvR): {best_val_metrics['auc']:.4f}")
    logging.info(f"Best Combined Clinical Metric: {best_metric_combined:.4f}")

    # Save final model state (optional)
    save_checkpoint(checkpoint_state, output_dir, "final_checkpoint.pth")

    if wandb_run:
        # Log final best metrics to WandB summary
        wandb_run.summary["best_val_loss"] = best_val_metrics["loss"]
        wandb_run.summary["best_val_accuracy"] = best_val_metrics["accuracy"]
        wandb_run.summary["best_val_f1_weighted"] = best_val_metrics["f1"]
        wandb_run.summary["best_val_qwk"] = best_val_metrics["qwk"]
        wandb_run.summary["best_val_avg_sensitivity"] = best_val_metrics["sensitivity"]
        wandb_run.summary["best_val_avg_specificity"] = best_val_metrics["specificity"]
        wandb_run.summary["best_val_auc_macro_ovr"] = best_val_metrics["auc"]
        wandb_run.summary["best_combined_clinical_metric"] = best_metric_combined
        wandb_run.summary["total_training_time_seconds"] = total_training_time

        wandb_run.finish()
        logging.info("WandB run finished.")


if __name__ == "__main__":
    main()
