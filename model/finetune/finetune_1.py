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
from torch.optim.swa_utils import AveragedModel, SWALR

import torchvision
import timm
import wandb

from data_pipeline import data_aug, data_set

class EnhancedLesionAttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(EnhancedLesionAttentionModule, self).__init__()
        # Channel attention path
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        
        # Spatial attention path
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        if x.dim() != 4:
            x = x.unsqueeze(0)
            
        # Channel attention
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        
        channel_att = torch.sigmoid(avg_out + max_out)
        x_channel = x * channel_att
        
        # Spatial attention
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = torch.sigmoid(self.conv_spatial(spatial_input))
        
        return x_channel * spatial_att


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class EnhancedGradeConsistencyHead(nn.Module):
    """Enforces consistency between grade levels with improved architecture"""
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
        
        # Ordinal relationship encoder
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
        
        #load the backbone
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        moco_state_dict = checkpoint['model_state_dict']
        
        config = checkpoint['config']
        self.backbone = timm.create_model(config['base_model'], pretrained=False, num_classes=0)
        
        backbone_state_dict = {}
        for k, v in moco_state_dict.items():
            if k.startswith('query_encoder.'):
                backbone_state_dict[k.replace('query_encoder.', '')] = v
        self.backbone.load_state_dict(backbone_state_dict)
        

        #backbone freeze intially
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        

        self.feature_dim = self.backbone.num_features
        
        self.attention = EnhancedLesionAttentionModule(self.feature_dim)
        
        # Main classifier head
        self.classifier = nn.Sequential(
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
        
        # Domain classifier for domain adaptation (for multiple datasets)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)  # Number of datasets (eyepacs, aptos, ddr, idrid, messdr)
        )
        
        # Prototypes for each DR grade
        self.register_buffer('prototypes', torch.zeros(num_classes, self.feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in [self.classifier, self.grade_head.grade_predictor, 
                      self.grade_head.ordinal_encoder, self.domain_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None):
        features = self.backbone.forward_features(x)
      
        attended_features = self.attention(features)
        
        h = torch.mean(attended_features, dim=(2, 3))
        
        # Main class prediction
        logits = self.classifier(h)
        
        # Grade consistency prediction
        grade_outputs = self.grade_head(h)
        
        if update_prototypes and labels is not None:
            with torch.no_grad():
                for i, label in enumerate(labels):
                    self.prototypes[label] = self.prototypes[label] * (self.prototype_counts[label] / (self.prototype_counts[label] + 1)) + \
                                           h[i] * (1 / (self.prototype_counts[label] + 1))
                    self.prototype_counts[label] += 1
        
        if alpha > 0:
            reversed_features = GradientReversal.apply(h, alpha)
            domain_logits = self.domain_classifier(reversed_features)
        else:
            domain_logits = None
        
        if get_attention:
            return logits, grade_outputs, domain_logits, h, attended_features
        
        return logits, grade_outputs, domain_logits
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def unfreeze_stage(self, stage_idx):
        """Unfreeze a specific backbone stage"""
        if hasattr(self.backbone, 'stages'):
            for param in self.backbone.stages[stage_idx].parameters():
                param.requires_grad = True
        else:
            logging.warning("Backbone doesn't have 'stages' attribute. Can't unfreeze specific stage.")



def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def combined_loss(outputs, labels, grade_outputs=None, domain_logits=None, domain_labels=None, 
                 lambda_consistency=0.3, lambda_domain=0.1):

    main_criterion = nn.CrossEntropyLoss()
    main_loss = main_criterion(outputs, labels)
    loss = main_loss
    
    # Enhanced grade consistency loss
    if grade_outputs is not None:
        grade_logits, ordinal_thresholds = grade_outputs
        batch_size = labels.size(0)
        num_classes = grade_logits.size(1)
        
        targets = torch.zeros_like(grade_logits)
        for i in range(batch_size):
            targets[i, :labels[i]+1] = 1.0
            
        consistency_loss = F.binary_cross_entropy_with_logits(grade_logits, targets)
        
        # Ordinal regression loss for thresholds
        # For each threshold k, if class > k then threshold k should be 1, else 0
        if ordinal_thresholds is not None:
            ordinal_targets = torch.zeros_like(ordinal_thresholds)
            for i in range(batch_size):
                for k in range(ordinal_thresholds.size(1)):
                    if labels[i] > k:
                        ordinal_targets[i, k] = 1.0
            
            ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_thresholds, ordinal_targets)
            consistency_loss = 0.7 * consistency_loss + 0.3 * ordinal_loss
        
        loss += lambda_consistency * consistency_loss
    
    # Domain classification loss if available
    if domain_logits is not None and domain_labels is not None:
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(domain_logits, domain_labels)
        loss += lambda_domain * domain_loss
    
    return loss


def create_optimizer_with_unfrozen_stages(model, backbone_stages, last_unfrozen_stage, args):
    """Create optimizer with appropriate learning rates for different stages"""
    param_groups = [
        {'params': model.classifier.parameters(), 'lr': args.lr / 4},
        {'params': model.grade_head.parameters(), 'lr': args.lr / 4},
        {'params': model.domain_classifier.parameters(), 'lr': args.lr / 4},
        {'params': model.attention.parameters(), 'lr': args.lr / 3}
    ]
    
    for i in range(len(backbone_stages)):
        if i <= last_unfrozen_stage:
            # Lower learning rates for earlier stages (further from classification head)
            stage_lr = args.lr / (10 + (last_unfrozen_stage - i) * 5)
            param_groups.append({'params': backbone_stages[i].parameters(), 'lr': stage_lr})
    
    return optim.AdamW(param_groups, weight_decay=args.weight_decay, eps=1e-8)


def train_one_epoch(model, dataloader, optimizer, device, epoch, wandb_run, scaler=None, 
                   lambda_consistency=0.3, lambda_domain=0.1, domain_adaptation=True,
                   use_mixup=True, mixup_alpha=0.4):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # Gradually increase domain adaptation strength
    alpha = min(2.0, 0.1 * epoch) if domain_adaptation else 0.0
    
    for i, (images, labels, domain_labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        domain_labels = domain_labels.to(device)
            
        optimizer.zero_grad()
        
        # Apply mixup if enabled
        if use_mixup and np.random.rand() < 0.8:  # 80% chance of applying mixup
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            mixed_target = True
        else:
            mixed_target = False
            
        if scaler is not None:
            with autocast():
                logits, grade_outputs, domain_logits = model(images, alpha=alpha, update_prototypes=not mixed_target, labels=labels if not mixed_target else None)
                
                if mixed_target:
                    # Custom loss handling for mixed targets
                    criterion = nn.CrossEntropyLoss()
                    main_loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                    # For simplicity, don't apply consistency loss with mixup
                    loss = main_loss
                    if domain_logits is not None:
                        domain_criterion = nn.CrossEntropyLoss()
                        domain_loss = domain_criterion(domain_logits, domain_labels)
                        loss += lambda_domain * domain_loss
                else:
                    loss = combined_loss(
                        logits, labels, 
                        grade_outputs=grade_outputs, 
                        domain_logits=domain_logits, 
                        domain_labels=domain_labels,
                        lambda_consistency=lambda_consistency,
                        lambda_domain=lambda_domain
                    )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, grade_outputs, domain_logits = model(images, alpha=alpha, update_prototypes=not mixed_target, labels=labels if not mixed_target else None)
            
            if mixed_target:
                criterion = nn.CrossEntropyLoss()
                main_loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                loss = main_loss
                if domain_logits is not None:
                    domain_criterion = nn.CrossEntropyLoss()
                    domain_loss = domain_criterion(domain_logits, domain_labels)
                    loss += lambda_domain * domain_loss
            else:
                loss = combined_loss(
                    logits, labels, 
                    grade_outputs=grade_outputs, 
                    domain_logits=domain_logits, 
                    domain_labels=domain_labels,
                    lambda_consistency=lambda_consistency,
                    lambda_domain=lambda_domain
                )
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        
        # Only collect metrics for non-mixup batches
        if not mixed_target:
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        if i % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")
            wandb_run.log({
                "train_loss": loss.item(), 
                "learning_rate": current_lr,
                "batch": i + epoch * len(dataloader),
                "domain_alpha": alpha
            })
    
    avg_loss = running_loss / len(dataloader)
    
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


def tta_validate(model, dataloader, device, epoch, wandb_run, lambda_consistency=0.3, num_augs=5):
    """Validation with test-time augmentation"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            batch_size = images.size(0)
            labels = labels.to(device)
            
            # Store predictions from multiple augmentations
            batch_probs = []
            
            # Original images
            images_orig = images.to(device)
            logits, grade_outputs, _ = model(images_orig)
            probs = torch.softmax(logits, dim=1)
            batch_probs.append(probs)
            
            # Calculate loss with original images
            loss = combined_loss(
                logits, labels, 
                grade_outputs=grade_outputs, 
                lambda_consistency=lambda_consistency
            )
            running_loss += loss.item()
            
            # Apply augmentations
            for _ in range(num_augs - 1):
                # Apply different augmentations
                images_aug = images.clone()
                
                # Random brightness and contrast adjustments
                brightness = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
                contrast = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
                images_aug = images_aug * brightness + contrast - brightness * contrast
                
                # Random horizontal flip
                if torch.rand(1).item() > 0.5:
                    images_aug = torch.flip(images_aug, dims=[3])
                    
                # Random rotation (small angle)
                angle = torch.FloatTensor(1).uniform_(-10, 10).item()
                if hasattr(torchvision.transforms.functional, 'rotate'):
                    # For newer torchvision versions
                    images_aug = torchvision.transforms.functional.pad(images_aug, padding=20)
                    images_aug = torchvision.transforms.functional.rotate(images_aug, angle)
                    images_aug = torchvision.transforms.functional.center_crop(images_aug, (images.size(2), images.size(3)))
                else:
                    # Fallback for older versions
                    images_aug = torch.nn.functional.interpolate(images_aug, scale_factor=1.2)
                    images_aug = torchvision.transforms.functional.rotate(images_aug, angle)
                    images_aug = torch.nn.functional.interpolate(images_aug, size=(images.size(2), images.size(3)))
                
                images_aug = images_aug.to(device)
                logits, _, _ = model(images_aug)
                probs = torch.softmax(logits, dim=1)
                batch_probs.append(probs)
            
            # Average probabilities across augmentations
            batch_probs = torch.stack(batch_probs)
            avg_probs = torch.mean(batch_probs, dim=0)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(avg_probs.cpu().numpy())
    
    # Calculate metrics with the averaged probabilities
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)
    all_labels = np.array(all_labels)
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    try:
        auc_scores = []
        for i in range(5):  # 5 DR classes
            if i in np.unique(all_labels):
                y_true_binary = (all_labels == i).astype(int)
                y_prob_binary = all_probs[:, i]
                auc = roc_auc_score(y_true_binary, y_prob_binary)
                auc_scores.append(auc)
        
        mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        mean_auc = 0.0
    
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
    
    logging.info(f"TTA Validation - Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
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
        pass
    
    return avg_loss, acc, f1, avg_sensitivity, avg_specificity, qwk, mean_auc


def validate(model, dataloader, device, epoch, wandb_run, lambda_consistency=0.3):
    """Regular validation without test-time augmentation"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  
                images, labels, _ = batch
            else:  
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            logits, grade_outputs, _ = model(images)
            
            # Combined loss without domain adaptation during validation
            loss = combined_loss(
                logits, labels, 
                grade_outputs=grade_outputs, 
                lambda_consistency=lambda_consistency
            )
            
            running_loss += loss.item()
            
            # For metrics calculation
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    all_probs = np.array(all_probs)
    try:
        auc_scores = []
        for i in range(5):  # 5 DR classes
            if i in np.unique(all_labels):
                y_true_binary = (np.array(all_labels) == i).astype(int)
                y_prob_binary = all_probs[:, i]
                auc = roc_auc_score(y_true_binary, y_prob_binary)
                auc_scores.append(auc)
        
        mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        mean_auc = 0.0
    
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
        pass
    
    return avg_loss, acc, f1, avg_sensitivity, avg_specificity, qwk, mean_auc


def save_checkpoint(state, checkpoint_dir, filename):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MoCo model for Diabetic Retinopathy Classification")
    parser.add_argument("--checkpoint", type=str, default="model/new/chckpt/moco/new/best_checkpoint.pth",
                        help="Path to MoCo checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=5, 
                        help="Number of epochs to train with frozen backbone")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of DR classes")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--use_mixup", action="store_true", default=True, help="Use Mixup augmentation")
    parser.add_argument("--early_stopping", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--lambda_consistency", type=float, default=0.3, help="Weight for grade consistency loss")
    parser.add_argument("--lambda_domain", type=float, default=0.1, help="Weight for domain adaptation loss")
    parser.add_argument("--domain_adaptation", action="store_true", default=True, help="Use domain adaptation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("enhanced_finetune.log"),
            logging.StreamHandler()
        ]
    )

    checkpoint_dir = "model/new/chckpt/enhanced_finetune" "chckpt/finetune1"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize wandb if needed
    # Commented out but keeping structure for future use
    wandb_run = wandb.init(
            project="DR-finetuning_1",
            config=vars(args),
            name=f"enhanced_finetune_{args.img_size}_{args.lr}"
        )
    
    # Initialize model
    model = EnhancedDRClassifier(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        freeze_backbone=True
    ).to(device)
    
    # Define transformations
    train_transform = data_aug.MoCoSingleAug(img_size=args.img_size)
    val_transform = data_aug.MoCoSingleAug(img_size=args.img_size)

    # Define datasets
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    
    # Create data loaders
    train_loader = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_transform,
        batch_size=args.batch_size,
        num_workers=5,
        sampler=True
    ).get_loader()

    val_loader = data_set.UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=val_transform,
        batch_size=args.batch_size,
        num_workers=3,
        sampler=True
    ).get_loader()

    # Extract backbone layers for reference
    backbone_layers = []
    if hasattr(model.backbone, 'stages'):
        backbone_layers = [
            model.backbone.stem,
            *[model.backbone.stages[i] for i in range(len(model.backbone.stages))]
        ]
    
    # Initialize optimizer with frozen backbone
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': args.lr},
        {'params': model.grade_head.parameters(), 'lr': args.lr},
        {'params': model.domain_classifier.parameters(), 'lr': args.lr},
        {'params': model.attention.parameters(), 'lr': args.lr * 1.5},  # Higher LR for attention
        {'params': model.backbone.parameters(), 'lr': 0.0}  # Initially frozen
    ], weight_decay=args.weight_decay, eps=1e-8)
    
    # Calculate steps per epoch for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    # Use OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr, args.lr, args.lr, args.lr * 1.5, 0.0],  # Match parameter groups in optimizer
        total_steps=total_steps,
        pct_start=0.1,  # Warmup for 10% of training
        div_factor=25,  # Initial LR = max_lr/div_factor
        final_div_factor=1000,  # Final LR = max_lr/(div_factor*final_div_factor)
        anneal_strategy='cos'
    )
    
    # Initialize AMP scaler if using mixed precision
    scaler = GradScaler() if args.use_amp else None

    # Initialize metrics tracking
    best_val_metrics = {
        "loss": float('inf'),
        "accuracy": 0,
        "f1": 0,
        "sensitivity": 0,
        "specificity": 0,
        "qwk": 0,
        "auc": 0
    }
    
    # Early stopping setup
    patience = args.early_stopping
    patience_counter = 0
    best_metric = 0  

    # Training loop
    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Unfreeze backbone after specified epochs
        if epoch == args.freeze_epochs:
            logging.info("Unfreezing backbone for full fine-tuning")
            model.unfreeze_backbone()
            
            # Reset optimizer with adjusted learning rates
            optimizer = optim.AdamW([
                {'params': model.classifier.parameters(), 'lr': args.lr / 5},
                {'params': model.grade_head.parameters(), 'lr': args.lr / 5},
                {'params': model.domain_classifier.parameters(), 'lr': args.lr / 5},
                {'params': model.attention.parameters(), 'lr': args.lr / 4},
                {'params': model.backbone.parameters(), 'lr': args.lr / 10}
            ], weight_decay=args.weight_decay, eps=1e-8)
            
            # Recalculate remaining steps for the new scheduler
            remaining_epochs = args.epochs - epoch
            total_steps = steps_per_epoch * remaining_epochs
            
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[args.lr / 5, args.lr / 5, args.lr / 5, args.lr / 4, args.lr / 10],
                total_steps=total_steps,
                pct_start=0.1,
                div_factor=10,
                final_div_factor=100,
                anneal_strategy='cos'
            )
            
            # Reset scaler if using mixed precision
            if args.use_amp:
                scaler = GradScaler()
        
        # Train for one epoch
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device, epoch, wandb_run, 
            scaler=scaler, lambda_consistency=args.lambda_consistency, 
            lambda_domain=args.lambda_domain, domain_adaptation=args.domain_adaptation,
            use_mixup=args.use_mixup
        )
        
        # Validate model
        val_loss, val_acc, val_f1, val_sensitivity, val_specificity, val_qwk, val_auc = validate(
            model, val_loader, device, epoch, wandb_run, 
            lambda_consistency=args.lambda_consistency
        )
        
        # Calculate combined clinical metric (emphasizing sensitivity and specificity)
        combined_metric = 0.3 * val_acc + 0.4 * val_sensitivity + 0.3 * val_specificity
        
        # Save checkpoint with current state
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_sensitivity': val_sensitivity,
            'val_specificity': val_specificity,
            'val_qwk': val_qwk,
            'val_auc': val_auc,
            'config': vars(args)
        }
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs to save disk space
            save_checkpoint(checkpoint_state, checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Track if any metrics improved
        metrics_improved = False
        
        # Track individual metrics improvement
        if val_loss < best_val_metrics["loss"]:
            best_val_metrics["loss"] = val_loss
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_loss_checkpoint.pth")
            metrics_improved = True
        
        if val_acc > best_val_metrics["accuracy"]:
            best_val_metrics["accuracy"] = val_acc
            metrics_improved = True
            
        if val_f1 > best_val_metrics["f1"]:
            best_val_metrics["f1"] = val_f1
            metrics_improved = True
            
        if val_sensitivity > best_val_metrics["sensitivity"]:
            best_val_metrics["sensitivity"] = val_sensitivity
            metrics_improved = True
            
        if val_specificity > best_val_metrics["specificity"]:
            best_val_metrics["specificity"] = val_specificity
            metrics_improved = True
            
        if val_qwk > best_val_metrics["qwk"]:
            best_val_metrics["qwk"] = val_qwk
            metrics_improved = True
            
        if val_auc > best_val_metrics["auc"]:
            best_val_metrics["auc"] = val_auc
            metrics_improved = True
        
        # Save best combined clinical metric model
        if combined_metric > best_metric:
            best_metric = combined_metric
            save_checkpoint(checkpoint_state, checkpoint_dir, "best_clinical_checkpoint.pth")
            logging.info(f"New best combined metric: {best_metric:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"No improvement in combined metric. Patience: {patience_counter}/{patience}")
        
        # Update scheduler
        scheduler.step()
        
        # Early stopping check
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
    # Final logging
    logging.info("Training complete!")
    logging.info(f"Best validation loss: {best_val_metrics['loss']:.4f}")
    logging.info(f"Best validation accuracy: {best_val_metrics['accuracy']:.4f}")
    logging.info(f"Best validation F1: {best_val_metrics['f1']:.4f}")
    logging.info(f"Best validation sensitivity: {best_val_metrics['sensitivity']:.4f}")
    logging.info(f"Best validation specificity: {best_val_metrics['specificity']:.4f}")
    logging.info(f"Best validation QWK: {best_val_metrics['qwk']:.4f}")
    logging.info(f"Best validation AUC: {best_val_metrics['auc']:.4f}")
    logging.info(f"Best combined clinical metric: {best_metric:.4f}")
    wandb_run.finish()

if __name__ == "__main__":
    main()
