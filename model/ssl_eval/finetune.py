import os
import logging
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_

import timm
import torchvision.transforms as transforms

# Assuming these are custom modules; adjust imports as needed
from data_pipeline import data_aug, data_set

# CBAM Module
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        avg_pool = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1), 1, 1)
        max_pool = F.max_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1), 1, 1)
        avg_out = self.channel_mlp(avg_pool)
        max_out = self.channel_mlp(max_pool)
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        return x

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Grade Consistency Head with Ordinal Regression
class GradeConsistencyHead(nn.Module):
    def __init__(self, in_features, num_grades=5):
        super(GradeConsistencyHead, self).__init__()
        self.grade_predictor = nn.Linear(in_features, num_grades - 1)  # Predict P(grade >= k) for k=1 to 4

    def forward(self, x):
        return self.grade_predictor(x)  # Outputs logits for ordinal thresholds

class EnhancedDRClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        super(EnhancedDRClassifier, self).__init__()
        
        # Use EfficientNet-B3 pre-trained on ImageNet
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 1536 for EfficientNet-B3
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.attention = CBAM(self.feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.grade_head = GradeConsistencyHead(self.feature_dim, num_grades=num_classes)
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)  # 5 domains
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in [self.classifier, self.grade_head.grade_predictor, self.domain_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, alpha=0.0, get_attention=False):
        features = self.backbone.forward_features(x)
        attended_features = self.attention(features)
        h = torch.mean(attended_features, dim=(2, 3))
        
        logits = self.classifier(h)
        grade_logits = self.grade_head(h)
        
        if alpha > 0:
            reversed_features = GradientReversal.apply(h, alpha)
            domain_logits = self.domain_classifier(reversed_features)
        else:
            domain_logits = None
        
        if get_attention:
            return logits, grade_logits, domain_logits, attended_features
        return logits, grade_logits, domain_logits
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def combined_loss(outputs, labels, grade_logits=None, domain_logits=None, domain_labels=None, 
                  lambda_consistency=0.7, lambda_domain=0.1, class_weight=None, device='cuda'):
    # Focal loss for main classification
    main_criterion = FocalLoss(gamma=2.0, alpha=class_weight.to(device))
    main_loss = main_criterion(outputs, labels)
    
    total_loss = main_loss
    
    # Ordinal regression loss for grade consistency
    if grade_logits is not None:
        batch_size = labels.size(0)
        num_thresholds = grade_logits.size(1)  # num_classes - 1
        targets = torch.zeros_like(grade_logits)
        for i in range(batch_size):
            targets[i, :labels[i]] = 1.0  # P(grade >= k) = 1 for k <= label
        consistency_loss = F.binary_cross_entropy_with_logits(grade_logits, targets)
        total_loss += lambda_consistency * consistency_loss
    
    # Domain adaptation loss
    if domain_logits is not None and domain_labels is not None:
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(domain_logits, domain_labels)
        total_loss += lambda_domain * domain_loss
    
    return total_loss

def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None, 
                    lambda_consistency=0.7, lambda_domain=0.1, domain_adaptation=True, class_weight=None):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    alpha = min(2.0, 0.1 * epoch) if domain_adaptation else 0.0
    
    for i, (images, labels, domain_labels) in enumerate(dataloader):
        images, labels, domain_labels = images.to(device), labels.to(device), domain_labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits, grade_logits, domain_logits = model(images, alpha=alpha)
            loss = combined_loss(
                logits, labels, grade_logits, domain_logits, domain_labels,
                lambda_consistency, lambda_domain, class_weight, device
            )
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        if i % 10 == 0:
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1

def validate(model, dataloader, device, epoch, lambda_consistency=0.7, class_weight=None):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, grade_logits, _ = model(images)
            loss = combined_loss(logits, labels, grade_logits, lambda_consistency=lambda_consistency, 
                                class_weight=class_weight, device=device)
            running_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    all_probs = np.array(all_probs)
    auc_scores = []
    for i in range(5):
        if i in np.unique(all_labels):
            y_true = (np.array(all_labels) == i).astype(int)
            y_prob = all_probs[:, i]
            auc = roc_auc_score(y_true, y_prob)
            auc_scores.append(auc)
    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    
    sensitivity = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(len(cm))]
    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)
    
    logging.info(f"Validation - Epoch {epoch+1}: Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, "
                 f"Sens: {avg_sensitivity:.4f}, Spec: {avg_specificity:.4f}, QWK: {qwk:.4f}, AUC: {mean_auc:.4f}")
    
    return avg_loss, acc, f1, avg_sensitivity, avg_specificity, qwk, mean_auc

def main():
    parser = argparse.ArgumentParser(description="Enhanced DR Classification")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=10, help="Epochs with frozen backbone")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img_size", type=int, default=384, help="Image size (EfficientNet-B3 optimal)")
    parser.add_argument("--early_stopping", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--lambda_consistency", type=float, default=0.7, help="Consistency loss weight")
    parser.add_argument("--lambda_domain", type=float, default=0.1, help="Domain loss weight")
    parser.add_argument("--domain_adaptation", action="store_true", default=True, help="Use domain adaptation")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    checkpoint_dir = "model/enhanced_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Class weights
    class_weight = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], device=device)

    # Model
    model = EnhancedDRClassifier(num_classes=5, freeze_backbone=True).to(device)

    # Enhanced Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders (assuming custom data_set module)
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    train_loader = data_set.UniformTrainDataloader(
        dataset_names=dataset_names, transformation=train_transform, batch_size=args.batch_size, num_workers=4, sampler=True
    ).get_loader()
    val_loader = data_set.UniformValidDataloader(
        dataset_names=dataset_names, transformation=val_transform, batch_size=args.batch_size, num_workers=4, sampler=True
    ).get_loader()

    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': args.lr},
        {'params': model.grade_head.parameters(), 'lr': args.lr},
        {'params': model.domain_classifier.parameters(), 'lr': args.lr},
        {'params': model.attention.parameters(), 'lr': args.lr},
        {'params': model.backbone.parameters(), 'lr': 0.0}
    ], weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()

    # Training Loop
    best_metric = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            logging.info("Unfreezing backbone...")
            model.unfreeze_backbone()
            optimizer = optim.AdamW([
                {'params': model.classifier.parameters(), 'lr': 5e-4},
                {'params': model.grade_head.parameters(), 'lr': 5e-4},
                {'params': model.domain_classifier.parameters(), 'lr': 5e-4},
                {'params': model.attention.parameters(), 'lr': 5e-4},
                {'params': model.backbone.parameters(), 'lr': 1e-5}
            ], weight_decay=args.weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            scaler = GradScaler()
        
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scaler, 
            args.lambda_consistency, args.lambda_domain, args.domain_adaptation, class_weight
        )
        
        val_loss, val_acc, val_f1, val_sens, val_spec, val_qwk, val_auc = validate(
            model, val_loader, device, epoch, args.lambda_consistency, class_weight
        )
        
        scheduler.step()
        
        combined_metric = 0.3 * val_acc + 0.3 * val_sens + 0.3 * val_qwk + 0.1 * val_auc
        if combined_metric > best_metric:
            best_metric = combined_metric
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.early_stopping:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

if __name__ == "__main__":
    main()