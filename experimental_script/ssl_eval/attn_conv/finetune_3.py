import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


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
# import wandb

class MedicalTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim*4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # MHA expects (seq_len, bs, dim)
        attn_out, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.norm1(x + attn_out.permute(1, 0, 2))
        mlp_out = self.mlp(x)
        return self.norm2(x + mlp_out)
    

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
            nn.Linear(256, num_grades - 1)  # Predict binary thresholds between grades
        )
        
    def forward(self, x):
        logits = self.grade_predictor(x)
        ordinal_thresholds = self.ordinal_encoder(x)
        
        return logits, ordinal_thresholds

class EnhancedDRClassifier(nn.Module):
    def __init__(self, checkpoint_path, num_classes=5, freeze_backbone=True):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        moco_state_dict = checkpoint['model_state_dict']
        
        config = checkpoint['config']
        self.backbone = timm.create_model(config['base_model'], pretrained=False, num_classes=0)
        
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
        
        # Feature dimension
        self.feature_dim = self.backbone.num_features
        
        self.transformer = nn.Sequential(
            MedicalTransformerBlock(self.feature_dim),
            MedicalTransformerBlock(self.feature_dim)
        )
        
        # Enhanced 3D lesion attention
        self.lesion_attention = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 7, 7)),  # (depth, height, width)
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=(3, 7, 7)),
            nn.Sigmoid()
        )
        
        # Prototype memory bank
        self.register_buffer('prototypes', torch.zeros(num_classes, self.feature_dim))
        self.register_buffer('proto_counts', torch.zeros(num_classes))
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

    def forward(self, x):
        # Backbone features
        features = self.backbone.forward_features(x)  # [B, C, H, W]
        
        # Add channel dimension for 3D conv
        features_3d = features.unsqueeze(1)  # [B, 1, C, H, W]
        
        # 3D lesion attention
        attn_weights = self.lesion_attention(features_3d)
        attended_features = features_3d * attn_weights
        
        # Transformer processing
        b, _, c, h, w = attended_features.shape
        spatial_features = attended_features.view(b, c, -1).permute(0, 2, 1)
        transformed = self.transformer(spatial_features)
        
        # Pooling
        pooled = transformed.mean(dim=1)
        
        # Classifier
        logits = self.classifier(pooled)
        return logits
    

def medical_loss(outputs, labels, gamma=2.0):
    ce_loss = nn.CrossEntropyLoss()(outputs, labels)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    
    # Ordinal ranking loss
    diff = outputs[:, 1:] - outputs[:, :-1]
    rank_loss = torch.mean(F.relu(0.5 - diff))
    
    return focal_loss + 0.3*rank_loss

class GCAdam(optim.Adam):
    """Gradient Centralized Adam"""
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                grad -= grad.mean(dim=tuple(range(1, grad.ndim)), keepdim=True)
                p.grad.data = grad
        super().step()


def train_one_epoch(model, ema_model, train_loader, optimizer, scaler, epoch, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Progressive unfreezing
        if epoch > 5:
            model.unfreeze_backbone(epoch)
        
        with autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = medical_loss(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Update EMA model
        ema_model.update(model)
        
        # Update prototypes
        with torch.no_grad():
            features = model.backbone.forward_features(images)
            for cls_idx in labels.unique():
                mask = labels == cls_idx
                cls_features = features[mask].mean(dim=0)
                model.prototypes[cls_idx] = 0.9 * model.prototypes[cls_idx] + 0.1 * cls_features
                model.proto_counts[cls_idx] += mask.sum()
        
        # Metrics
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()
        
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Clinical metrics
    sensitivity = cm[1:, 1:].sum() / cm[1:].sum()
    specificity = cm[0, 0] / cm[0].sum()
    
    return total_loss/len(train_loader), acc, f1, sensitivity, specificity

def validate(ema_model, val_loader, device):
    ema_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = ema_model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = cm[1:, 1:].sum() / cm[1:].sum()
    specificity = cm[0, 0] / cm[0].sum()
    
    return acc, sensitivity, specificity

def main():
    # Initialize model
    model = EnhancedDRClassifier(checkpoint_path=args.checkpoint).to(device)
    ema_model = ModelEMA(model)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr/10)
    
    # Optimizer with gradient centralization
    optimizer = GCAdam([
        {'params': model.classifier.parameters(), 'lr': args.lr},
        {'params': model.transformer.parameters(), 'lr': args.lr*1.2},
        {'params': model.backbone.parameters(), 'lr': args.lr/10, 'weight_decay': 0}
    ], weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
                          steps_per_epoch=len(train_loader), 
                          epochs=args.epochs)
    
    # Training loop
    best_combined = 0
    for epoch in range(args.epochs):
        train_loss, acc, f1, sens, spec = train_one_epoch(
            model, ema_model, train_loader, optimizer, scaler, epoch, device
        )
        
        # Validation with EMA model
        val_acc, val_sens, val_spec = validate(ema_model, val_loader, device)
        combined = val_sens * 0.6 + val_spec * 0.4
        
        # SWA update
        if epoch > args.epochs//2:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # Save best clinical model
        if combined > best_combined:
            torch.save(ema_model.state_dict(), 'best_clinical_model.pth')
            best_combined = combined
            
        # Early stopping
        if (epoch - best_epoch) > args.patience:
            break

if __name__ == "__main__":
    main()