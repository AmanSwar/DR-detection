import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import timm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging
import argparse
import warnings
from data_pipeline import data_set , data_aug

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Prototype loss function using cosine similarity
def prototype_loss(features, labels, prototypes):
    """
    Compute prototype loss based on cosine similarity between features and class prototypes.
    
    Args:
        features (torch.Tensor): Feature embeddings from the model.
        labels (torch.Tensor): Ground truth labels.
        prototypes (torch.Tensor): Class prototypes.
    
    Returns:
        torch.Tensor: Prototype loss value.
    """
    features_norm = F.normalize(features, dim=1)
    prototypes_norm = F.normalize(prototypes, dim=1)
    sim = torch.matmul(features_norm, prototypes_norm.T)
    return F.cross_entropy(sim, labels)

# Combined loss function with weighted cross-entropy
def combined_loss(outputs, labels, grade_outputs=None, domain_logits=None, domain_labels=None,
                  lambda_consistency=0.3, lambda_domain=0.1, class_weights=None):
    """
    Compute the combined loss including weighted cross-entropy, consistency, and domain loss.
    
    Args:
        outputs (torch.Tensor): Model logits.
        labels (torch.Tensor): Ground truth labels.
        grade_outputs (tuple, optional): Grade logits and ordinal thresholds.
        domain_logits (torch.Tensor, optional): Domain classification logits.
        domain_labels (torch.Tensor, optional): Domain labels.
        lambda_consistency (float): Weight for consistency loss.
        lambda_domain (float): Weight for domain loss.
        class_weights (torch.Tensor, optional): Weights for cross-entropy loss.
    
    Returns:
        torch.Tensor: Total combined loss.
    """
    if class_weights is not None:
        main_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        main_criterion = nn.CrossEntropyLoss()
    main_loss = main_criterion(outputs, labels)
    
    loss = main_loss
    
    if grade_outputs is not None:
        grade_logits, ordinal_thresholds = grade_outputs
        batch_size = labels.size(0)
        num_classes = grade_logits.size(1)
        
        targets = torch.zeros_like(grade_logits)
        for i in range(batch_size):
            targets[i, :labels[i]+1] = 1.0
            
        consistency_loss = F.binary_cross_entropy_with_logits(grade_logits, targets)
        
        if ordinal_thresholds is not None:
            ordinal_targets = torch.zeros_like(ordinal_thresholds)
            for i in range(batch_size):
                for k in range(ordinal_thresholds.size(1)):
                    if labels[i] > k:
                        ordinal_targets[i, k] = 1.0
            ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_thresholds, ordinal_targets)
            consistency_loss = 0.7 * consistency_loss + 0.3 * ordinal_loss
        
        loss += lambda_consistency * consistency_loss
    
    if domain_logits is not None and domain_labels is not None:
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(domain_logits, domain_labels)
        loss += lambda_domain * domain_loss
    
    return loss

# Mixup criterion for handling mixed targets
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixup loss for mixed targets.
    
    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): First set of labels.
        y_b (torch.Tensor): Second set of labels.
        lam (float): Mixup interpolation factor.
    
    Returns:
        torch.Tensor: Mixed loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Enhanced DR Classifier with prototypes and attention
class EnhancedDRClassifier(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=5, freeze_backbone=True):
        super().__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=False)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
        
        self.feature_dim = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.grade_head = nn.Linear(self.feature_dim, num_classes)
        self.domain_classifier = nn.Linear(self.feature_dim, 2)
        
        self.register_buffer("prototypes", torch.zeros(num_classes, self.feature_dim))
        self.register_buffer("prototype_counts", torch.zeros(num_classes))
    
    def unfreeze_stage(self, stage_idx):
        """Unfreeze a specific stage of the backbone."""
        if hasattr(self.backbone, 'stages'):
            for param in self.backbone.stages[stage_idx].parameters():
                param.requires_grad = True
        else:
            logging.warning("Backbone doesn't have 'stages' attribute.")
    
    def forward(self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None):
        h = self.backbone(x)  # Global average pooling
        
        attention_weights = self.attention(h)
        attended_features = h * attention_weights
        
        logits = self.classifier(attended_features)
        grade_outputs = self.grade_head(attended_features), None
        domain_logits = None
        
        if alpha > 0:
            domain_logits_reversed = self.domain_classifier(torch.flip(attended_features, dims=[0]))
            domain_logits = (1 - alpha) * self.domain_classifier(attended_features) + alpha * domain_logits_reversed
        
        if update_prototypes and labels is not None:
            with torch.no_grad():
                for i, label in enumerate(labels):
                    self.prototypes[label] = self.prototypes[label] * (self.prototype_counts[label] / (self.prototype_counts[label] + 1)) + \
                                             h[i] * (1 / (self.prototype_counts[label] + 1))
                    self.prototype_counts[label] += 1
        
        if get_attention:
            return logits, grade_outputs, domain_logits, h, attended_features
        return logits, grade_outputs, domain_logits

# Training function for one epoch
def train_one_epoch(model, dataloader, optimizer, device, epoch, lambda_consistency=0.3,
                    lambda_domain=0.1, domain_adaptation=True, use_mixup=True, mixup_alpha=0.4,
                    lambda_proto=0.1, class_weights=None):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    alpha = min(2.0, 0.1 * epoch) if domain_adaptation else 0.0
    
    for i, (images, labels, domain_labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        domain_labels = domain_labels.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup and np.random.rand() < 0.8:
            # Placeholder for mixup_data (assumed to be defined elsewhere)
            images_mixed, labels_a, labels_b, lam = images, labels, labels, 0.5  # Simplified
            mixed_target = True
        else:
            mixed_target = False
        
        if not mixed_target:
            logits, grade_outputs, domain_logits, features, _ = model(images, alpha=alpha,
                                                                     get_attention=True,
                                                                     update_prototypes=True,
                                                                     labels=labels)
        else:
            logits, grade_outputs, domain_logits = model(images, alpha=alpha, get_attention=False,
                                                         update_prototypes=False)
        
        if mixed_target:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            main_loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss = main_loss
            if domain_logits is not None:
                domain_criterion = nn.CrossEntropyLoss()
                domain_loss = domain_criterion(domain_logits, domain_labels)
                loss += lambda_domain * domain_loss
        else:
            loss = combined_loss(logits, labels, grade_outputs=grade_outputs,
                                 domain_logits=domain_logits, domain_labels=domain_labels,
                                 lambda_consistency=lambda_consistency, lambda_domain=lambda_domain,
                                 class_weights=class_weights)
            proto_loss = prototype_loss(features, labels, model.prototypes)
            loss += lambda_proto * proto_loss
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        if not mixed_target:
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        if i % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")
    
    avg_loss = running_loss / len(dataloader)
    
    if len(all_preds) > 0:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, acc, f1
    return avg_loss, None, None

# Validation function
def validate(model, dataloader, device, epoch, lambda_consistency=0.3, class_weights=None):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels, domain_labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            domain_labels = domain_labels.to(device)
            
            logits, grade_outputs, domain_logits = model(images, alpha=0.0)
            
            loss = combined_loss(logits, labels, grade_outputs=grade_outputs,
                                 domain_logits=domain_logits, domain_labels=domain_labels,
                                 lambda_consistency=lambda_consistency, class_weights=class_weights)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # Simplified metrics (sensitivity, specificity, QWK, AUC omitted for brevity)
    return avg_loss, acc, f1, None, None, None, None

# Test-time augmentation validation
def tta_validate(model, dataloader, device, num_augmentations=5, class_weights=None):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels, domain_labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            batch_logits = []
            for _ in range(num_augmentations):
                # Placeholder for augmentation (assumed to be handled elsewhere)
                aug_images = images  # Simplified
                logits, _, _ = model(aug_images)
                batch_logits.append(logits)
            
            avg_logits = torch.stack(batch_logits).mean(dim=0)
            _, predicted = torch.max(avg_logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1, None, None, None, None

# Main training loop
def main():
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXt for DR Classification")
    parser.add_argument("--checkpoint", type=str, default=None, help="model/checkpoint/moco/best_checkpoint.pth")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of DR classes")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lambda_consistency", type=float, default=0.3, help="Consistency loss weight")
    parser.add_argument("--lambda_domain", type=float, default=0.1, help="Domain loss weight")
    parser.add_argument("--lambda_proto", type=float, default=0.1, help="Prototype loss weight")
    parser.add_argument("--domain_adaptation", action="store_true", help="Enable domain adaptation")
    parser.add_argument("--use_mixup", action="store_true", help="Enable mixup")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define class weights for weighted cross-entropy
    class_weights = torch.tensor([1.0, 1.2, 1.5, 2.0, 2.5]).to(device)
    
    # Initialize model
    model = EnhancedDRClassifier(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        freeze_backbone=True
    ).to(device)
    
    # Initial optimizer with non-backbone parameters
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': args.lr},
        {'params': model.grade_head.parameters(), 'lr': args.lr},
        {'params': model.domain_classifier.parameters(), 'lr': args.lr},
        {'params': model.attention.parameters(), 'lr': args.lr * 1.5},
    ], weight_decay=args.weight_decay)
    
    # Scheduler for dynamic learning rate adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Unfreeze schedule for backbone stages
    unfreeze_epochs = [5, 10, 15, 20]
    stage_lrs = [args.lr / 5, args.lr / 10, args.lr / 20, args.lr / 40]  # For stages 3, 2, 1, 0
    
    train_transform = data_aug.MoCoSingleAug(img_size=args.img_size)
    val_transform = data_aug.MoCoSingleAug(img_size=args.img_size)

    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
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
    
    for epoch in range(args.epochs):
        # Gradual unfreezing of backbone stages
        for i, unfreeze_epoch in enumerate(unfreeze_epochs):
            if epoch == unfreeze_epoch:
                stage_idx = 3 - i
                logging.info(f"Unfreezing stage {stage_idx}")
                model.unfreeze_stage(stage_idx)
                optimizer.add_param_group({'params': model.backbone.stages[stage_idx].parameters(),
                                           'lr': stage_lrs[i]})
        
        # Train one epoch
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            lambda_consistency=args.lambda_consistency,
            lambda_domain=args.lambda_domain,
            domain_adaptation=args.domain_adaptation,
            use_mixup=args.use_mixup,
            mixup_alpha=0.4,
            lambda_proto=args.lambda_proto,
            class_weights=class_weights
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _, _, _ = validate(
            model, val_loader, device, epoch,
            lambda_consistency=args.lambda_consistency,
            class_weights=class_weights
        )
        
        # Step scheduler with validation loss
        scheduler.step(val_loss)
        
        logging.info(f"Epoch [{epoch+1}/{args.epochs}] "
                     f"Train Loss: {train_loss:.4f} Acc: {train_acc or 0:.4f} "
                     f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()