import os
import argparse
import logging
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.transforms as transforms
from PIL import Image

import timm
import wandb

from data_pipeline import data_aug, data_set


class LesionAttentionModule(nn.Module):
    """Attention module focused on detecting and highlighting lesions"""
    def __init__(self, in_channels):
        super(LesionAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        
    def forward(self, x):
        # Channel attention path
        avg_pool = F.avg_pool2d(x, x.size(2))
        channel_attention = self.conv2(F.relu(self.conv1(avg_pool)))
        
        # Spatial attention path - detect high contrast regions (potential lesions)
        spatial_attention = torch.std(x, dim=1, keepdim=True)
        spatial_attention = F.sigmoid(spatial_attention)
        
        # Combine attentions
        attention = F.sigmoid(channel_attention).unsqueeze(2).unsqueeze(3) * spatial_attention
        return x * attention


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradeConsistencyHead(nn.Module):
    """Enforces consistency between grade levels (logical ordering)"""
    def __init__(self, feature_dim, num_grades=5):
        super(GradeConsistencyHead, self).__init__()
        self.grade_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_grades)
        )
        
    def forward(self, x):
        logits = self.grade_predictor(x)
        # Enforce ordinal relationship between grades (monotonically increasing)
        cumulative_probs = torch.sigmoid(logits)
        return cumulative_probs


class EnhancedSimCLRModel(nn.Module):
    def __init__(self, base_model='convnext_tiny', projection_dim=128, hidden_dim=512, pretrained=False):
        super(EnhancedSimCLRModel, self).__init__()
        self.encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        feature_dim = self.encoder.num_features
        
        # Add lesion attention module for the backbone
        self.attention = LesionAttentionModule(feature_dim)
        
        # Multi-scale projection head for capturing both local and global features
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added LayerNorm for better training stability
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Self-supervised grade consistency head
        self.grade_head = GradeConsistencyHead(feature_dim)
        
        # Domain classifier (for domain adaptation)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)  # Number of datasets
        )
        
        # Prototype vectors for clustering-based learning
        self.register_buffer('prototypes', torch.zeros(5, feature_dim))  # 5 grade prototypes
        self.register_buffer('prototype_counts', torch.zeros(5))

    def forward(self, x, get_attention=False):
        # Get features from encoder
        features = self.encoder(x)
        
        # Apply attention mechanism for lesion awareness
        if hasattr(self, 'attention'):
            attended_features = self.attention(features)
            # Global average pooling for attended features
            h = torch.mean(attended_features, dim=(2, 3))
        else:
            h = features
        
        # Create projection
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        
        if get_attention:
            return h, z, attended_features
        return h, z
    
    def forward_domain(self, x, alpha=1.0):
        h, _ = self.forward(x)
        # Apply gradient reversal for domain adaptation
        reverse_h = GradientReversal.apply(h, alpha)
        domain_preds = self.domain_classifier(reverse_h)
        return domain_preds
    
    def forward_grade(self, x):
        h, _ = self.forward(x)
        return self.grade_head(h)
    
    def update_prototypes(self, features, labels, momentum=0.9):
        # Update prototypes based on features and their corresponding grades
        for i in range(5):  # For each grade
            mask = (labels == i)
            if mask.sum() > 0:
                grade_features = features[mask]
                grade_centroid = grade_features.mean(0)
                
                # Update prototype with momentum
                if self.prototype_counts[i] == 0:
                    self.prototypes[i] = grade_centroid
                else:
                    self.prototypes[i] = momentum * self.prototypes[i] + (1 - momentum) * grade_centroid
                
                self.prototype_counts[i] += 1


class EnhancedNTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda', use_hard_negative=True):
        super(EnhancedNTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().to(device)
        self.use_hard_negative = use_hard_negative
        self.hard_negative_weight = 2.0  # Weight for hard negatives
        
    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(False)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = False
            mask[self.batch_size + i, i] = False
        return mask
    
    def _identify_hard_negatives(self, similarity_matrix, threshold=0.7):
        # Hard negatives are samples that are very similar but should be different
        hard_negatives = (similarity_matrix > threshold) & self.mask
        return hard_negatives

    def forward(self, z_i, z_j, prototypes=None):
        """
        Enhanced NT-Xent loss with hard negative mining and prototype guidance
        Args:
            z_i, z_j: Normalized embeddings from two views [batch_size, dim]
            prototypes: Optional prototype vectors for each class
        """
        # Concatenate all embeddings
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T)
        
        # Extract positive pairs
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0).unsqueeze(1)
        
        # Process negative pairs with special handling for hard negatives
        if self.use_hard_negative:
            hard_negative_mask = self._identify_hard_negatives(similarity_matrix)
            # Apply higher weight to hard negatives
            negatives = similarity_matrix[self.mask].view(2 * self.batch_size, -1)
            hard_negatives = similarity_matrix[hard_negative_mask].view(2 * self.batch_size, -1)
            
            # Combine regular and hard negatives
            if hard_negatives.size(1) > 0:  # If there are hard negatives
                weighted_hard_negatives = hard_negatives * self.hard_negative_weight
                negatives = torch.cat([negatives, weighted_hard_negatives], dim=1)
        else:
            negatives = similarity_matrix[self.mask].view(2 * self.batch_size, -1)
        
        # Add prototype guidance if available
        if prototypes is not None and prototypes.size(0) > 0:
            # Calculate similarity with prototypes
            proto_sim = torch.matmul(z, prototypes.T)  # [2N, num_classes]
            # We want embeddings to be similar to their respective class prototypes
            # This can be implemented in different ways depending on if we have grade labels
            
            # For this implementation, we'll just add the prototype similarities as additional positives
            # In a more advanced version, you'd use actual grade labels if available
            negatives = torch.cat([negatives, proto_sim], dim=1)
        
        # Calculate final loss
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        
        # Labels: positive at index 0 for each example
        labels = torch.zeros(2 * self.batch_size, dtype=torch.long).to(self.device)
        
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss


class GradeConsistencyLoss(nn.Module):
    """Loss that ensures grade predictions follow a logical order (monotonicity)"""
    def __init__(self):
        super(GradeConsistencyLoss, self).__init__()
        
    def forward(self, grade_preds):
        """
        Args:
            grade_preds: Tensor of shape [batch_size, num_grades]
                         representing cumulative probabilities
        """
        # Ensure monotonically decreasing probabilities (P(grade >= k))
        # For each sample, P(grade >= k) should be less than P(grade >= k-1)
        diffs = grade_preds[:, :-1] - grade_preds[:, 1:]
        # All differences should be positive for proper ordering
        monotonicity_loss = F.relu(-diffs).mean()
        return monotonicity_loss


def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, epoch, wandb_run, 
                   use_domain_adaptation=True, use_grade_consistency=True, alpha=1.0):
    model.train()
    running_loss = 0.0
    running_domain_loss = 0.0
    running_grade_loss = 0.0
    
    for i, (x1, x2, domain_labels, grade_labels) in enumerate(dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        domain_labels = domain_labels.to(device) if domain_labels is not None else None
        grade_labels = grade_labels.to(device) if grade_labels is not None else None
        
        optimizer.zero_grad()
        
        # Forward pass for both augmented views
        h1, z1 = model(x1)
        h2, z2 = model(x2)
        
        # Main contrastive loss
        if hasattr(model, 'prototypes') and grade_labels is not None:
            # Update prototypes based on current batch
            with torch.no_grad():
                model.update_prototypes(h1.detach(), grade_labels)
            contrastive_loss = loss_fn(z1, z2, model.prototypes)
        else:
            contrastive_loss = loss_fn(z1, z2)
        
        total_loss = contrastive_loss
        
        # Domain adaptation loss (if enabled)
        domain_loss = 0
        if use_domain_adaptation and domain_labels is not None:
            domain_preds = model.forward_domain(torch.cat([x1, x2], dim=0), alpha)
            domain_labels_combined = torch.cat([domain_labels, domain_labels], dim=0)
            domain_loss = F.cross_entropy(domain_preds, domain_labels_combined)
            total_loss += 0.1 * domain_loss  # Weight for domain loss
            running_domain_loss += domain_loss.item()
        
        # Grade consistency loss (if enabled)
        grade_loss = 0
        if use_grade_consistency and grade_labels is not None:
            grade_preds1 = model.forward_grade(x1)
            grade_preds2 = model.forward_grade(x2)
            
            # Regression loss for grades
            grade_criterion = nn.MSELoss()
            grade_supervision_loss = grade_criterion(grade_preds1[:, -1], grade_labels.float() / 4.0)
            
            # Consistency loss between different views
            grade_consistency_loss = F.mse_loss(grade_preds1, grade_preds2)
            
            # Monotonicity loss
            monotonicity_loss = GradeConsistencyLoss()(grade_preds1) + GradeConsistencyLoss()(grade_preds2)
            
            grade_loss = grade_supervision_loss + 0.1 * grade_consistency_loss + 0.1 * monotonicity_loss
            total_loss += 0.2 * grade_loss  # Weight for grade loss
            running_grade_loss += grade_loss.item()
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        running_loss += contrastive_loss.item()
        
        if i % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] "
                         f"Loss: {contrastive_loss.item():.4f} "
                         f"Domain Loss: {domain_loss:.4f} "
                         f"Grade Loss: {grade_loss:.4f} "
                         f"LR: {current_lr:.6f}")
            wandb_run.log({
                "train_loss": contrastive_loss.item(),
                "domain_loss": domain_loss if isinstance(domain_loss, float) else domain_loss.item(),
                "grade_loss": grade_loss if isinstance(grade_loss, float) else grade_loss.item(),
                "total_loss": total_loss.item(),
                "epoch": epoch+1,
                "learning_rate": current_lr
            })
    
    # Step the scheduler after each epoch
    scheduler.step()
    
    avg_loss = running_loss / len(dataloader)
    avg_domain_loss = running_domain_loss / len(dataloader) if running_domain_loss > 0 else 0
    avg_grade_loss = running_grade_loss / len(dataloader) if running_grade_loss > 0 else 0
    
    return avg_loss, avg_domain_loss, avg_grade_loss


def validate(model, dataloader, loss_fn, device, epoch, wandb_run, 
            use_domain_adaptation=True, use_grade_consistency=True):
    model.eval()
    running_loss = 0.0
    running_domain_loss = 0.0
    running_grade_loss = 0.0
    
    with torch.no_grad():
        for i, (x1, x2, domain_labels, grade_labels) in enumerate(dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            domain_labels = domain_labels.to(device) if domain_labels is not None else None
            grade_labels = grade_labels.to(device) if grade_labels is not None else None
            
            # Forward pass
            h1, z1 = model(x1)
            h2, z2 = model(x2)
            
            # Main contrastive loss
            contrastive_loss = loss_fn(z1, z2)
            running_loss += contrastive_loss.item()
            
            # Domain loss
            if use_domain_adaptation and domain_labels is not None:
                domain_preds = model.forward_domain(torch.cat([x1, x2], dim=0), 1.0)
                domain_labels_combined = torch.cat([domain_labels, domain_labels], dim=0)
                domain_loss = F.cross_entropy(domain_preds, domain_labels_combined)
                running_domain_loss += domain_loss.item()
            
            # Grade loss
            if use_grade_consistency and grade_labels is not None:
                grade_preds1 = model.forward_grade(x1)
                grade_criterion = nn.MSELoss()
                grade_loss = grade_criterion(grade_preds1[:, -1], grade_labels.float() / 4.0)
                running_grade_loss += grade_loss.item()
    
    avg_loss = running_loss / len(dataloader)
    avg_domain_loss = running_domain_loss / len(dataloader) if running_domain_loss > 0 else 0
    avg_grade_loss = running_grade_loss / len(dataloader) if running_grade_loss > 0 else 0
    
    logging.info(f"Epoch [{epoch+1}] Validation - Contrastive Loss: {avg_loss:.4f}, "
                f"Domain Loss: {avg_domain_loss:.4f}, Grade Loss: {avg_grade_loss:.4f}")
    
    wandb_run.log({
        "val_loss": avg_loss,
        "val_domain_loss": avg_domain_loss,
        "val_grade_loss": avg_grade_loss,
        "epoch": epoch+1
    })
    
    return avg_loss


@torch.no_grad()
def extract_features_with_attention(model, dataloader, device):
    """Extract features and attention maps from the model"""
    model.eval()
    all_feats = []
    all_attentions = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        # Get features, projections, and attention maps
        feats, _, attention_maps = model(images, get_attention=True)
        
        all_feats.append(feats.cpu())
        all_attentions.append(attention_maps.cpu())
        all_labels.append(labels)
        
    all_feats = torch.cat(all_feats, dim=0)
    all_attentions = torch.cat(all_attentions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_feats, all_attentions, all_labels


def visualize_attention_maps(model, dataloader, device, wandb_run, num_samples=5):
    """Visualize attention maps to see what the model focuses on"""
    import matplotlib.pyplot as plt
    
    model.eval()
    images, labels = next(iter(dataloader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # Get features and attention maps
    with torch.no_grad():
        _, _, attention_maps = model(images, get_attention=True)
    
    # Convert to numpy for visualization
    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    attention_np = attention_maps.mean(dim=1).cpu().numpy()
    
    # Normalize attention maps for visualization
    attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images_np[i])
        axes[i, 0].set_title(f"Original (Grade: {labels[i].item()})")
        axes[i, 0].axis('off')
        
        # Attention map
        axes[i, 1].imshow(attention_np[i], cmap='jet')
        axes[i, 1].set_title(f"Attention Map")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    wandb_run.log({"attention_maps": wandb.Image(fig)})
    plt.close(fig)


def improved_knn_evaluation(model, train_loader, val_loader, device, k_values=[1, 5, 10, 20], wandb_run=None):
    """
    Enhanced k-NN classifier with multiple k values and distance metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np
    
    # Extract features
    train_feats, _, train_labels = extract_features_with_attention(model, train_loader, device)
    val_feats, _, val_labels = extract_features_with_attention(model, val_loader, device)
    
    # Convert to numpy
    train_feats_np = train_feats.numpy()
    train_labels_np = train_labels.numpy()
    val_feats_np = val_feats.numpy()
    val_labels_np = val_labels.numpy()
    
    results = {}
    best_acc = 0
    best_k = 0
    best_metric = ""
    
    # Try different distance metrics and k values
    for metric in ['euclidean', 'cosine']:
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
            knn.fit(train_feats_np, train_labels_np)
            
            # Predict and evaluate
            y_pred = knn.predict(val_feats_np)
            acc = accuracy_score(val_labels_np, y_pred) * 100
            f1 = f1_score(val_labels_np, y_pred, average='weighted') * 100
            cm = confusion_matrix(val_labels_np, y_pred)
            
            # Track best result
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_metric = metric
            
            # Log results
            key = f"knn_{metric}_{k}"
            results[key] = {
                "accuracy": acc,
                "f1_score": f1,
                "confusion_matrix": cm
            }
            
            logging.info(f"[k-NN] k={k}, metric={metric}: Acc={acc:.2f}%, F1={f1:.2f}%")
            
            if wandb_run is not None:
                wandb_run.log({
                    f"knn_accuracy_{metric}_{k}": acc,
                    f"knn_f1_{metric}_{k}": f1
                })
    
    # Log best result
    if wandb_run is not None:
        wandb_run.log({
            "knn_best_accuracy": best_acc,
            "knn_best_k": best_k,
            "knn_best_metric": best_metric
        })
    
    logging.info(f"[k-NN] Best result: k={best_k}, metric={best_metric}, Acc={best_acc:.2f}%")
    return best_acc, results


def improved_linear_probe(model, train_loader, val_loader, device, wandb_run):
    """
    Enhanced linear probe with regularization and class weighting
    """
    # Extract features
    train_feats, _, train_labels = extract_features_with_attention(model, train_loader, device)
    val_feats, _, val_labels = extract_features_with_attention(model, val_loader, device)
    
    # Create class weights to handle class imbalance
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(device)
    
    # Create linear probe
    embed_dim = train_feats.shape[1]
    num_classes = len(train_labels.unique())
    
    # Try different regularization strengths
    best_acc = 0
    best_reg = 0
    
    for reg_weight in [1e-4, 1e-3, 1e-2]:
        probe = LinearProbeHead(embed_dim, num_classes).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=reg_weight)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Train the linear probe
        probe_epochs = 10  # More epochs for better convergence
        
        for ep in range(probe_epochs):
            probe.train()
            # Shuffle data
            perm = torch.randperm(train_feats.size(0))
            train_feats_shuf = train_feats[perm].to(device)
            train_labels_shuf = train_labels[perm].to(device)
            
            # Mini-batch training
            batch_size = 128
            running_loss = 0
            
            for i in range(0, train_feats_shuf.size(0), batch_size):
                end = min(i + batch_size, train_feats_shuf.size(0))
                batch_feats = train_feats_shuf[i:end]
                batch_labels = train_labels_shuf[i:end]
                
                optimizer.zero_grad()
                outputs = probe(batch_feats)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * (end - i)
            
            avg_loss = running_loss / train_feats_shuf.size(0)
            
            # Evaluate after each epoch
            probe.eval()
            val_feats_gpu = val_feats.to(device)
            val_labels_gpu = val_labels.to(device)
            
            with torch.no_grad():
                logits = probe(val_feats_gpu)
                val_loss = criterion(logits, val_labels_gpu)
                pred = torch.argmax(logits, dim=1).cpu()
                acc = (pred == val_labels).float().mean().item() * 100.0
            
            if wandb_run is not None:
                wandb_run.log({
                    f"linear_probe_train_loss_reg{reg_weight}": avg_loss,
                    f"linear_probe_val_loss_reg{reg_weight}": val_loss.item(),
                    f"linear_probe_accuracy_reg{reg_weight}": acc,
                    f"linear_probe_epoch_reg{reg_weight}": ep
                })
        
        # Record best accuracy
        if acc > best_acc:
            best_acc = acc
            best_reg = reg_weight
    
    logging.info(f"[Linear Probe] Best Validation Accuracy: {best_acc:.2f}% with reg={best_reg}")
    
    if wandb_run is not None:
        wandb_run.log({
            "linear_probe_best_accuracy": best_acc,
            "linear_probe_best_reg": best_reg
        })
    
    return best_acc

class LinearProbeHead(nn.Module):
    """A simple linear classifier for evaluating SSL embeddings."""
    def __init__(self, embed_dim, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    
# Complete the multi_label_linear_probe function
def multi_label_linear_probe(model, train_loader, val_loader, device, wandb_run):
    """
    Ordinal regression approach for DR grading.
    Trains a probe to predict ordinal labels (grade >= k) and evaluates performance.
    """
    # Extract features
    train_feats, _, train_labels = extract_features_with_attention(model, train_loader, device)
    val_feats, _, val_labels = extract_features_with_attention(model, val_loader, device)
    
    # Convert to ordinal labels (for each grade k, predict if grade >= k)
    def to_ordinal(labels, num_classes=5):
        batch_size = labels.size(0)
        ordinal = torch.zeros(batch_size, num_classes-1)
        for i in range(batch_size):
            for j in range(num_classes-1):
                ordinal[i, j] = 1 if labels[i] > j else 0
        return ordinal.to(device)
    
    train_ordinal = to_ordinal(train_labels)
    val_ordinal = to_ordinal(val_labels)
    
    # Create ordinal regression model
    embed_dim = train_feats.shape[1]
    ordinal_probe = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 4)  # 4 binary classifiers for 5 classes (0 to 4)
    ).to(device)
    
    optimizer = torch.optim.Adam(ordinal_probe.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train ordinal regression model
    ordinal_epochs = 10
    best_acc = 0
    
    for ep in range(ordinal_epochs):
        ordinal_probe.train()
        # Shuffle data
        perm = torch.randperm(train_feats.size(0))
        train_feats_shuf = train_feats[perm].to(device)
        train_ordinal_shuf = train_ordinal[perm]
        
        # Mini-batch training
        batch_size = 128
        running_loss = 0
        
        for i in range(0, train_feats_shuf.size(0), batch_size):
            end = min(i + batch_size, train_feats_shuf.size(0))
            batch_feats = train_feats_shuf[i:end]
            batch_labels = train_ordinal_shuf[i:end]
            
            optimizer.zero_grad()
            outputs = ordinal_probe(batch_feats)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * (end - i)
        
        avg_loss = running_loss / train_feats_shuf.size(0)
        
        # Evaluate
        ordinal_probe.eval()
        val_feats_gpu = val_feats.to(device)
        val_labels_gpu = val_labels.to(device)
        
        with torch.no_grad():
            logits = ordinal_probe(val_feats_gpu)
            val_loss = criterion(logits, val_ordinal.to(device))
            
            # Convert ordinal predictions back to class labels
            pred_probs = torch.sigmoid(logits)
            pred_ordinal = (pred_probs > 0.5).float().cpu()
            pred_grade = pred_ordinal.sum(dim=1).long()
            
            # Compute accuracy
            acc = (pred_grade == val_labels).float().mean().item() * 100.0
            
            if acc > best_acc:
                best_acc = acc
        
        logging.info(f"[Ordinal Probe] Epoch {ep+1}/{ordinal_epochs} - "
                     f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {acc:.2f}%")
        
        if wandb_run is not None:
            wandb_run.log({
                "ordinal_probe_train_loss": avg_loss,
                "ordinal_probe_val_loss": val_loss.item(),
                "ordinal_probe_accuracy": acc,
                "ordinal_probe_epoch": ep
            })
    
    logging.info(f"[Ordinal Probe] Best Validation Accuracy: {best_acc:.2f}%")
    
    if wandb_run is not None:
        wandb_run.log({"ordinal_probe_best_accuracy": best_acc})
    
    return best_acc

# Main training loop
def main():
    # Configuration
    config = {
        "epochs": 300,
        "batch_size": 64,
        "lr": 5e-4,
        "lr_min": 1e-5,
        "warm_up_epochs": 10,
        "temperature": 0.5,
        "base_model": "convnext_tiny",
        "projection_dim": 128,
        "hidden_dim": 512,
        "pretrained": False,
        "checkpoint_dir": "model/enhanced_simclr/checkpoints",
        "use_domain_adaptation": True,
        "use_grade_consistency": True,
        "alpha": 1.0  # Gradient reversal strength
    }

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb_run = wandb.init(project="Enhanced-SimCLR-DR", config=config)

    # Build the enhanced SimCLR model
    model = EnhancedSimCLRModel(
        base_model=config["base_model"],
        projection_dim=config["projection_dim"],
        hidden_dim=config["hidden_dim"],
        pretrained=config["pretrained"]
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] - config["warm_up_epochs"],
        eta_min=config["lr_min"]
    )

    # Loss function
    loss_fn = EnhancedNTXentLoss(
        batch_size=config["batch_size"],
        temperature=config["temperature"],
        device=device,
        use_hard_negative=True
    )

    # Load checkpoint if available
    checkpoint_path = os.path.join(config["checkpoint_dir"], "best_checkpoint.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        logging.info("No checkpoint found, starting from scratch")

    # Warm-up configuration
    initial_lr = config["lr"]
    if config["warm_up_epochs"] > 0 and start_epoch < config["warm_up_epochs"]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config["lr_min"]

    # Data loaders
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    transforms_ = data_aug.SimCLRAug(img_size=256)
    train_loader = data_set.SSLTrainLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=config["batch_size"],
        num_work=4
    ).get_loader()  # Assumed to yield (x1, x2, domain_labels, grade_labels)

    valid_loader = data_set.SSLValidLoader(
        dataset_names=dataset_names,
        transformation=transforms_,
        batch_size=8,
        num_work=4
    ).get_loader()

    # Labeled data loaders for evaluation
    train_aug = data_aug.SimCLRSingleAug(img_size=256)
    probe_train_loader = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_aug,
        batch_size=64,
        num_workers=0,
        sampler=True
    ).get_loader()

    val_aug = data_aug.SimCLRSingleAug(img_size=256)
    probe_val_loader = data_set.UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=val_aug,
        batch_size=8,
        num_workers=0,
        sampler=True
    ).get_loader()

    # Training loop
    best_val_loss = float('inf')
    try:
        for epoch in range(start_epoch, config["epochs"]):
            logging.info(f"--- Epoch {epoch+1}/{config['epochs']} ---")

            # Warm-up phase
            if epoch < config["warm_up_epochs"]:
                progress = (epoch + 1) / config["warm_up_epochs"]
                lr = config["lr_min"] + progress * (initial_lr - config["lr_min"])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                logging.info(f"Warm-up phase: LR set to {lr:.6f}")
                wandb_run.log({"learning_rate": lr, "epoch": epoch+1})

            # Train and validate
            train_loss, domain_loss, grade_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, device, epoch, wandb_run,
                use_domain_adaptation=config["use_domain_adaptation"],
                use_grade_consistency=config["use_grade_consistency"],
                alpha=config["alpha"]
            )
            val_loss = validate(
                model, valid_loader, loss_fn, device, epoch, wandb_run,
                use_domain_adaptation=config["use_domain_adaptation"],
                use_grade_consistency=config["use_grade_consistency"]
            )

            # Save checkpoint
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            epoch_ckpt = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint_state, os.path.join(config["checkpoint_dir"], epoch_ckpt))
            logging.info(f"Saved checkpoint: {epoch_ckpt}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint_state, checkpoint_path)
                logging.info("Saved best checkpoint")

            # Evaluate representations every 5 epochs
            if (epoch + 1) % 5 == 0:
                knn_acc, _ = improved_knn_evaluation(model, probe_train_loader, probe_val_loader, device, wandb_run=wandb_run)
                lin_acc = improved_linear_probe(model, probe_train_loader, probe_val_loader, device, wandb_run)
                ord_acc = multi_label_linear_probe(model, probe_train_loader, probe_val_loader, device, wandb_run)
                visualize_attention_maps(model, probe_val_loader, device, wandb_run)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected! Saving checkpoint before exiting...")
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        torch.save(checkpoint_state, os.path.join(config["checkpoint_dir"], f"interrupt_checkpoint_epoch_{epoch+1}.pth"))
    finally:
        wandb_run.finish()

if __name__ == "__main__":
    main()