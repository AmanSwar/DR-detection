import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from captum.attr import LRP
from data_pipeline.data_set import UniformValidDataloader, UniformTrainDataloader
from data_pipeline.data_aug import MoCoSingleAug
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, cohen_kappa_score
)
import logging

# [Existing GradientReversal and GradeConsistencyHead classes remain unchanged]

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradeConsistencyHead(nn.Module):
    def __init__(self, feature_dim, num_grades=5):
        super(GradeConsistencyHead, self).__init__()
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
        self.ordinal_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_grades - 1)
        )
        
    def forward(self, x):
        logits = self.grade_predictor(x)
        ordinal_thresholds = self.ordinal_encoder(x)
        return logits, ordinal_thresholds
    



class LesionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(LesionAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        
        channel_att = torch.sigmoid(avg_out + max_out)
        x_channel = x * channel_att
        
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = torch.sigmoid(self.conv_spatial(spatial_input))
        output = x_channel * spatial_att
        
        return output, spatial_att  # Return both output and spatial attention map

class EnhancedDRClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        super(EnhancedDRClassifier, self).__init__()
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.attention = LesionAttentionModule(self.feature_dim)
        
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
        
        self.grade_head = GradeConsistencyHead(self.feature_dim, num_grades=num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)
        )
        
        self.register_buffer('prototypes', torch.zeros(num_classes, self.feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        
    def forward(self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None):
        features = self.backbone.forward_features(x)
        attended_features, spatial_att = self.attention(features)
        h = torch.mean(attended_features, dim=(2, 3))
        
        logits = self.classifier(h)
        grade_probs = self.grade_head(h)
        
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
            return logits, grade_probs, domain_logits, h, features, spatial_att
        return logits, grade_probs, domain_logits

# Device and model setup
device = torch.device("cuda")
model = EnhancedDRClassifier(num_classes=5, freeze_backbone=False).to(device)
checkpoint = torch.load("checkpoint/no_w/checkpoint_epoch_95.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

# Data loader setup
dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
val_transform = MoCoSingleAug(img_size=256)
val_loader = UniformTrainDataloader(
    dataset_names=dataset_names,
    transformation=val_transform,
    batch_size=8,
    num_workers=4,
    sampler=True
).get_loader()



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



# [Existing OrdinalDomainLoss function remains unchanged]

# Denormalization function for visualization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def denormalize(image):
    image = image.clone().cpu()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image.clamp(0, 1)

def normalize_map(map):
    return (map - map.min()) / (map.max() - map.min() + 1e-8)

def validate(model, dataloader, visualize=True, vis_dir='vis'):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    vis_images = None
    vis_labels = None

    with torch.no_grad():
        batch_idx = 0
        for batch_data in dataloader:
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass for metrics
            logits, grade_outputs, _ = model(images, alpha=0.0, update_prototypes=False)
            if visualize and batch_idx == 0:
                vis_images = images[:5].clone().detach().cpu()
                vis_labels = labels[:5].clone().detach().cpu()

            loss = OrdinalDomainLoss(
                logits, labels,
                grade_outputs=grade_outputs,
                domain_logits=None, domain_labels=None,
                lambda_consistency=0.1,
                lambda_domain=0.0,
                ordinal_weight=0.3,
                num_classes=5
            )

            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if visualize and batch_idx == 0:
                vis_images = images[:5].clone().detach().cpu()
                vis_labels = labels[:5].clone().detach().cpu()

            # Visualization for the first 5 images of the first batch
    if visualize and batch_idx == 0:
        os.makedirs(vis_dir, exist_ok=True)
        for i in range(min(5, images.size(0))):
            image = images[i].clone().detach().requires_grad_(True)
            label = labels[i]

            # Forward pass with attention
            logits, _, _, h, features, spatial_att = model(image.unsqueeze(0), alpha=0.0, get_attention=True)
            pred = logits.argmax(dim=1).item()

            # Compute gradients for Saliency and Grad-CAM
            model.zero_grad()
            logits[0, pred].backward()
            
            # Saliency Map
            saliency = image.grad.abs().max(dim=0)[0].cpu().numpy()
            saliency = normalize_map(saliency)
            
            # Grad-CAM
            grads = features.grad
            weights = F.adaptive_avg_pool2d(grads, 1).squeeze()
            cam = F.relu((weights.view(-1, 1, 1) * features).sum(dim=1)).squeeze().cpu().numpy()
            cam = cv2.resize(cam, (image.shape[2], image.shape[1]))
            cam = normalize_map(cam)
            
            # Attention Map
            spatial_att_map = spatial_att.squeeze().cpu().numpy()
            if spatial_att_map.shape != (image.shape[1], image.shape[2]):
                spatial_att_map = cv2.resize(spatial_att_map, (image.shape[2], image.shape[1]))
            spatial_att_map = normalize_map(spatial_att_map)
            
            # LRP
            lrp = LRP(model)
            attribution = lrp.attribute(image.unsqueeze(0), target=pred)
            attribution = attribution.squeeze().cpu().detach().numpy()
            if attribution.ndim == 3:
                attribution = np.sum(attribution, axis=0)
            attribution = normalize_map(attribution)
            
            # Save original image
            original_img = denormalize(image).numpy().transpose(1, 2, 0)
            plt.imsave(os.path.join(vis_dir, f'image_{i}.png'), original_img)
            
            # Save visualizations
            plt.imsave(os.path.join(vis_dir, f'saliency_{i}.png'), saliency, cmap='hot')
            plt.imsave(os.path.join(vis_dir, f'gradcam_{i}.png'), cam, cmap='hot')
            plt.imsave(os.path.join(vis_dir, f'attention_{i}.png'), spatial_att_map, cmap='hot')
            plt.imsave(os.path.join(vis_dir, f'lrp_{i}.png'), attribution, cmap='hot')

        batch_idx += 1

    # Calculate Metrics
    avg_loss = running_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    try:
        cm = confusion_matrix(all_labels, all_preds)
        sensitivity = []
        specificity = []
        for i in range(5):
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
        sensitivity = [0.0] * 5
        specificity = [0.0] * 5

    try:
        present_classes = np.unique(all_labels)
        if len(present_classes) == 5 and all_probs.shape[1] == 5:
            auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        elif len(present_classes) > 1:
            logging.warning(f"Only {len(present_classes)}/5 classes present. AUC might be unreliable.")
            auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro', labels=present_classes)
        else:
            auc_macro_ovr = 0.0
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        auc_macro_ovr = 0.0

    # Logging
    print(
        f"Validation: Loss: {avg_loss:.4f}, Acc: {acc:.4f}, "
        f"F1(W): {f1_weighted:.4f}, QWK: {qwk:.4f}"
    )
    print(
        f"Validation - Avg Sensitivity: {avg_sensitivity:.4f}, Avg Specificity: {avg_specificity:.4f}, AUC(Macro-OvR): {auc_macro_ovr:.4f}"
    )

# Run validation with visualization
validate(model=model, dataloader=val_loader, visualize=True)