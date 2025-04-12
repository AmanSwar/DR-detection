import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import (
    GradientShap, 
    LayerGradCam, 
    Occlusion, 
    IntegratedGradients, 
    LRP,
    GuidedBackprop, 
    DeepLift
)
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Subset
import os
import random
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

import torch
import torch.nn as nn
import timm

from data_pipeline.data_set import UniformValidDataloader , UniformTrainDataloader
from data_pipeline.data_aug import MoCoSingleAug
import torch.nn.functional as F
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, cohen_kappa_score
)
import numpy as np




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
        
        return x_channel * spatial_att

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

class EnhancedDRClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        super(EnhancedDRClassifier, self).__init__()
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # moco_state_dict = checkpoint['model_state_dict']
        # config = checkpoint['config']
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0)
        
        # backbone_state_dict = {k.replace('query_encoder.', ''): v for k, v in moco_state_dict.items() if k.startswith('query_encoder.')}
        # self.backbone.load_state_dict(backbone_state_dict)
        
        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

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
        
        # self._initialize_weights()
        
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
        
    def forward(self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None):
        features = self.backbone.forward_features(x)
        attended_features = self.attention(features)
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
            return logits, grade_probs, domain_logits, h, attended_features
        return logits, grade_probs, domain_logits


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



def validate(
        model,
        dataloader
):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

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
                lambda_consistency=0.1,
                lambda_domain=0.0, # Ensure domain loss weight is 0
                ordinal_weight=0.3,
                num_classes=5
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


    # Calculate AUC (Macro average One-vs-Rest)
    try:
        # Ensure all classes are present, otherwise AUC might fail or be misleading
        present_classes = np.unique(all_labels)
        if len(present_classes) == 5 and all_probs.shape[1] == 5:
             auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        elif len(present_classes) > 1 and all_probs.shape[1] == 5:
             logging.warning(f"Only {len(present_classes)}/{5} classes present in validation batch. AUC might be unreliable.")
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
    print(
        f"Validation : , Loss: {avg_loss:.4f}, Acc: {acc:.4f}, "
        f"F1(W): {f1_weighted:.4f}, QWK: {qwk:.4f}"
    )
    print(
        f"Validation - Avg Sensitivity: {avg_sensitivity:.4f}, Avg Specificity: {avg_specificity:.4f}, AUC(Macro-OvR): {auc_macro_ovr:.4f}"
     )
    




os.makedirs("explainability_results", exist_ok=True)

def validate_with_explainability(
        model,
        dataloader,
        num_samples=5,
        save_path="explainability_results"
):
    model.eval()
    
    # For TCAV: we'll need to collect activations
    concept_activations = {
        'hemorrhage': [], 
        'exudate': [], 
        'microaneurysm': [],
        'healthy_retina': [],
        'optic_disc': []
    }
    concept_labels = []
    
    # Generate class activation maps and saliency for random samples
    all_samples = []
    
    # First pass: collect all valid samples
    for batch_data in dataloader:
        if len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            images, labels = batch_data
            
        for i in range(len(images)):
            all_samples.append((images[i].unsqueeze(0), labels[i].item()))
            
        if len(all_samples) >= 50:  # Collect more than we need to choose from
            break
    
    # Randomly select samples
    selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
    
    # Setup for accessing intermediate layers
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks to capture intermediate activations
    model.backbone.stages[-1].register_forward_hook(get_activation('final_conv'))
    
    # For each selected sample, apply explainability methods
    for idx, (image, label) in enumerate(selected_samples):
        image = image.to(device, non_blocking=True)
        label_tensor = torch.tensor([label]).to(device)
        
        # Create a normalized version of the image for visualization
        img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        # Normalize to 0-1 range for visualization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Forward pass
        image.requires_grad = True
        logits, grade_outputs, _ = model(image, alpha=0.0, update_prototypes=False)
        pred_prob = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1).item()
        
        # 1. Gradient-based Saliency Map
        image.grad = None
        model.zero_grad()
        
        # Get gradients with respect to predicted class
        logits[0, pred_label].backward(retain_graph=True)
        gradients = image.grad.squeeze().cpu().numpy()
        
        # Convert gradients to a saliency map
        saliency_map = np.max(np.abs(gradients), axis=0)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        # 2. Grad-CAM
        # Get the final convolutional layer's output
        final_conv_output = activation['final_conv']
        
        # Calculate gradients of the target class score with respect to final conv output
        model.zero_grad()
        class_score = logits[0, pred_label]
        class_score.backward(retain_graph=True)
        
        # Get the gradients from the final convolutional layer
        gradients_for_cam = model.backbone.stages[-1].blocks[-1].grad_fn
        
        # Use activations and gradients to compute class activation map
        with torch.no_grad():
            # Get the gradients of the feature maps
            weights = torch.mean(torch.autograd.grad(class_score, final_conv_output)[0], dim=(2, 3))
            
            # Create a weighted combination of feature maps
            batch_size, num_channels, h, w = final_conv_output.shape
            cam = torch.zeros((h, w), device=device)
            
            for c in range(num_channels):
                cam += weights[0, c] * final_conv_output[0, c, :, :]
            
            # Apply ReLU to focus on features that have a positive influence
            cam = torch.relu(cam)
            
            # Normalize the CAM to 0-1 range
            cam = cam - cam.min()
            if cam.max() > 0:  # Avoid division by zero
                cam = cam / cam.max()
                
            # Upsample the CAM to match image dimensions
            cam = cam.cpu().numpy()
            cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
            
        # 3. Layer-wise Relevance Propagation (LRP)
        # We'll use Captum's LRP implementation
        lrp = LRP(model)
        attribution = lrp.attribute(image, target=pred_label)
        lrp_attr = attribution.squeeze().cpu().detach().numpy()
        lrp_attr = np.sum(np.abs(lrp_attr), axis=0)
        lrp_attr = (lrp_attr - lrp_attr.min()) / (lrp_attr.max() - lrp_attr.min() + 1e-8)
        
        # 4. Attention Flow Visualization
        # Use the attended features from your model to generate a visualization
        _, _, _, _, attended_features = model(image, alpha=0.0, get_attention=True)
        
        # Create an attention map by averaging across channels
        attention_map = torch.mean(attended_features, dim=1).squeeze().detach().cpu().numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        attention_map = cv2.resize(attention_map, (image.shape[3], image.shape[2]))
        
        # 5. TCAV (simplified implementation)
        # For a real TCAV, you'd need concept data and training
        # Here, we'll just collect activations for later TCAV analysis
        # In practice, you'd have pre-identified concept examples
        
        # Get the features before the classifier
        _, _, _, features, _ = model(image, alpha=0.0, get_attention=True)
        features_np = features.cpu().detach().numpy()
        
        # Plot and save all visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f'Original Image\nTrue: {label}, Pred: {pred_label}')
        axes[0, 0].axis('off')
        
        # Grad-CAM
        axes[0, 1].imshow(img_np)
        axes[0, 1].imshow(cam, cmap='jet', alpha=0.5)
        axes[0, 1].set_title('Grad-CAM')
        axes[0, 1].axis('off')
        
        # Saliency Map
        axes[0, 2].imshow(saliency_map, cmap='hot')
        axes[0, 2].set_title('Gradient Saliency')
        axes[0, 2].axis('off')
        
        # LRP
        axes[1, 0].imshow(lrp_attr, cmap='bwr')
        axes[1, 0].set_title('Layer-wise Relevance Propagation')
        axes[1, 0].axis('off')
        
        # Attention Flow
        axes[1, 1].imshow(attention_map, cmap='viridis')
        axes[1, 1].set_title('Attention Flow')
        axes[1, 1].axis('off')
        
        # Combined visualization
        combined = np.zeros_like(img_np)
        combined[:,:,0] = cam  # Red channel for Grad-CAM
        combined[:,:,1] = attention_map  # Green channel for Attention
        combined[:,:,2] = saliency_map  # Blue channel for Saliency
        axes[1, 2].imshow(img_np)
        axes[1, 2].imshow(combined, alpha=0.5)
        axes[1, 2].set_title('Combined Visualization')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/sample_{idx}_grade_{label}_pred_{pred_label}.png")
        plt.close()
        
        # Save features for TCAV
        # In practice, you'd have separate data for each concept
        # This is just a placeholder to demonstrate the concept
        concept_key = random.choice(list(concept_activations.keys()))
        concept_activations[concept_key].append(features_np[0])
        concept_labels.append(concept_key)
    
    # Run standard validation metrics
    standard_metrics = validate(model, dataloader)
    
    # Implement simplified TCAV
    # In a real implementation, you would have examples of each concept
    # and train linear classifiers to separate concept vs. random examples
    if len(concept_activations['hemorrhage']) > 0:
        print("Creating simplified TCAV visualization...")
        perform_simplified_tcav(concept_activations, concept_labels, save_path)
    
    return standard_metrics

def perform_simplified_tcav(concept_activations, concept_labels, save_path):
    """
    A simplified version of TCAV to demonstrate the concept.
    In a real implementation, you would:
    1. Have many examples of each concept
    2. Train linear classifiers to separate concept vs. random examples
    3. Compute directional derivatives to get TCAV scores
    """
    # Convert activations to a format for training a classifier
    all_activations = []
    all_concept_labels = []
    
    # Get available concepts that have data
    available_concepts = [c for c in concept_activations if len(concept_activations[c]) > 0]
    
    if len(available_concepts) < 2:
        print("Not enough concept data for TCAV visualization")
        return
    
    # Flatten and collect activations
    for concept in available_concepts:
        for activation in concept_activations[concept]:
            all_activations.append(activation.flatten())
            all_concept_labels.append(concept)
    
    # Convert to numpy arrays
    X = np.array(all_activations)
    y = np.array(all_concept_labels)
    
    # Perform dimensionality reduction for visualization
    # (For a real implementation, you would use t-SNE or UMAP)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot the concept space
    plt.figure(figsize=(10, 8))
    
    # Use a colormap for different concepts
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_concepts)))
    
    for i, concept in enumerate(available_concepts):
        mask = (y == concept)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    label=concept, color=colors[i], alpha=0.7)
    
    plt.title('TCAV: Concept Space Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/tcav_concept_space.png")
    plt.close()
    
    # Try to train a simple classifier to distinguish between concepts
    # (This is a simplified version of what TCAV does)
    if len(np.unique(y)) > 1:
        # For each DR grade (0-4), see which concepts are most influential
        concept_importance = {}
        
        # Create a binary classifier for each concept vs. others
        for concept in available_concepts:
            binary_y = (y == concept).astype(int)
            
            # Train a simple linear classifier
            clf = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=42)
            try:
                clf.fit(X, binary_y)
                
                # The weights represent the direction in feature space
                # that separates the concept from others
                concept_vector = clf.coef_[0]
                
                # Store the concept vector for visualization
                concept_importance[concept] = np.linalg.norm(concept_vector)
            except:
                print(f"Not enough examples to train classifier for concept: {concept}")
        
        # Create a bar chart of concept importance
        if concept_importance:
            plt.figure(figsize=(12, 6))
            concepts = list(concept_importance.keys())
            importance_values = [concept_importance[c] for c in concepts]
            
            # Sort by importance
            sorted_indices = np.argsort(importance_values)[::-1]
            concepts = [concepts[i] for i in sorted_indices]
            importance_values = [importance_values[i] for i in sorted_indices]
            
            plt.bar(concepts, importance_values, color=colors[:len(concepts)])
            plt.title('TCAV: Concept Importance')
            plt.xlabel('Concepts')
            plt.ylabel('Importance (Vector Norm)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{save_path}/tcav_concept_importance.png")
            plt.close()

# Select a few samples for explainability visualization
def select_samples_for_explanation(dataloader, num_samples=5):
    all_samples = []
    
    # First, collect samples from different classes if possible
    class_samples = {i: [] for i in range(5)}  # For 5 DR grades
    
    for batch_data in dataloader:
        if len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            images, labels = batch_data
            
        for i in range(len(images)):
            label = labels[i].item()
            if len(class_samples[label]) < num_samples:
                class_samples[label].append((images[i], label))
    
    # Try to get at least one sample from each class
    for label in range(5):
        if class_samples[label]:
            all_samples.append(random.choice(class_samples[label]))
    
    # Fill the rest with random samples
    remaining_samples_needed = num_samples - len(all_samples)
    if remaining_samples_needed > 0:
        # Flatten the class_samples dictionary
        flat_samples = []
        for samples in class_samples.values():
            flat_samples.extend(samples)
        
        # Add random samples if we have enough
        if flat_samples:
            additional_samples = random.sample(
                flat_samples, 
                min(remaining_samples_needed, len(flat_samples))
            )
            all_samples.extend(additional_samples)
    
    return all_samples

# Main execution
if __name__ == "__main__":
    # Load your model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedDRClassifier(num_classes=5, freeze_backbone=False).to(device)
    checkpoint = torch.load("checkpoint/no_w/checkpoint_epoch_95.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Setup data loader
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    val_transform = MoCoSingleAug(img_size=256)
    val_loader = UniformTrainDataloader( 
        dataset_names=dataset_names,
        transformation=val_transform,
        batch_size=8,
        num_workers=4,
        sampler=True
    ).get_loader()
    
    # Run validation with explainability
    validate_with_explainability(
        model=model,
        dataloader=val_loader,
        num_samples=5,  
        save_path="explainability_results"
    )
    
    print("Explainability analysis complete. Results saved to 'explainability_results' directory.")