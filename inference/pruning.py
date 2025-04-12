import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import copy
from tqdm import tqdm
import timm

import torch
import torch.nn as nn
import timm

from data_pipeline.data_set import UniformValidDataloader , UniformTrainDataloader
from data_pipeline.data_eval import UniTestLoader
from data_pipeline.data_aug import MoCoSingleAug
import torch.nn.functional as F
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, cohen_kappa_score
)
import numpy as np
import os
from tqdm import tqdm


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
    
    main_criterion = nn.CrossEntropyLoss()
    main_loss = main_criterion(outputs, labels)
    loss = main_loss
    consistency_loss_val = 0.0
    domain_loss_val = 0.0


    # Grade Consistency Loss
    if grade_outputs is not None and lambda_consistency > 0:
        grade_logits, ordinal_thresholds = grade_outputs
        batch_size = labels.size(0)

        # Ensure labels are within valid range
        labels = labels.clamp(0, num_classes - 1)

        # Vectorized cumulative targets
        indices = torch.arange(num_classes, device=labels.device).unsqueeze(0).expand(batch_size, num_classes)
        label_expanded = labels.unsqueeze(1).expand(batch_size, num_classes)
        targets_cumulative = (indices <= label_expanded).float()

        # Binary Cross Entropy for cumulative probabilities
        consistency_loss_bce = F.binary_cross_entropy_with_logits(grade_logits, targets_cumulative, reduction='mean')
        consistency_loss = consistency_loss_bce

        # Ordinal Threshold Loss
        if ordinal_thresholds is not None and ordinal_weight > 0:
            k_indices = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0).expand(batch_size, num_classes - 1)
            label_expanded = labels.unsqueeze(1).expand(batch_size, num_classes - 1)
            ordinal_targets = (label_expanded > k_indices).float()
            ordinal_loss_bce = F.binary_cross_entropy_with_logits(ordinal_thresholds, ordinal_targets, reduction='mean')
            consistency_loss = (1.0 - ordinal_weight) * consistency_loss_bce + ordinal_weight * ordinal_loss_bce

        loss += lambda_consistency * consistency_loss
        consistency_loss_val = consistency_loss.item()

    # Domain Adversarial Loss
    if domain_logits is not None and domain_labels is not None and lambda_domain > 0:
        domain_criterion = nn.CrossEntropyLoss()
        domain_loss = domain_criterion(domain_logits, domain_labels)
        loss += lambda_domain * domain_loss
        domain_loss_val = domain_loss.item()

    return loss  # Optionally return loss components for logging





def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_flops(model, input_size=(1, 3, 256, 256), device="cuda"):
    """Estimate FLOPs using a dummy forward pass"""
    from fvcore.nn import FlopCountAnalysis
    
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    flops = FlopCountAnalysis(copy.deepcopy(model).to(device), dummy_input)
    return flops.total()

def get_layer_info(model):
    """Extract layer name, type, and parameter count"""
    layer_info = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            layer_info.append({
                'name': name,
                'type': module.__class__.__name__,
                'params': params
            })
    return layer_info

def apply_channel_pruning(model, pruning_rates, input_size=(1, 3, 256, 256), device="cuda"):
    """
    Apply channel pruning to convolutional layers in the model
    
    Args:
        model: The PyTorch model to prune
        pruning_rates: Dict mapping layer names to pruning rates (0.0-1.0)
        input_size: Input tensor shape for measuring FLOPs
        device: Device to use for pruning
    
    Returns:
        pruned_model: Model with pruned channels
    """
    model.eval()
    model = copy.deepcopy(model)
    model = model.to(device)
    
    # Get baseline metrics
    original_params = count_parameters(model)
    original_flops = measure_flops(model, input_size, device)
    
    print(f"Original model: {original_params:,} parameters, {original_flops/1e9:.2f} GFLOPs")
    
    # Extract importance scores for each layer's channels
    importance_scores = {}
    
    # Forward hook to collect feature map statistics
    activation_stats = {}
    
    def forward_hook(name):
        def hook(module, input, output):
            # For Conv2d: shape is [B, C, H, W]
            if isinstance(module, nn.Conv2d):
                # Calculate L1 norm across batch, spatial dimensions as importance
                importance = output.abs().mean(dim=(0, 2, 3)).detach().cpu()
            # For Linear: shape is [B, C]
            elif isinstance(module, nn.Linear):
                importance = output.abs().mean(dim=0).detach().cpu()
            
            activation_stats[name] = importance
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name in pruning_rates:
            hook = module.register_forward_hook(forward_hook(name))
            hooks.append(hook)
    
    # Run forward pass to collect activation statistics
    dummy_input = torch.randn(input_size).to(device)
    model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Perform pruning based on activation statistics
    pruned_channels = {}
    
    # First pass: identify channels to prune
    for name, module in model.named_modules():
        if name in pruning_rates and name in activation_stats:
            rate = pruning_rates[name]
            if rate <= 0:
                continue
                
            importance = activation_stats[name]
            num_channels = len(importance)
            num_to_prune = int(num_channels * rate)
            
            if num_to_prune == 0:
                continue
                
            # Get indices of least important channels
            _, indices = torch.topk(importance, k=num_channels-num_to_prune, largest=True)
            mask = torch.ones(num_channels)
            mask[indices] = 0
            pruned_indices = mask.nonzero().view(-1)
            
            pruned_channels[name] = pruned_indices.tolist()
    
    # Second pass: actually prune channels
    # Note: This is a simplified version that works mainly for basic architectures
    # More complex models with skip connections require careful handling
    for name, module in model.named_modules():
        if name in pruned_channels:
            if isinstance(module, nn.Conv2d):
                # Prune output channels
                indices = pruned_channels[name]
                prune.ln_structured(module, name="weight", amount=0, n=1, dim=0)
                mask = module.weight_mask.clone()
                mask[indices, :, :, :] = 0
                module.weight_mask = mask
                
                # Also prune corresponding bias if present
                if module.bias is not None:
                    prune.ln_structured(module, name="bias", amount=0, n=1, dim=0)
                    mask = module.bias_mask.clone()
                    mask[indices] = 0
                    module.bias_mask = mask
            
            elif isinstance(module, nn.Linear):
                # Prune output features
                indices = pruned_channels[name]
                prune.ln_structured(module, name="weight", amount=0, n=1, dim=0)
                mask = module.weight_mask.clone()
                mask[indices, :] = 0
                module.weight_mask = mask
                
                # Also prune corresponding bias if present
                if module.bias is not None:
                    prune.ln_structured(module, name="bias", amount=0, n=1, dim=0)
                    mask = module.bias_mask.clone()
                    mask[indices] = 0
                    module.bias_mask = mask
    
    # Apply pruning permanently
    for name, module in model.named_modules():
        if name in pruned_channels:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.remove(module, "weight")
                if module.bias is not None:
                    prune.remove(module, "bias")
    
    # Get pruned metrics
    pruned_params = count_parameters(model)
    pruned_flops = measure_flops(model, input_size, device)
    
    print(f"Pruned model: {pruned_params:,} parameters ({pruned_params/original_params*100:.2f}%), "
          f"{pruned_flops/1e9:.2f} GFLOPs ({pruned_flops/original_flops*100:.2f}%)")
    
    return model


def optimize_model(model, input_size=(1, 3, 256, 256), device="cuda", 
                  prune_backbone=True, prune_attention=True, prune_classifiers=True):
    """Complete model optimization pipeline"""
    # Analyze model
    print("Analyzing model architecture...")
    layer_info = get_layer_info(model)
    
    # Sort layers by parameter count to identify the largest contributors
    layer_info = sorted(layer_info, key=lambda x: x['params'], reverse=True)
    
    for layer in layer_info[:10]:  # Show top 10 layers by parameter count
        print(f"{layer['name']}: {layer['type']} - {layer['params']:,} parameters")
    
    # Define pruning rates for each layer
    pruning_rates = {}
    
    # Set different pruning rates based on layer importance
    # Higher pruning rates for later layers, lower for early feature extraction
    for layer in layer_info:
        name = layer['name']
        
        # Skip pruning layers we want to preserve
        if not prune_backbone and 'backbone' in name:
            continue
        if not prune_attention and 'attention' in name:
            continue
        if not prune_classifiers and ('classifier' in name or 'grade_head' in name or 'domain_classifier' in name):
            continue
            
        # Customize pruning rates by layer location
        if 'backbone' in name:
            if 'downsample' in name:  # Be careful with downsampling layers
                pruning_rates[name] = 0.2
            else:
                # Prune early layers less aggressively
                depth = name.count('.')
                pruning_rates[name] = min(0.5, 0.1 + depth * 0.05)
                
        elif 'attention' in name:
            pruning_rates[name] = 0.3  # Moderate pruning for attention
            
        elif 'classifier' in name or 'grade_head' in name:
            # More aggressive pruning for classification heads
            if 'classifier.0' in name or 'grade_predictor.0' in name:  # First FC layer
                pruning_rates[name] = 0.6
            elif 'classifier.4' in name or 'grade_predictor.4' in name:  # Last FC layer
                pruning_rates[name] = 0.3  # Be more careful with final classification
            else:
                pruning_rates[name] = 0.5
                
        elif 'domain_classifier' in name:
            pruning_rates[name] = 0.7  # Most aggressive for domain classifier
            
        else:
            pruning_rates[name] = 0.3  # Default pruning rate
    
    # Apply pruning
    print("\nApplying structural pruning...")
    pruned_model = apply_channel_pruning(model, pruning_rates, input_size, device)
    
    return pruned_model


def validate(model, dataloader, device, epoch, num_epochs, wandb_run=None,
             lambda_consistency=0.1, ordinal_weight=0.3, num_classes=5):
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = [] # Store probabilities for AUC calculation
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad(): # Disable gradient calculations
        for _ , batch_data in tqdm(enumerate(dataloader)):
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
            # loss = OrdinalDomainLoss(
            #     logits, labels,
            #     grade_outputs=grade_outputs,
            #     domain_logits=None, domain_labels=None, # No domain loss in validation
            #     lambda_consistency=lambda_consistency,
            #     lambda_domain=0.0, # Ensure domain loss weight is 0
            #     ordinal_weight=ordinal_weight,
            #     num_classes=num_classes
            # )

            loss = loss_fn(logits, labels)

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
    print(
        f"Validation - Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, "
        f"F1(W): {f1_weighted:.4f}, QWK: {qwk:.4f}"
    )
    print(
        f"Validation - Avg Sensitivity: {avg_sensitivity:.4f}, Avg Specificity: {avg_specificity:.4f}, AUC(Macro-OvR): {auc_macro_ovr:.4f}"
     )
    


def prune_and_finetune(model, train_loader, val_loader, device="cuda", 
                      epochs=5, lr=1e-4, lambda_consistency=0.1, lambda_domain=0.05):
    """Prune model and then finetune to recover accuracy"""
    from torch.optim import Adam
    
    # Step 1: Prune the model
    pruned_model = optimize_model(model, device=device)
    
    # Step 2: Finetune the pruned model
    print("\nFinetuning pruned model...")
    pruned_model.train()
    optimizer = Adam(filter(lambda p: p.requires_grad, pruned_model.parameters()), lr=lr)
    
    for epoch in range(epochs):
        # Training loop
        pruned_model.train()
        train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader)):
            if len(batch_data) == 3:
                images, labels, domain_labels = batch_data
            else:
                images, labels = batch_data
                domain_labels = None
                
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if domain_labels is not None:
                domain_labels = domain_labels.to(device, non_blocking=True)
            
            # Alpha schedule (from 0 to 1) for gradient reversal layer
            p = min(1.0, (batch_idx + epoch * len(train_loader)) / (epochs * len(train_loader)))
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            
            optimizer.zero_grad()
            
            # Forward pass with alpha for domain adaptation
            logits, grade_outputs, domain_logits = pruned_model(
                images, alpha=alpha if domain_labels is not None else 0.0
            )
            
            # Calculate loss
            loss = OrdinalDomainLoss(
                logits, labels,
                grade_outputs=grade_outputs,
                domain_logits=domain_logits, domain_labels=domain_labels,
                lambda_consistency=lambda_consistency,
                lambda_domain=lambda_domain
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        validate(pruned_model, val_loader, device, epoch, epochs)
    
    return pruned_model


# Example usage
if __name__ == "__main__":
    from data_pipeline.data_set import UniformValidDataloader
    from data_pipeline.data_eval import UniTestLoader
    from data_pipeline.data_aug import MoCoSingleAug
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create and load model
    model = EnhancedDRClassifier(num_classes=5, freeze_backbone=False).to(device)
    checkpoint = torch.load("good_chkpt/fine_3_local/best_clinical_checkpoint.pth", 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    ds_name = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    val_transform = MoCoSingleAug(img_size=256)
    val_loader = UniTestLoader(
        dataset_name=ds_name,
        transforms=val_transform,
        batch_size=32,
        num_worker=3
    ).get_loader()
    
    # Setup training data for fine-tuning (optional)
    train_transform = MoCoSingleAug(img_size=256)
    train_loader = UniformTrainDataloader(
        dataset_names=ds_name,
        transformation=train_transform,
        batch_size=32,
        num_workers=3
    ).get_loader()
    
    # Just prune model without fine-tuning
    pruned_model = optimize_model(model, device=device)
    
    # Validate the pruned model
    validate(pruned_model, val_loader, device, 0, 1)
    
    # Save the pruned model
    torch.save({
        "model_state_dict": pruned_model.state_dict(),
    }, "pruned_model_checkpoint.pth")
    
    # Optional: Fine-tune and save
    # pruned_finetuned_model = prune_and_finetune(model, train_loader, val_loader, device=device)
    # torch.save({
    #     "model_state_dict": pruned_finetuned_model.state_dict(),
    # }, "pruned_finetuned_model_checkpoint.pth")