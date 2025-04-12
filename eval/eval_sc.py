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


if __name__ == "__main__":
    
    device = torch.device("cuda")
    model = EnhancedDRClassifier(num_classes=5 , freeze_backbone=False).to(device=device)
    # dataset_name = ["aptos" , "idirid" , "ddr" , "messdr" , "sustech"]
    # dataset_name = ["sustech"]
    # ds_name = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    ds_name = ["ddr"]
    from torchvision import transforms
    # val_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    val_transform = MoCoSingleAug(img_size=256)
    val_loader = UniformValidDataloader(
        dataset_names=ds_name,
        transformation=val_transform,
        batch_size=32,
        num_workers=3
    ).get_loader()

    

    # val_loader = UniTestLoader(dataset_name=dataset_name , transforms=val_transform , batch_size=32 , num_worker=3).get_loader()


    
    # chckpt_dir = "good_chkpt/fine_3_local"
    # for pth in os.listdir(chckpt_dir):
    #     checkpoint_path = os.path.join(chckpt_dir , pth)
    #     checkpoint = torch.load(checkpoint_path , map_location=device , weights_only=False)
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     print("\n")
    #     print(f"Running validation on :  {pth}")
    #     validate(model=model , dataloader=val_loader , device=device , epoch=1 , num_epochs=1 , wandb_run=None)

    checkpt_dir = "good_chkpt/fine_3_local/best_clinical_checkpoint.pth"
    checkpoint = torch.load(checkpt_dir , map_location=device , weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    from tqdm import tqdm
    for i in tqdm(range(5)):
        val_loader = UniTestLoader(
            dataset_name="aptos",
            transforms=val_transform,
            batch_size=32,
            num_worker=34
        ).get_loader()
        validate(model=model , dataloader=val_loader , device=device , epoch=1 , num_epochs=1 , wandb_run=None)
    