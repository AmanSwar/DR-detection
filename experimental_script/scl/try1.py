"""
CHATGPT o3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import wandb
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
# -------------------------------
# 1. Supervised Contrastive Loss
# -------------------------------
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: Tensor of shape [batch_size, feature_dim]
            labels: Tensor of shape [batch_size]
        Returns:
            loss: A scalar tensor for the loss.
        """
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T)  # [B, B]
        logits = similarity_matrix / self.temperature

        # Mask out self-similarities
        logits_mask = torch.scatter(torch.ones_like(logits),
                                    1,
                                    torch.arange(batch_size).view(-1, 1).to(device),
                                    0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-8)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# -------------------------------
# 2. Augmentation Pipelines
# -------------------------------
def get_train_augmentation():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # tuned crop scale for retinal images
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_augmentation():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# -------------------------------
# 3. Model Definition
# -------------------------------
class ConvNeXtContrastiveModel(nn.Module):
    def __init__(self, num_classes, proj_out_dim=128):
        super(ConvNeXtContrastiveModel, self).__init__()
        # Load pretrained ConvNeXt (using convnext_tiny here)
        self.backbone = models.convnext_tiny(pretrained=True)
        # Remove original classifier so backbone returns a feature vector
        self.backbone.classifier = nn.Identity()  # e.g., 768-dim features

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, proj_out_dim)
        )
        
        # Classification head for cross-entropy
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Two augmented views of shape [B, C, H, W]
        Returns:
            proj_features: [2B, proj_out_dim] for contrastive loss.
            logits: [B, num_classes] computed from first view for classification.
        """
        # Concatenate both views along batch dimension
        x = torch.cat([x1, x2], dim=0)  # [2B, C, H, W]
        features = self.backbone(x)   # [2B, 768]
        features = features.flatten(1)
        proj_features = self.projection_head(features)  # [2B, proj_out_dim]
        logits = self.classifier(features[:x1.size(0)])   # use first view for classification
        return proj_features, logits


# -------------------------------
# 4. GradCAM for Attention Map Visualization
# -------------------------------
class GradCAM:
    """
    A simple GradCAM implementation that registers hooks on the target layer.
    In this example, we register on the last convolutional layer of the backbone.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks on target layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_image, target_class=None):
        """
        Args:
            input_image: Tensor of shape [B, C, H, W]
            target_class: Tensor of target class indices. If None, use argmax of model output.
        Returns:
            List of CAM heatmaps (one per image in batch)
        """
        self.model.zero_grad()
        # Forward pass through backbone only (assumes backbone outputs conv features then pooled features)
        features = self.model.backbone(input_image)  # [B, 768]
        # For CAM, we need a classification output:
        logits = self.model.classifier(features)
        if target_class is None:
            target_class = logits.argmax(dim=1)
            
        cams = []
        for i in range(input_image.size(0)):
            self.model.zero_grad()
            score = logits[i, target_class[i]]
            score.backward(retain_graph=True)
            # gradients & activations from target layer for sample i:
            gradients = self.gradients[i]   # [C, H, W]
            activations = self.activations[i]  # [C, H, W]
            # Global-average-pool gradients over spatial dims
            weights = gradients.mean(dim=(1, 2))  # [C]
            cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0)
            cam = F.relu(cam)
            cam -= cam.min()
            cam /= (cam.max() + 1e-8)
            # Upsample to input image size
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                size=input_image.shape[2:],
                                mode='bilinear',
                                align_corners=False)
            cams.append(cam.squeeze().cpu().numpy())
        return cams


# -------------------------------
# 5. Training and Validation Loops
# -------------------------------
def train_one_epoch(model, train_loader, optimizer, device, epoch, con_weight, train_aug):
    model.train()
    scl_criterion = SupervisedContrastiveLoss(temperature=0.07)
    ce_criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        # Generate two augmented views (convert tensor to PIL and back)
        x1 = torch.stack([
            train_aug(Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)))
            for img in images
        ]).to(device)  # Move x1 to device
        
        x2 = torch.stack([
            train_aug(Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)))
            for img in images
        ]).to(device)
        optimizer.zero_grad()
        proj_features, logits = model(x1, x2)
        # Duplicate labels for contrastive loss (one per view)
        combined_labels = torch.cat([labels, labels], dim=0)
        scl_loss = scl_criterion(proj_features, combined_labels)
        ce_loss = ce_criterion(logits, labels)
        
        # loss = con_weight * scl_loss + (1 - con_weight) * ce_loss
        loss = scl_loss
        loss.backward()
        optimizer.step()
        # print(f"LOSS : {loss} ")
        print(f"SCL LOSS : {scl_loss}   |  CE LOSS : {ce_loss} ")
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, val_loader, device, val_aug):
    model.eval()
    ce_criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            # For validation, use a single (center-cropped) view
            x = torch.stack([
                val_aug(Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)))
                for img in images
            ]).to(device) 
            # To satisfy the model’s forward, pass x as both views.
            _, logits = model(x, x)
            loss = ce_criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1

def log_attention_maps(model, val_loader, device, val_aug, epoch, gradcam):
    """
    Logs attention (GradCAM) maps for a batch from the validation set.
    """
    model.eval()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    x = torch.stack([val_aug(Image.fromarray((img.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)))
                     for img in images])
    cams = gradcam.generate_cam(x)
    
    logged_images = []
    # Convert images for visualization (denormalize)
    images_np = x.cpu().numpy()
    for i, cam in enumerate(cams):
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Denormalize image (assumed ImageNet stats)
        img = images_np[i].transpose(1,2,0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        img = np.uint8(255 * img)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        logged_images.append(wandb.Image(overlay, caption=f"Epoch {epoch}"))
    wandb.log({"Attention Maps": logged_images, "epoch": epoch})


# -------------------------------
# 6. Main Training Script with WandB
# -------------------------------
def main():
    from data_pipeline.data_set import UniformTrainDataloader , UniformValidDataloader
    wandb.init(project="DR_Classification", name="DR_Contrastive_Exp")
    
    # Hyperparameters (tweak as needed)
    num_epochs = 200
    con_weight = 0.5
    lr = 1e-4
    batch_size = 32
    num_classes = 6  # DR grades
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    train_aug = get_train_augmentation()
    val_aug = get_val_augmentation()

    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    trainloader = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_aug,
        batch_size=batch_size,
        num_workers=0,
        sampler=True
    ).get_loader()

    validloader = UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=val_aug,
        batch_size=batch_size,
        num_workers=0,
        sampler=True
    ).get_loader()
    
    
    
    # Initialize model and optimizer
    model = ConvNeXtContrastiveModel(num_classes=num_classes, proj_out_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup GradCAM on the last conv layer of the backbone.
    # (For convnext_tiny, we assume model.backbone.features[-1] is a suitable layer.)
    target_layer = model.backbone.features[-1]
    gradcam = GradCAM(model, target_layer)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(model, trainloader, optimizer, device, epoch, con_weight, train_aug)
        val_loss, val_acc, val_f1 = validate(model, validloader, device, val_aug)
        
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        })
        
        # Every 50 epochs, log attention maps
        if epoch % 50 == 0:
            log_attention_maps(model, validloader, device, val_aug, epoch, gradcam)
    
    wandb.finish()

if __name__ == "__main__":
    main()
