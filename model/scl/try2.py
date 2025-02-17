"""
DEEPSEEK R1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import # wandb
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import convnext_tiny
from torchvision import transforms

# Initialize W&B


# Hyperparameters (Optimized for Fundus Images)
config = {
    "batch_size": 256,  # Large batch for contrastive learning
    "lr": 4e-5,  # Lower LR for fine-tuning
    "temperature": 0.15,  # Higher than default for medical images
    "con_weight": 0.7,  # Emphasize contrastive learning
    "epochs": 300,
    "img_size": 512,  # Higher resolution for retinal details
    "proj_dim": 256,
    "dropout": 0.3,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "scheduler": "cosine",
}

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        batch_size = features.shape[0]

        # Mask for self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(similarity_matrix),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        
        # Mask for positive pairs
        labels = labels.view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float()
        mask = label_mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_probs = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean log-likelihood
        mean_log_prob_pos = (mask * log_probs).sum(1) / (mask.sum(1) + 1e-8)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss


# Modified ConvNeXt with Attention Maps
class FundusConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = convnext_tiny(pretrained=True)
        
        # Feature Extractor
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection Head
        self.projection = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6),
            nn.Linear(768, config["proj_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"])
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6),
            nn.Dropout(config["dropout"]),
            nn.Linear(768, num_classes)
        )
        
        # Grad-CAM Hook
        self.gradients = None
        self.activations = None
        self.features[-2].register_forward_hook(self.forward_hook)
        self.features[-2].register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def get_attention_map(self):
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        return torch.mul(self.activations, weights).sum(dim=1, keepdim=True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        
        return self.projection(features), self.classifier(features)

# Enhanced Training Script
class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(device)
        self.scaler = GradScaler()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"]
        )
        
        self.scl_criterion = SupervisedContrastiveLoss(
            temperature=config["temperature"]
        )
        self.ce_criterion = nn.CrossEntropyLoss()

    def compute_metrics(self, preds, labels):
        return {
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    def log_attention_maps(self, images, epoch):
        self.model.eval()
        with torch.no_grad():
            _, logits = self.model(images[:4])
            cls_pred = logits.argmax(1)
            
            # Compute gradients
            one_hot = torch.zeros_like(logits).scatter_(1, cls_pred.view(-1, 1), 1.0)
            logits.backward(gradient=one_hot)
            
        attn_maps = self.model.get_attention_map()
        # wandb.log({"Attention Maps": [
            # wandb.Image(img, caption=f"Class {pred}\nAttention")
        #     for img, pred in zip(images, cls_pred)
        # ]}, step=epoch)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for images, labels in self.train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast():
                proj_features, logits = self.model(images)
                scl_loss = self.scl_criterion(proj_features, labels)
                ce_loss = self.ce_criterion(logits, labels)
                loss = config["con_weight"]*scl_loss + (1-config["con_weight"])*ce_loss
                
            # Backprop
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), config["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Metrics
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            
        # Logging
        metrics = self.compute_metrics(all_preds, all_labels)
        metrics.update({
            "train_loss": total_loss/len(self.train_loader),
            "lr": self.optimizer.param_groups[0]["lr"]
        })
        # wandb.log(metrics, step=epoch)
        
        self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                proj_features, logits = self.model(images)
                loss = self.ce_criterion(logits, labels)
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
                
        metrics = self.compute_metrics(all_preds, all_labels)
        metrics["val_loss"] = total_loss/len(self.val_loader)
        # wandb.log({f"val_{k}": v for k,v in metrics.items()}, step=epoch)
        
        # Log attention maps every 50 epochs
        if epoch % 50 == 0:
            self.log_attention_maps(images, epoch)

if __name__ == "__main__":
    from data_pipeline.data_set import UniformTrainDataloader, UniformValidDataloader
    # wandb.init(project="DR-Grading", entity="your-entity")
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config["img_size"], scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.GaussianBlur(21, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.372, 0.278, 0.188], std=[0.282, 0.216, 0.153])  # Fundus-specific
        ])

    trainloader = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_transform,
        batch_size=8,
        num_workers=0,
        sampler=True
    ).get_loader()

    validloader = UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=train_transform,
        batch_size=8,
        num_workers=0,
        sampler=True
    ).get_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FundusConvNeXt(num_classes=6)  # 5 DR grades

    trainer = Trainer(model, trainloader, validloader)
    for epoch in range(config["epochs"]):
        trainer.train_epoch(epoch)
        trainer.validate(epoch)