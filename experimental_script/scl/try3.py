"""
Gemini thinking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# --- Import Grad-CAM related libraries ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


# Define the Supervised Contrastive Loss (same as before)
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        # ... (loss function code - same as provided before) ...
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, p=2, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        logits = similarity_matrix / self.temperature

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
        loss = loss.mean()
        return loss


# Define ConvNeXt Model (same as before)
class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes, projection_dim=128, pretrained=True):
        super(ConvNeXtModel, self).__init__()
        self.convnext = models.convnext_tiny(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.convnext.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.projection_head = nn.Sequential(
            nn.Linear(self.convnext.classifier[0].in_features, self.convnext.classifier[0].in_features),
            nn.ReLU(),
            nn.Linear(self.convnext.classifier[0].in_features, projection_dim)
        )
        self.classification_head = nn.Linear(self.convnext.classifier[0].in_features, num_classes)
        nn.init.xavier_uniform_(self.classification_head.weight)
        nn.init.zeros_(self.classification_head.bias)
        self.target_layer = self.convnext.features[-1][0].norm1 # Target layer for Grad-CAM


    def forward(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        proj_features = self.projection_head(features)
        logits = self.classification_head(features)
        return proj_features, logits

    def get_cam_target_layer(self):
        return [self.target_layer]


# Define Data Augmentations (Tweaked for Retinal Images)
def get_augmentations(img_size=224, stronger=True): # Added stronger augmentation option
    if stronger:
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)), # Keep random crop
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), # Added vertical flip - retinal images can be flipped
            transforms.RandomRotation(degrees=45), # Added rotation
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2) # Increased jitter strength
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # Weaker augmentation for validation/less aggressive training
        augmentation = transforms.Compose([
            transforms.Resize(img_size), # Or CenterCrop if you prefer
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return augmentation



def calculate_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    return accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

def calculate_f1_weighted(logits, labels):
    _, predicted = torch.max(logits, 1)
    return f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted') # Use weighted F1 for class imbalance



def train_one_epoch(model, dataloader, optimizer, scl_criterion, ce_criterion, con_weight, device, epoch, wandb_log=True):
    model.train()
    total_loss = 0.0
    total_scl_loss = 0.0
    total_ce_loss = 0.0
    total_accuracy = 0.0
    total_f1_score = 0.0

    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)

        augmentation, _ = get_augmentations(stronger=True) # Stronger augmentation for training
        data_aug1 = torch.stack([augmentation(img) for img in data])
        data_aug2 = torch.stack([augmentation(img) for img in data]) # Or augmentation_prime

        proj_features1, logits1 = model(data_aug1)
        proj_features2, logits2 = model(data_aug2)

        proj_features = torch.cat([proj_features1, proj_features2], dim=0)
        labels_scl = torch.cat([labels, labels], dim=0)

        scl_loss = scl_criterion(proj_features, labels_scl)
        ce_loss = ce_criterion(logits1, labels)  # Using logits1 for CE loss

        loss = con_weight * scl_loss + (1 - con_weight) * ce_loss
        print(f"SCL LOSS : {scl_loss}       |    CE LOSS : {ce_loss}         |   LOSS : {loss} ")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_accuracy = calculate_accuracy(logits1, labels)
        batch_f1_score = calculate_f1_weighted(logits1, labels)

        total_loss += loss.item()
        total_scl_loss += scl_loss.item()
        total_ce_loss += ce_loss.item()
        total_accuracy += batch_accuracy
        total_f1_score += batch_f1_score

        if batch_idx % 10 == 0 and wandb_log:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, SCL: {scl_loss.item():.4f}, CE: {ce_loss.item():.4f}, Acc: {batch_accuracy:.4f}, F1: {batch_f1_score:.4f}')


    avg_loss = total_loss / len(dataloader)
    avg_scl_loss = total_scl_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    avg_f1_score = total_f1_score / len(dataloader)

    if wandb_log:
        wandb.log({
            "train_loss": avg_loss,
            "train_scl_loss": avg_scl_loss,
            "train_ce_loss": avg_ce_loss,
            "train_accuracy": avg_accuracy,
            "train_f1_score": avg_f1_score,
            "epoch": epoch
        })
    print(f'Epoch {epoch} Train - Avg Loss: {avg_loss:.4f}, SCL: {avg_scl_loss:.4f}, CE: {avg_ce_loss:.4f}, Acc: {avg_accuracy:.4f}, F1: {avg_f1_score:.4f}')
    return avg_loss, avg_accuracy, avg_f1_score


def validate_one_epoch(model, dataloader, ce_criterion, device, epoch, wandb_log=True, generate_heatmaps=False, heatmap_epoch_interval=50):
    model.eval()
    total_ce_loss = 0.0
    total_accuracy = 0.0
    total_f1_score = 0.0

    heatmap_images_logged = False # Flag to ensure heatmaps are logged only once per heatmap interval epoch

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)

            augmentation = get_augmentations(stronger=False) # Weaker/no augmentation for validation
            data_val = torch.stack([augmentation(img) for img in data])

            _, logits = model(data_val)
            ce_loss = ce_criterion(logits, labels)

            batch_accuracy = calculate_accuracy(logits, labels)
            batch_f1_score = calculate_f1_weighted(logits, labels)

            total_ce_loss += ce_loss.item()
            total_accuracy += batch_accuracy
            total_f1_score += batch_f1_score


            if generate_heatmaps and epoch % heatmap_epoch_interval == 0 and not heatmap_images_logged and batch_idx < 5: # Log heatmap for first 5 batches of validation set
                heatmap_images_logged = True # Set the flag to True after logging heatmaps

                for idx in range(data_val.shape[0]): # Create heatmaps for each image in the batch
                    input_tensor = data_val[idx].unsqueeze(0) # Prepare input for Grad-CAM
                    label = labels[idx].item() # Get the true label

                    # --- Generate Heatmap using Grad-CAM ---
                    target_layers = model.get_cam_target_layer()
                    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device=='cuda')
                    grayscale_cam = cam(input_tensor=input_tensor, targets=[torch.tensor(label)]) # Target is the true label
                    rgb_img = np.float32(data_val[idx].cpu().numpy().transpose(1, 2, 0))
                    rgb_img = rgb_img / np.max(rgb_img) # Normalize to 0-1 if it's not already
                    visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
                    heatmap_image = wandb.Image(visualization, caption=f"Heatmap - Val Img {batch_idx*dataloader.batch_size + idx}, Label: {label}")
                    if wandb_log:
                        wandb.log({"val_heatmaps": heatmap_image, "epoch": epoch})


    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    avg_f1_score = total_f1_score / len(dataloader)


    if wandb_log:
        wandb.log({
            "val_ce_loss": avg_ce_loss,
            "val_accuracy": avg_accuracy,
            "val_f1_score": avg_f1_score,
            "epoch": epoch
        })

    print(f'Epoch {epoch} Validation - Avg CE Loss: {avg_ce_loss:.4f}, Val Acc: {avg_accuracy:.4f}, Val F1: {avg_f1_score:.4f}')
    return avg_ce_loss, avg_accuracy, avg_f1_score



# --- Main Training Script ---
if __name__ == '__main__':
    from data_pipeline.data_set import UniformTrainDataloader , UniformValidDataloader
    # --- Hyperparameters (Tweaked for Retinal Images & Large Dataset) ---
    config = dict(
        num_classes = 6, # Example: DR grades 0-4 (adjust based on your classes)
        projection_dim = 128, # Keep projection dim reasonable
        learning_rate = 5e-5,    # Lower learning rate - large dataset, potentially complex task
        con_weight = 0.5,       # Balance SCL and CE loss
        num_epochs = 100,       # Train for more epochs - large dataset
        batch_size = 32,        # Adjust batch size based on GPU memory
        temperature = 0.07,     # SCL temperature parameter
        weight_decay = 1e-5,    # Added weight decay for regularization
        stronger_augmentation = True, # Use stronger augmentation for training
        heatmap_epoch_interval = 50 # Log heatmaps every 50 epochs
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # --- Wandb Initialization ---
    wandb.init(project="diabetic-retinopathy-classifier", config=config, job_type="training")
    wandb_config = wandb.config # Access config within wandb environment

    # --- Model, Loss, Optimizer ---
    model = ConvNeXtModel(num_classes=wandb_config.num_classes, projection_dim=wandb_config.projection_dim).to(device)
    scl_criterion = SupervisedContrastiveLoss(temperature=wandb_config.temperature)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=wandb_config.learning_rate, weight_decay=wandb_config.weight_decay)


    train_aug = get_augmentations(stronger=True)
    valid_aug = get_augmentations(stronger=True)
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    trainloader = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_aug,
        batch_size=config["batch_size"],
        num_workers=0,
        sampler=True
    ).get_loader()

    validloader = UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=valid_aug,
        batch_size=config["batch_size"],
        num_workers=0,
        sampler=True
    ).get_loader()
    
    


    # --- Training and Validation Loop ---
    best_val_f1 = 0.0 # Track best validation F1 score
    for epoch in range(wandb_config.num_epochs):
        train_loss, train_accuracy, train_f1_score = train_one_epoch(model, trainloader, optimizer, scl_criterion, ce_criterion, wandb_config.con_weight, device, epoch, wandb_log=True)
        val_ce_loss, val_accuracy, val_f1_score = validate_one_epoch(model, validloader, ce_criterion, device, epoch, wandb_log=True, generate_heatmaps=True, heatmap_epoch_interval=wandb_config.heatmap_epoch_interval)


        # --- Save best model based on validation F1 score ---
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            if wandb.run is not None:
                best_model_path = f"best_model_epoch_{epoch}_f1_{val_f1_score:.4f}.pth"
                torch.save(model.state_dict(), best_model_path)
                wandb.save(best_model_path) # Save to wandb artifacts

    if wandb.run is not None:
        wandb.finish()
    print("Training finished!")