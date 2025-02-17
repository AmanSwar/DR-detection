import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm  
import math
import wandb  # Make sure to install wandb via pip if you haven't already
from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np


IMG_SIZE = 384
#########################################
# 1. Supervised Contrastive Loss Module #
#########################################
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

        similarity_matrix = torch.matmul(features, features.T)  # shape: [B, B]
        logits = similarity_matrix / self.temperature

        logits_mask = torch.scatter(torch.ones_like(logits),
                                    1,
                                    torch.arange(batch_size).view(-1, 1).to(device),
                                    0)

        # Create mask where mask[i, j] = 1 if labels[i]==labels[j], else 0.
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask * logits_mask

        # Compute the denominator: sum over all examples except self
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # For each sample, compute the mean log-likelihood over its positive examples
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-8)

        # Final loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss
        

#######################################
# 2. Swin Transformer Encoder Module  #
#######################################
class SwinTransformerEncoder(nn.Module):
    def __init__(self, 
                 model_name="swin_tiny_patch4_window7_224", 
                 pretrained=False,
                 img_size=IMG_SIZE,         # Updated image size
                 patch_size=4          # Increased patch size to reduce token count
                ):
        super(SwinTransformerEncoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(0)
        
        # Update patch embedding parameters for higher resolution
        self.model.patch_embed.img_size = (img_size , img_size)
        self.model.patch_embed.patch_size = patch_size
        
        new_resolution = img_size // patch_size  # e.g., 512 / 8 = 64
        print(f"New feature map resolution: {new_resolution} x {new_resolution}")
        
        # Optionally, adjust the window size (make sure it divides new_resolution)
        # Here we set it to 8, but you can change it based on your design.
        
        # (Optional) If using pretrained weights, you may need to interpolate 
        # positional embeddings to match the new grid. This example leaves that as a placeholder.
        if hasattr(self.model, "absolute_pos_embed") and self.model.absolute_pos_embed is not None:
            # A simple placeholder for positional embedding interpolation:
            pos_embed = self.model.absolute_pos_embed
            # Assuming pos_embed shape is [1, num_tokens, embed_dim] or [1, C, H, W]
            # Implement interpolation here if necessary.
            pass

        self.num_features = self.model.num_features
        print(f"Encoder num_features: {self.num_features}")

    def forward(self, x, return_feature_map=False):
        # Get the features from the backbone
        x = self.model.forward_features(x)
        
        if x.ndim == 4:
            B, H, W, C = x.shape  # Original shape from timm: [B, H, W, C]
            # Permute to [B, C, H, W] for easier visualization/processing
            feature_map = x.permute(0, 3, 1, 2)
            if return_feature_map:
                return feature_map
            # Flatten spatial dimensions and average pool
            x = x.reshape(B, H * W, C)  # [B, H*W, C]
            x = x.mean(dim=1)           # [B, C]
        elif x.ndim == 3:
            # For 3D output [B, L, C]
            x = x.mean(dim=1)  # [B, C]
        return x

#####################################
# 3. Retinopathy Model Definition   #
#####################################
class RetinopathyModel(nn.Module):
    def __init__(self, 
                 num_classes=6, 
                 projection_dim=128, 
                 swin_model_name="swin_tiny_patch4_window7_224",
                 img_size=IMG_SIZE,         # Pass updated image size to encoder
                 patch_size=8          # Pass updated patch size to encoder
                ):
        super(RetinopathyModel, self).__init__()
        self.encoder = SwinTransformerEncoder(model_name=swin_model_name,
                                                pretrained=True,
                                                img_size=img_size,
                                                patch_size=patch_size)
        encoder_out_features = self.encoder.num_features
        
        print(f"Initializing projection head with input dim: {encoder_out_features}")
        
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_out_features, encoder_out_features),
            nn.ReLU(),
            nn.Linear(encoder_out_features, projection_dim)
        )
        
        self.classifier = nn.Linear(encoder_out_features, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)  # Default: returns pooled features [B, C]
        proj_features = self.projection_head(features)
        logits = self.classifier(features)
        return features, proj_features, logits

#####################################################
# 4. Training and Validation Functions with Logging #
#####################################################
def train_epoch(model, optimizer, dataloader, scl_criterion, ce_criterion, epoch, con_weight=0.7, log_interval=10):
    device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    total_scl_loss = 0.0
    total_ce_loss = 0.0

    # Wrap the dataloader with tqdm for a progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}", unit="batch")
    for batch_idx, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        try:
            features, proj_features, logits = model(images)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise e
        
        # Compute losses
        scl_loss = scl_criterion(proj_features, labels)
        ce_loss = ce_criterion(logits, labels)
        loss = con_weight * scl_loss + (1 - con_weight) * ce_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_scl_loss += scl_loss.item()
        total_ce_loss += ce_loss.item()
        
        # Log batch metrics to wandb every log_interval batches
        if batch_idx % log_interval == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_scl_loss": scl_loss.item(),
                "batch_ce_loss": ce_loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })
            if batch_idx == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: SCL Loss = {scl_loss.item():.4f}, "
                      f"CE Loss = {ce_loss.item():.4f}, Total Loss = {loss.item():.4f}")
        
        # Update tqdm description with the current batch losses
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "scl_loss": f"{scl_loss.item():.4f}",
            "ce_loss": f"{ce_loss.item():.4f}"
        })

    avg_loss = total_loss / len(dataloader)
    avg_scl_loss = total_scl_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    
    return avg_loss, avg_scl_loss, avg_ce_loss

def validate_epoch(model, dataloader, ce_criterion, epoch):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            _, _, logits = model(images)
            loss = ce_criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    wandb.log({
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "epoch": epoch
    })
    print(f"Validation - Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy*100:.2f}%")
    return avg_loss, accuracy

def save_checkpoint(epoch, loss, model, optim, scheduler):
    save_dir = "checkpoints/custom_scl"
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()

    # Save regular checkpoint
    torch.save(checkpoint, save_dir / f"checkpoint_ep_{epoch}.pt")
    
def log_attention_maps(model, valid_loader, epoch, device):
    """
    Logs attention maps / heatmaps from the encoder's feature maps.
    Here we use the feature maps from the encoder (before pooling), average across channels,
    and log them as heatmaps.
    """
    model.eval()
    try:
        images, labels = next(iter(valid_loader))
    except StopIteration:
        print("Validation loader is empty!")
        return

    images = images.to(device)
    # Get the feature maps from the encoder (shape: [B, C, H, W])
    feature_maps = model.encoder(images, return_feature_map=True)
    # Average across the channel dimension to obtain a 2D map per image
    attn_maps = feature_maps.mean(dim=1, keepdim=True)  # shape: [B, 1, H, W]
    # Normalize each attention map for visualization
    attn_maps_min = attn_maps.view(attn_maps.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    attn_maps_max = attn_maps.view(attn_maps.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    attn_maps = (attn_maps - attn_maps_min) / (attn_maps_max - attn_maps_min + 1e-8)
    
    # Prepare a list of heatmap images (log at most 4 samples)
    heatmaps = []
    num_samples = min(4, images.size(0))
    for i in range(num_samples):
        # Convert the attention map to a numpy array
        heatmap = attn_maps[i, 0].cpu().numpy()
        # Optionally, you can apply a colormap here using matplotlib if desired.
        heatmaps.append(wandb.Image(heatmap, caption=f"Attention map for sample {i}"))
    
    wandb.log({"Attention_Maps": heatmaps, "epoch": epoch})
    print(f"Logged attention maps for epoch {epoch+1}.")

def train(n_epoch, train_loader, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, optimizer, and loss functions (created only once)
    # Here, we use 512 as the input image size.
    model = RetinopathyModel(num_classes=6, projection_dim=128, img_size=IMG_SIZE, patch_size=4).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scl_criterion = SupervisedContrastiveLoss(temperature=0.07)
    ce_criterion = nn.CrossEntropyLoss()
    
    # Initialize wandb and log configuration details
    # wandb.init(project="retinopathy_scl_training", config={
    #     "learning_rate": 1e-4,
    #     "batch_size": train_loader.batch_size,
    #     "epochs": n_epoch,
    #     "con_weight": 0.7,
    #     "img_size": IMG_SIZE,
    #     "patch_size": 8
    # })
    # wandb.watch(model, log="all")

    try:
        for epoch in range(n_epoch):
            print(f"Epoch {epoch+1}/{n_epoch}")
            avg_loss, avg_scl_loss, avg_ce_loss = train_epoch(
                model, optimizer, train_loader,
                scl_criterion, ce_criterion, epoch,
                con_weight=0.7, log_interval=10
            )
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, SCL Loss = {avg_scl_loss:.4f}, CE Loss = {avg_ce_loss:.4f}")
            
            # Run validation at the end of each epoch
            val_loss, val_acc = validate_epoch(model, valid_loader, ce_criterion, epoch)
            
            # Every 50 epochs, log attention maps/heatmaps for xAI
            if (epoch + 1) % 50 == 0:
                log_attention_maps(model, valid_loader, epoch, device)
            
            # Log epoch-level metrics to wandb
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch_scl_loss": avg_scl_loss,
                "epoch_ce_loss": avg_ce_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch
            })
    except KeyboardInterrupt:
        save_checkpoint(epoch=epoch, loss=avg_loss, model=model, optim=optimizer, scheduler=None)

#####################################################
# 5. Main: Data Setup and Training Entry Point     #
#####################################################
if __name__ == "__main__":
    from data_pipeline.data_set import UniformTrainDataloader, UniformValidDataloader
    from data_pipeline.data_aug import scl_trans

    device = torch.device("cuda")
    dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
    trainloader = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=scl_trans,
        batch_size=8,
        num_workers=0,
        sampler=True
    ).get_loader()

    validloader = UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=scl_trans,
        batch_size=8,
        num_workers=0,
        sampler=True
    ).get_loader()
    
    n_epoch = 300  

    train(n_epoch, trainloader, validloader)
