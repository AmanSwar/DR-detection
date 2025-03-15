import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import wandb

# Assuming MoCoV3Model is defined in your original script
from your_moco_script import MoCoV3Model  # Replace with actual import path

# Configuration
config = {
    "base_model": "convnext_small",  # Match your MoCo script
    "projection_dim": 256,
    "hidden_dim": 1024,
    "queue_size": 4096,
    "momentum": 0.99,
    "lr": 1e-4,  # Fine-tuning learning rate
    "num_epochs": 50,
    "checkpoint_path": "model/new/chckpt/moco/checkpoint_epoch_90.pth",
    "num_classes": 5  # Diabetic retinopathy grades
}

# Set up device and logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("model/new/chckpt/moco/finetuned", exist_ok=True)

# Initialize Weights & Biases for logging
wandb_run = wandb.init(project="MoCoV3-DR-Finetune", config=config)

# Load the pretrained MoCo model
checkpoint = torch.load(config["checkpoint_path"], map_location=device)
model = MoCoV3Model(
    base_model=config["base_model"],
    projection_dim=config["projection_dim"],
    hidden_dim=config["hidden_dim"],
    queue_size=config["queue_size"],
    momentum=config["momentum"],
    pretrained=False  # Load from checkpoint instead
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Extract the query_encoder (ConvNeXt backbone)
encoder = model.query_encoder

# Define a simple classification head
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Attach classification head
embed_dim = encoder.num_features  # Output dimension of ConvNeXt
classifier = ClassificationHead(embed_dim, config["num_classes"]).to(device)

# Freeze early layers to preserve general features
for name, param in encoder.named_parameters():
    if "stages.0" in name or "stages.1" in name:  # Freeze first two stages
        param.requires_grad = False

# Optimizer and scheduler
optimizer = optim.AdamW(
    list(encoder.parameters()) + list(classifier.parameters()),
    lr=config["lr"],
    weight_decay=1e-5
)
scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-6)

# Data loaders (from your MoCo script)
train_loader = probe_train_loader  # Labeled training data
val_loader = probe_val_loader      # Labeled validation data

# Loss function
criterion = nn.CrossEntropyLoss()

# Training function
def train_finetune_epoch(model, classifier, dataloader, optimizer, device):
    model.train()
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features = model(images)
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# Validation function
def validate_finetune(model, classifier, dataloader, device):
    model.eval()
    classifier.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, all_preds, all_labels

# Compute additional metrics
def compute_metrics(preds, labels, num_classes=5):
    report = classification_report(labels, preds, output_dict=True)
    cm = confusion_matrix(labels, preds)
    sensitivity = [cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(num_classes)]
    specificity = [
        np.sum(np.delete(np.delete(cm, i, 0), i, 1)) / 
        (cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) 
        if (cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]) > 0 else 0 
        for i in range(num_classes)
    ]
    return report, sensitivity, specificity

# Fine-tuning loop
best_val_acc = 0.0
for epoch in range(config["num_epochs"]):
    # Train
    train_loss, train_acc = train_finetune_epoch(encoder, classifier, train_loader, optimizer, device)
    
    # Validate
    val_loss, val_acc, val_preds, val_labels = validate_finetune(encoder, classifier, val_loader, device)
    
    # Compute metrics
    report, sensitivity, specificity = compute_metrics(val_preds, val_labels)
    
    # Log to wandb
    wandb_run.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": report['weighted avg']['f1-score']
    })
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'val_accuracy': val_acc
        }, "model/new/chckpt/moco/finetuned_best.pth")
    
    # Step scheduler
    scheduler.step()

# Finalize wandb run
wandb_run.finish()