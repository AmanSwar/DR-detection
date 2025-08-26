import torch
import torch.nn as nn
import timm
import os
from data_pipeline import data_aug, data_set
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb 

class MoCoV3Model(nn.Module):
    def __init__(self, base_model='convnext_small', projection_dim=256, hidden_dim=1024,
                 queue_size=4096, momentum=0.99, pretrained=False):
        super(MoCoV3Model, self).__init__()
        self.m = momentum
        self.query_encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        feature_dim = self.query_encoder.num_features
        self.query_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.key_encoder = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        self.key_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(queue_size, projection_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, im_q, im_k):
        q = self.query_encoder(im_q)
        q = self.query_projector(q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            k = self.key_encoder(im_k)
            k = self.key_projector(k)
            k = nn.functional.normalize(k, dim=1)
        return q, k

    @torch.no_grad()
    def update_key_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            overflow = (ptr + batch_size) - self.queue_size
            self.queue[ptr:self.queue_size] = keys[:(batch_size - overflow)]
            self.queue[0:overflow] = keys[(batch_size - overflow):]
            self.queue_ptr[0] = overflow
        else:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

class LinearProbeHead(nn.Module):
    def __init__(self, embed_dim, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    all_feats = []
    all_labels = []
    print(f"Extracting features from {len(dataloader)} batches...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        try:
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")
            images = images.to(device)
            feats = model.query_encoder(images)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue
    if not all_feats:
        raise ValueError("No features were extracted successfully")
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"Successfully extracted features: {all_feats.shape}, labels: {all_labels.shape}")
    return all_feats, all_labels

import tqdm
def linear_probe_evaluation(model, train_loader, val_loader, device):
    try:
        print("Starting feature extraction for linear probe...")
        train_feats, train_labels = extract_features(model, train_loader, device)
        val_feats, val_labels = extract_features(model, val_loader, device)
        print(f"Training linear probe on {train_feats.shape[0]} samples...")
        embed_dim = train_feats.shape[1]
        num_classes = len(train_labels.unique())
        probe = LinearProbeHead(embed_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        probe_epochs = 15
        for ep in range(probe_epochs):
            probe.train()
            perm = torch.randperm(train_feats.size(0))
            train_feats_shuf = train_feats[perm].to(device)
            train_labels_shuf = train_labels[perm].to(device)
            batch_size = 8
            running_loss = 0.0
            num_batches = 0
            for i in range(0, train_feats_shuf.size(0), batch_size):
                end = min(i + batch_size, train_feats_shuf.size(0))
                batch_feats = train_feats_shuf[i:end]
                batch_labels = train_labels_shuf[i:end]
                optimizer.zero_grad()
                outputs = probe(batch_feats)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num_batches += 1
            avg_loss = running_loss / num_batches
            print(f"Epoch {ep+1}/{probe_epochs}, Avg Loss: {avg_loss:.4f}")
        print("Evaluating linear probe...")
        probe.eval()
        val_feats_gpu = val_feats.to(device)
        with torch.no_grad():
            logits = probe(val_feats_gpu)
            pred = torch.argmax(logits, dim=1).cpu()
            acc = (pred == val_labels).float().mean().item() * 100.0
        print(f"[Linear Probe] Validation Accuracy: {acc:.2f}%")
        return acc
    except Exception as e:
        print(f"Error in linear probe evaluation: {str(e)}")
        return None

def knn_evaluation(model, train_loader, val_loader, device, k=5):
    from collections import Counter
    import numpy as np
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)
    train_feats_np = train_feats.numpy()
    train_labels_np = train_labels.numpy()
    val_feats_np = val_feats.numpy()
    val_labels_np = val_labels.numpy()
    correct = 0
    for i in tqdm.tqdm(range(len(val_feats_np))):
        diff = train_feats_np - val_feats_np[i]
        dist = np.sum(diff**2, axis=1)
        idx = np.argsort(dist)[:k]
        neighbors = train_labels_np[idx]
        majority = Counter(neighbors).most_common(1)[0][0]
        if majority == val_labels_np[i]:
            correct += 1
    acc = 100.0 * correct / len(val_feats_np)
    print(f"[k-NN (k={k})] Validation Accuracy: {acc:.2f}%")
    return acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MoCoV3Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=300 - 10, eta_min=1e-5)

# Checkpoint loading
checkpoint_pth = "model/checkpoint/moco/best_checkpoint.pth"
start_epoch = 0  # Default value if checkpoint doesn't exist
if os.path.exists(checkpoint_pth):
    checkpoint = torch.load(checkpoint_pth, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']

# Data loaders
train_aug = data_aug.MoCoSingleAug(img_size=256)
val_aug = data_aug.MoCoSingleAug(img_size=256)
dataset_names = ["aptos", "ddr", "idrid", "messdr"]
probe_train_loader = data_set.UniformTrainDataloader(
    dataset_names=dataset_names,
    transformation=train_aug,
    batch_size=32,
    num_workers=0,
    sampler=True
).get_loader()
probe_val_loader = data_set.UniformValidDataloader(
    dataset_names=dataset_names,
    transformation=val_aug,
    batch_size=32,
    num_workers=0,
    sampler=True
).get_loader()

# Set model to evaluation mode
model.eval()

# Initialize wandb
# wandb.init(project="MoCoV3_Evaluation", name=f"eval_epoch_{start_epoch}", config={
#     "model": "MoCoV3",
#     "base_model": "convnext_small",
#     "dataset_names": dataset_names,
#     "checkpoint": checkpoint_pth,
#     "epoch": start_epoch,
#     "linear_probe_epochs": 5,
#     "knn_k": 5,
# })

# Evaluate the model
print("Evaluating the pretrained MoCo v3 model...")
linear_acc = linear_probe_evaluation(model, probe_train_loader, probe_val_loader, device)
knn_acc = knn_evaluation(model, probe_train_loader, probe_val_loader, device, k=5)

# Log accuracies to wandb
# wandb.log({
#     "linear_probe_accuracy": linear_acc,
#     "knn_accuracy": knn_acc,



#     "epoch": start_epoch
# })

# Print results
print(f"Accuracy (Linear: {linear_acc:.2f}%, k-NN: {knn_acc:.2f}%)")

# Finish the wandb run
# wandb.finish()