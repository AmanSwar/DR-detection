import torch
import torch.nn as nn
import timm
from data_pipeline.data_set import UniformTrainDataloader , UniformValidDataloader
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]


class Convnext(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=6,
        )

    def forward(self , x):
        out = self.model(x)

        return out
    


def get_train_augmentation():
    return transforms.Compose([
        transforms.Resize(size=(256 , 256)),  
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

def get_val_augmentation(): 
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

train_aug = get_train_augmentation()
val_aug = get_val_augmentation()

trainloader = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=train_aug,
        batch_size=32,
        num_workers=0,
        sampler=True
    ).get_loader()



device = torch.device("cuda")
model = Convnext()
model.to(device=device)
loss_fn = nn.CrossEntropyLoss()
lr = 1e-4
optim = torch.optim.Adam(params=model.parameters() , lr=lr)

def train_one_epoch():
    
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for image , label in tqdm(trainloader):
        image, label = image.to(device), label.to(device)

        logits = model(image)
        loss = loss_fn(logits , label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        running_loss += loss.item()

        print(f"BATCH LOSS  : {loss.item():.3f}")

    print("\n")
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1

num_epoch = 100
for epoch in range(1 , num_epoch + 1):
    train_loss , train_acc , train_f1 = train_one_epoch()
    print(f"Epoch {epoch}:")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        







