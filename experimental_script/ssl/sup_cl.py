"""
Supervised Contrastive learning
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.models as models
import torch.nn.functional as F

class DataAug:
    def __init__(self , img_size):
        self.transforms = v2.Compose(
            [
                v2.RandomResizedCrop(img_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(0,4 , 0.4 , 0.4 , 0.1),
                v2.RandomGrayscale(p=0.2),
                v2.GaussianBlur(kernel_size=3),
                v2.ToTensor(),
            ]
        )
        

    def __call__(self , img):
        return self.transforms(img) , self.transforms(img)
    

class SupCon(nn.Module):

    def __init__(self , encoder_backbone="resnet152" , hidden_layer=2048):
        
        self.encoder_backbone = models.__dict__[encoder_backbone](weights=False)
        self.encoder_backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(2048 , hidden_layer),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer , hidden_layer),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer , 128),
            
        )

        

    def forward(self, x):

        encoded_x = self.encoder_backbone(x)
        norm_x1 = F.normalize(encoded_x)
        proj_x = self.projector(norm_x1)
        out = F.normalize(proj_x)

        return out



class SupConLoss(nn.Module):

    def __init__(self , temp=0.07 , base_temp=0.07):
        
        self.temp = temp
        self.base_temp = base_temp
    
    def forward(self , features , labels):

        batch_size = features.size[0]

        labels = labels.contiguous().view(-1,1)
        mask = torch.eq(labels , labels.T).float()

        features = F.normalize(features , dim=1)

        similarity_matrix = torch.div(
            torch.matmul(features , features.T),
            self.temp
        )

        logits_max , _ = torch.max(similarity_matrix , dim=1 , keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1,1).to(features.device),
            0
        )

        mask = mask * logits_mask

        exp_logits_sum = torch.sum(exp_logits * logits_mask , dim=1 , keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum)

        mask_sum = torch.sum(mask ,dim=1)
        mask_sum = torch.where(mask_sum == 0 , torch.ones_like(mask_sum) , mask_sum)

        loss = -(self.temp / self.base_temp) * torch.sum(
            mask * log_prob , dim=1) / mask_sum
        
        return loss.mean()
    
        


class SupConTrainer:

    def __init__(
            self,
            data_loader,
            loss_fn,
            optimizer,
            lr,
            epochs

    )

