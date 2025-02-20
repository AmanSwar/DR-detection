import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SubConLoss(nn.Module):

    def __init__(self , temp=0.07):

        super().__init__()
        self.temp = temp

    def forward(self , features , labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1 , 1)
        mask = torch.eq(labels , labels.T).float().to(device)

        features = F.normalize(features , dim=1)

        similarity_matrix = torch.matmul(features , features.T) / self.temp

        logits_max , _ = torch.max(similarity_matrix , dim=1 , keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1 , 1).to(device),
            0
        )

        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1 , keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss
    
class DRSupConModel(nn.Module):
    def __init__(
            self,
            ijepa_model,
            num_classes,
            proj_dim,
            dropout
    ):
        super().__init__()
        self.ijepa_backbone = ijepa_model
        embed_dim = ijepa_model.predictor[-1].out_features

        for param in self.ijepa_backbone.parameters():
            param.requires_grad = False

        self.encoder = self.ijepa_backbone.target_encoder

        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim , embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim , proj_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim , embed_dim // 2),
            nn.LayerNorm(embed_dim //2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2 , num_classes)
        )

    def forward(self , x , return_features=False):
        x = self.ijepa_backbone.patch_embed(x)

        features = self.encoder(x)

        features = rearrange(features , 'b (h w) c -> b c h w' , h=int(features.size(1) ** 0.5))

        features = features.squeeze(-1).squeeze(-1)

        projection = self.proj_head(features)

        logits = self.classifier(features)

        if return_features:
            return features , projection , logits
        
        return projection , logits
    
def train_step(model , image , label , optim , sup_con_loss_fn , ce_loss_fn , con_weight=0.5 , use_amp=False):

    features , projection , logits = model(image , return_features=True)
    sup_con_loss = sup_con_loss_fn(projection , label)

    ce_loss = ce_loss_fn(logits , label)

    total_loss = con_weight * sup_con_loss + (1 - con_weight) * ce_loss
    

    return total_loss , sup_con_loss , ce_loss


class DRTrainer:

    def __init__(
            self,
            model ,
            optim,
            device,
            con_weight,
            temp
    ):
        
        self.model = model
        self.optim = optim
        self.device = device
        self.con_weight = con_weight
        self.sup_con_loss_fn = SubConLoss(temp=temp).to(device)
        self.ce_loss_fn = nn.CrossEntropyLoss().to(device)


    def train_epoch(self , train_loader):

        self.model.train()
        total_loss = 0
        total_sup_con_loss = 0
        total_ce_loss = 0

        for batch_idx , (image , label) in enumerate(train_loader):

            image = image.to(self.device)
            label = label.to(self.device)

            loss , sup_con_loss , ce_loss = train_step(
                model=self.model,
                image=image,
                label=label,
                optim=self.optim,
                sup_con_loss_fn=self.sup_con_loss_fn,
                ce_loss_fn=self.ce_loss_fn,
                con_weight=self.con_weight
            )

            total_loss += loss.item()
            total_sup_con_loss += sup_con_loss.item()
            total_ce_loss += ce_loss.item()

            if batch_idx % 50 == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'SupCon Loss: {sup_con_loss.item():.4f}, '
                      f'CE Loss: {ce_loss.item():.4f}')
                
        
        return {
            'loss': total_loss / len(train_loader),
            'sup_con_loss' : total_sup_con_loss / len(train_loader),
            'ce_loss' : total_ce_loss / len(train_loader)
        }


