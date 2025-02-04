import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange

import wandb
from tqdm import tqdm
import time

class PatchEmbedding(nn.Module):

    def __init__(self, img_size=224 , patch_size=16 , in_chan=3 , embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chan, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, C, H', W')
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    

class TransEncoder(nn.Module):

    def __init__(self , dim , depth , n_heads , mlp_dim , dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            block = nn.ModuleList(
                [
                    nn.LayerNorm(dim),
                    nn.MultiheadAttention(dim, n_heads, dropout=dropout),
                    nn.LayerNorm(dim),
                    nn.Sequential(
                        nn.Linear(dim, mlp_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(mlp_dim, dim),
                        nn.Dropout(dropout)
                    )
                ]
            )
            self.layers.append(block)


    def forward(self , x):
        for norm1 , attn , norm2 , mlp in self.layers:
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + mlp(norm2(x))
        return x
    


class IJEPA(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        encoder_depth=12,
        predictor_depth=4,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        
        # Image embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        # Target encoder
        self.target_encoder = TransEncoder(
            embed_dim,
            encoder_depth,
            num_heads,
            embed_dim * mlp_ratio,
            dropout
        )
        
        # Context encoder
        self.context_encoder = TransEncoder(
            embed_dim,
            encoder_depth,
            num_heads,
            embed_dim * mlp_ratio,
            dropout
        )
        
        # Predictor
        self.predictor = TransEncoder(
            embed_dim,
            predictor_depth,
            num_heads,
            embed_dim * mlp_ratio,
            dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self , m):

        if isinstance(m , nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def momentum_update(self , target_encoder: nn.Module , context_encoder: nn.Module , momentum=0.999):
        for target_param, context_param in zip(target_encoder.parameters(), context_encoder.parameters()):
            target_param.data.mul_(momentum).add_((1 - momentum) * context_param.data)

    
    def get_random_boxes(self , batch_size , n_boxes = 4):
        """Generate random target boxes for masked prediction"""
        boxes = []
        for _ in range(batch_size):
            batch_boxes = []
            for _ in range(n_boxes):
                x1 = torch.randint(0, 14, (1,)).item()
                y1 = torch.randint(0, 14, (1,)).item()
                w = torch.randint(2, 6, (1,)).item()
                h = torch.randint(2, 6, (1,)).item()
                batch_boxes.append([x1, y1, x1 + w, y1 + h])
            boxes.append(batch_boxes)
        return torch.tensor(boxes)
    

    def extract_targets(self, features, boxes):
        """Extract target features based on boxes"""
        B, N, D = features.shape
        H = W = int(N ** 0.5)
        features = rearrange(features, 'b (h w) d -> b d h w', h=H)
        
        target_features = []
        #batch loop
        for b in range(B):
            batch_targets = []
            #different boxes
            for box in boxes[b]:
                x1, y1, x2, y2 = box
                target = features[b:b+1, :, y1:y2, x1:x2]
                target = F.adaptive_avg_pool2d(target, (1, 1)).squeeze(-1).squeeze(-1)
                batch_targets.append(target)
            target_features.append(torch.cat(batch_targets, dim=0))
        return torch.stack(target_features)
    
    def forward(self, images, boxes=None):
        B = images.shape[0]
        if boxes is None:
            boxes = self.get_random_boxes(B)
            
        # Get patch embeddings
        x = self.patch_embed(images)
        
        # Get context features
        context_features = self.context_encoder(x)
        
        # Get target features (with no gradient)
        with torch.no_grad():
            target_features = self.target_encoder(x)
            target_features = self.extract_targets(target_features, boxes)
        
        # Predict target features
        predicted_features = self.predictor(context_features)
        predicted_features = self.extract_targets(predicted_features, boxes)
        
        return predicted_features, target_features
    

class IJEPALoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, predicted_features, target_features):
        # Normalize features
        predicted_features = F.normalize(predicted_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        
        # Compute cosine similarity
        sim = torch.einsum('bnd,bnd->bn', predicted_features, target_features)

       
        # Compute loss (negative cosine similarity)
        temperature = 0.1
        sim = sim / temperature
        loss = -F.log_softmax(sim, dim=-1).mean()
        
        return loss


def create_ijepa_model(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    encoder_depth=12,
    predictor_depth=4,
    num_heads=12,
    mlp_ratio=4,
    dropout=0.1
):
    model = IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        predictor_depth=predictor_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout
    )
    criterion = IJEPALoss()
    return model, criterion



class IJEPATrainer:
    """
    Class for Ijepa trianing in single GPU
    """

    def __init__(
            self,
            model,
            loss_fn,
            train_loader,
            optim,
            scheduler=None,
            max_ep = 300,
            save_dir = "checkpoints",
            log_interval = 100,
            save_interval = 10,
            device = 'cuda'

    ):
        
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.optim = optim
        self.scheduler = scheduler
        self.max_epoch = max_ep
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.device = device

    def save_checkpoints(self , epoch , loss):
        
        model_state = self.model.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optim_state_dict': self.optim.state_dict(),
            'loss': loss,
        }
    
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        

        torch.save(checkpoint , self.save_dir / f"cehckpoint_ep_{epoch}.pt")
 
    def train_epoch(
            self,
            epoch,
    ):
        self.model.train()
        self.model.to(self.device)
        n_batch = len(self.train_loader)
        
        total_loss = 0
        pbar = tqdm(total=n_batch , desc=f"Epoch :{epoch}")
        for batch_idx , img in enumerate(self.train_loader):

            img = img.to(self.device)
            self.optim.zero_grad()

            pred_feat , target_feat = self.model(img)
            loss = self.loss_fn(pred_feat , target_feat)

            loss.backward()
            self.optim.step()

            self.model.momentum_update(
                self.model.target_encoder,
                self.model.context_encoder
            )

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0 :
                wandb.log(
                    {
                        'batch_loss' : loss.item(),
                        'epoch' : epoch,
                        'batch' : batch_idx
                    }
                )

            pbar.update(1)

        pbar.close()


        avg_loss = total_loss / n_batch

        return avg_loss
    

    def train(self):

        best_loss = float("inf")

        for ep in tqdm(range(self.max_epoch)):


            epoch_start_time = time.time()
            loss = self.train_epoch(ep)
            epoch_duration = time.time() - epoch_start_time

            if self.scheduler is not None:
                self.scheduler.step()

            wandb.log(
                {
                    'epoch_loss' : loss,
                    'epoch' : ep,
                    'learning_rate' : self.optim.param_groups[0]['lr'],
                    'epoch_duration' : epoch_duration
                }
            )

            print(f"Epoch {ep} : Loss = {loss:.4f} , Time={epoch_duration:.2f}s")

            if ep % self.save_interval == 0 or loss < best_loss:
                self.save_checkpoints(ep , loss)
                if loss < best_loss:
                    best_loss = loss


if __name__ == "__main__":
    print("Hii I am here")
    BATCH_SIZE = 64
    from data_pipeline import data_set , data_aug
    dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid"]
    uniform_data_ld = data_set.UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=data_aug.IJEPAAugmentation(),
        batch_size=BATCH_SIZE,
        num_workers=4,
        sampler=True
    )

    data_ld = uniform_data_ld.get_loader()
    model , loss_fn = create_ijepa_model()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        betas=(0.9 , 0.95),
        weight_decay=0.05
        )
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=300,
        eta_min=1e-6
    )


    trainer = IJEPATrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader= data_ld,
        optim=optim,
        scheduler=scheduler,
    )

    trainer.train()

    
