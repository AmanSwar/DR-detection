import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


from tqdm import tqdm
import wandb
import time

class Patchify(nn.Module):
    """
    convert images into patches , with hierachial project to support large img size
    """

    def __init__(self , img_size = 2048 , patch_size =32, in_chan=3 , embed_dim = 1024):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # for better learned representation of image ? idk claude suggested me but I am sceptical 
        self.hierarch_proj = nn.Sequential(
            #
            nn.Conv2d(in_chan , embed_dim // 4 , kernel_size=7 , stride=2 , padding=3),
            nn.LayerNorm([embed_dim // 4 , img_size // 2 , img_size //2]),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4 , embed_dim // 2 , kernel_size=3 , stride=2 , padding=1),
            nn.LayerNorm([embed_dim // 2 , img_size // 4 , img_size //4]),
            nn.GELU(),
            nn.Conv2d(embed_dim //2 , embed_dim , kernel_size=patch_size // 4 , stride=patch_size //4)
            
        )

    def forward(self , x):
        x = self.hierarch_proj(x) # -> (batch channel height width)
        x = rearrange(x , 'b c h w -> b (h w) c')
        return x


class TransformerEncoder(nn.Module):

    def __init__(self , dim , depth , heads , mlp_dim , dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            block = nn.ModuleList(
                [
                    nn.LayerNorm(dim),
                    nn.MultiheadAttention(dim, heads, dropout=dropout),
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

    def forward(self,  x):

        for norm1 , attn, norm2 , mlp in self.layers:
            x_norm = norm1(x)
            x_attn = attn(x_norm , x_norm ,x_norm)
            print(type(x))
            print(type(x_attn))
            print(len(x_attn))
            
            x = x + x_attn

            x_norm_2 = norm2(x)
            x_mlp = mlp(x_norm_2)
            x = x + x_mlp

        return x
    
class DRIjepa(nn.Module):
    def __init__(
            self,
            img_size = 2048, 
            patch_size = 32 ,
            in_chan = 3 , 
            embed_dim = 1024 , 
            encoder_depth = 12 , 
            pred_depth = 4 ,
            n_heads = 16 , 
            mlp_ratio =4, 
            drop = 0.1
            ):
        
        super().__init__()

        self.patch_embed = Patchify(img_size=img_size , patch_size=patch_size , in_chan=in_chan , embed_dim=embed_dim)

        self.target_encoder = TransformerEncoder(
            dim=embed_dim , 
            depth=encoder_depth,
            heads=n_heads,
            mlp_dim= embed_dim * mlp_ratio,
            dropout=drop
        )


        self.context_encoder = TransformerEncoder(
            dim=embed_dim , 
            depth=encoder_depth,
            heads=n_heads,
            mlp_dim= embed_dim * mlp_ratio,
            dropout=drop
        )


        self.predictor = nn.Sequential(
            TransformerEncoder(
                dim=embed_dim , 
                depth=pred_depth,
                heads=n_heads,
                mlp_dim= embed_dim * mlp_ratio,
                dropout=drop
            ),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , embed_dim)
        )


        self.apply(self._init_weights)

        self.grid_size = img_size // patch_size

    def _init_weights(self , m):

        if isinstance(m , nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias , 0)

        elif isinstance(m , nn.LayerNorm):
            nn.init.constant_(m.bias , 0)

            nn.init.constant_(m.weight , 1)

    
    def get_random_boxes(self, batch_size, n_box=6):
        boxes = []
        for b in range(batch_size):
            batch_boxes = []
            for _ in range(n_box):
                center_bias = torch.randn(2) * 0.5
                x_center = self.grid_size // 2 + int(center_bias[0] * self.grid_size // 4)
                y_center = self.grid_size // 2 + int(center_bias[1] * self.grid_size // 4)
                w = torch.randint(4, 8, (1,)).item()
                h = torch.randint(4, 8, (1,)).item()
                x1 = max(0, x_center - w // 2)
                y1 = max(0, y_center - h // 2)
                x2 = min(self.grid_size, x1 + w)
                y2 = min(self.grid_size, y1 + h)
                batch_boxes.append([x1, y1, x2, y2])
            boxes.append(batch_boxes)
        return torch.tensor(boxes)
    
    def extract_target(self , feature , boxes):
        B, N, D = feature.shape
        H = W = int(N ** 0.5)
        features = rearrange(features, 'b (h w) d -> b d h w', h=H)
        
        target_features = []
        for b in range(B):
            batch_targets = []
            for box in boxes[b]:
                x1, y1, x2, y2 = box
                target = features[b:b+1, :, y1:y2, x1:x2]

                target = F.adaptive_avg_pool2d(target, (min(H, 2), min(W, 2)))
                target = rearrange(target, 'b c h w -> b (h w) c')
                batch_targets.append(target)
            target_features.append(torch.cat(batch_targets, dim=1))
        return torch.stack(target_features)
    
    @torch.no_grad()
    def momentum_update(
        self , 
        target_encoder: nn.Module , 
        context_encoder: nn.Module , 
        momentum=0.999
        
        ):
        for target_param, context_param in zip(target_encoder.parameters(), context_encoder.parameters()):
            target_param.data.mul_(momentum).add_((1 - momentum) * context_param.data)
    

    def forward(self , images , boxes=None):
        B = images.shape[0]

        if boxes is None:
            boxes = self.get_random_boxes(B)

        x = self.patch_embed(images)


        context_features = self.context_encoder(x)

        with torch.no_grad():
            target_feat = self.target_encoder(x)
            target_feat = self.extract_target(target_feat , boxes)

        predicted_feature = self.predictor(context_features)
        predicted_feature = self.extract_target(predicted_feature , boxes)

        return predicted_feature , target_feat


class IJEPALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self , predicted_features , target_features):

        predicted_features = F.normalize(predicted_features , dim=-1)
        target_features = F.normalize(target_features , dim=-1)

        #sim but using einsum
        sim = torch.einsum('bnd , bnd -> bn' , predicted_features , target_features)

        loss = -sim.mean()
        return loss
    
def create_DRijepa(
        img_size = 2048,
        patch_size = 32 , 
        in_chan = 3 , 
        embed_dim = 1024 ,
        encoder_depth = 12,
        pred_depth = 4,
        n_heads = 16,
        mlp_ratio = 4,
        dropout = 0.1
):
    model = DRIjepa(
        img_size=img_size,
        patch_size=patch_size,
        in_chan=in_chan,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        pred_depth=pred_depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        drop=dropout
    )

    loss = IJEPALoss()
    return model , loss

class Trainer:

    def __init__(
            self,
            model,
            loss_fn,
            train_loader,
            optim,
            scheduler=None,
            max_ep=300,
            save_dir="checkpoint",
            log_interval=100,
            save_interval=10,
            device="cuda"
    ):
        
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.optim = optim
        self.scheduler = scheduler
        self.max_ep = max_ep
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.device = device

    def save_checkpoint(
            self,
            epoch,
            loss
    ):
        
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

    def train_epoch(self, epoch):

        self.model.train()

        

        n_batch = len(self.train_loader)

        total_loss = 0
        pbar = tqdm(total=n_batch , desc=f"Epoch : {epoch}")

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


            if batch_idx % self.log_interval == 0:

                wandb.log(
                    {
                        'batch_loss' : loss.item(),
                        'epoch' : epoch,
                        'batch' : batch_idx
                    }
                )


            pbar.update()


        pbar.close()

        avg_loss = total_loss / n_batch
        return avg_loss
    

    def train(self):

        best_loss = float("inf")
        self.model.to(self.device)
        for ep in tqdm(range(self.max_ep)):

            epoch_start_time = time.time()

            loss = self.train_epoch(ep)
            ep_dur = time.time() - epoch_start_time

            if self.scheduler is not None:
                self.scheduler.step()


            wandb.log(
                {
                    'epoch_loss' : loss,
                    'epoch' : ep,
                    'learning_rate' : self.optim.param_groups[0]['lr'],
                    'epoch_duration' : ep_dur
                }
            )

            print(f"Epoch {ep} : Loss = {loss:.4f} , Time={ep_dur:.2f}s")

            if ep % self.save_interval == 0 or loss < best_loss:
                self.save_checkpoint(ep , loss)
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
    model , loss_fn = create_DRijepa()
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


    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader= data_ld,
        optim=optim,
        scheduler=scheduler,
    )

    trainer.train()
    
