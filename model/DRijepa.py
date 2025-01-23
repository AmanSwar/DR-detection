import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Patchify(nn.Module):
    """
    convert images into patches , with hierachial project to support large img size
    """

    def __init__(self , img_size = 2048 , patch_size =32, in_chan=3 , embed_dim = 1024):
        super().__init__()

        self.img_size = img_size
        self.path_size = patch_size
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
        self.block = nn.ModuleList([
            nn.LayerNorm(dim),
            nn.MultiheadAttention(dim , heads , dropout=dropout),
            nn.LayerNorm(dim),
            nn.Sequential(
                nn.Linear(dim , mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim , dim),
                nn.Dropout(dropout)
            )
        ])
        for _ in range(depth):
            self.layers.append(self.block)

    def forward(self,  x):

        for norm1 , attn, norm2 , mlp in self.layers:
            x_norm = norm1(x)
            x_attn = attn(x_norm , x_norm ,x_norm)
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

    
    def get_random_boxes(self , batch_size , n_box=6):

        boxes = []

        for _ in range(n_box):
            batch_wise_box = []
            for _ in range(batch_size):
                # to generate images more inclined to center
                center_bias = torch.randn(2) * 0.5
                x_center = self.grid_size // 2 + int(center_bias[0] * self.grid_size // 4)
                y_center = self.grid_size // 2 + int(center_bias[1] * self.grid_size // 4)
                w = torch.randint(4 , 8 , (1,)).item()
                h = torch.randint(4 , 8 , (1,)).item()

                x1 = max(0 , x_center - w // 2)
                y1 = max(0 , y_center - h // 2)
                x2 = min(self.grid_size , x1 + w)
                y2 = min(self.grid_size , y1 + h)

                batch_wise_box.append([x1 , y1 , x2 , y2])

            boxes.append(batch_wise_box)


        return torch.tensor(boxes)
    
    def extract_target(self , feature , boxes):
        B, N, D = features.shape
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
    
def create_ijepa(
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




        

