import torch.nn as nn
from einops import rearrange

class RearrangeAndLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        
    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.layer_norm(x)
        # return to original shape if needed
        x = rearrange(x, 'b h w c -> b c h w')
        return x


vit_config = {
    "img_size" : 512,
    "patch_size" : 32,
    "in_chans" : 3,
    "embed_dim" : 512,
    "depth" : 6,
    "num_heads" : 12,
    "mlp_ratio" : 4,
    "mask_ratio" : 0.4, #only for ibot
    "num_regs" : 8, # only for dino with regs
    "pred_depth" : 4

}

"""
vit_config["img_size"]
vit_config["patch_size"]
vit_config["num_regs"]
vit_config["mlp_ratio"]
vit_config["num_heads"]
vit_config["depth"]
vit_config["embed_dim"]
vit_config["in_chans"]

"""