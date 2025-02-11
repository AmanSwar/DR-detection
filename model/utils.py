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
    "img_size" : 1024,
    "patch_size" : 32,
    "in_chans" : 3,
    "embed_dim" : 1024,
    "depth" : 12,
    "num_heads" : 16,
    "mlp_ratio" : 4,
    "mask_ratio" : 0.4, #only for ibot
    "num_regs" : 8, # only for dino with regs
    "pred_depth" : 4

}