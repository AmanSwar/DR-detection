import torchvision.models as models
import torch.nn as nn

class VisionTrans(nn.Module):

    def __init__(self):
        super().__init__()

        self.base_model = models.vit_b_16(weights=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1000 , out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512 , out_features=4)
        )

    def forward(self , x):
        x = self.base_model(x)
        x = self.mlp(x)

        return x
