from model.ijepa import PatchEmbedding , IJEPA , IJEPALoss , IJEPATrainer

import torch
from typing import Tuple

def sim_img(img_size) -> torch.tensor :
    img = torch.randn(size=(img_size , img_size))
    return img



def test_patch_embedding(img_size):

    img = sim_img(img_size=img_size)

    model = PatchEmbedding(
        img_size=img_size,
    )

    return model(img)

def test_trans_encoder(img_size):

    img = sim_img(img_size=img_size)
    


if __name__ == "__main__":

    out = test_patch_embedding(img_size=224)
    print(out)
