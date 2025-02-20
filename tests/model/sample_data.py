import torch
from torch.utils.data import DataLoader , Dataset
import os
from PIL import Image
from data_pipeline.data_aug import DinowregAug , IbotRetAug , IJEPAAugmentation , DINOAugmentation
from tests.utils import show_img , show_multi_img

img_dir = "tests/data"

class SslDs(Dataset):

    def __init__(self , trans):

        img_names = os.listdir(img_dir)
        from data_pipeline.utils import add_path

        add_path(img_names , img_dir)

        self.images = img_names
        self.trans = trans
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self , index):
        
        img_path = self.images[index]
        img = Image.open(img_path)

        if self.trans:

            trans_img = self.trans(img)
            return trans_img
        
        return img

# data_aug = IbotRetAug(img_size=512)
# train_ds = SslDs(trans=data_aug)

# print(len(train_ds))
# show_img(train_ds=train_ds)





        