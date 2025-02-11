import os
from torch.utils.data import DataLoader , Dataset

from PIL import Image

class ssltestdataset(Dataset):

    def __init__(self , trans):
        super().__init__()
        self.trans = trans
        self.base_dir = "tests/data"

        self.imgs = os.listdir(self.base_dir)

        for i in range(len(self.imgs)):

            self.imgs[i] = os.path.join(self.base_dir , self.imgs[i])

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        
        img_path = self.imgs[index]

        img = Image.open(img_path)

        if self.trans is not None:

            trans_img = self.trans(image=img)

            return trans_img
        
        return img



