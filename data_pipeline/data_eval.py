import os
from torch.utils.data import DataLoader , Dataset
import pandas as pd
from data_pipeline.data_load import (
    DdrGradingDataset ,
    AptosGradingDataset , 
    IdridGradingDataset,
    MessdrGradingDataset,
    SustechDataset
)

from PIL import Image

"""
Eyepacs
aptos
idrid
ddr
messidior
sustech
deepdr
"""



aptos_test_set = AptosGradingDataset()
idirid_test_set = IdridGradingDataset()
ddr_test_set = DdrGradingDataset(root_dir='data/ddr')
messdr_test_set = MessdrGradingDataset()
sustech_test_set = SustechDataset()


class UniTestDataset(Dataset):

    def __init__(self, dataset_name , transform):

        self.images = []
        self.labels = []
        self.transform = transform
        self.dataset_name = dataset_name

        self._get_data()

    def _get_data(self):
        ds = self.dataset_name

        if ds == "aptos":
            img, label = AptosGradingDataset().get_test_set()

        elif ds == "idirid":
            img, label = IdridGradingDataset().get_test_set()

        elif ds == "ddr":
            img, label = DdrGradingDataset(root_dir='data/ddr').get_test_set()
        elif ds == "messdr":
            img, label = MessdrGradingDataset().get_test_set()
        elif ds == "sustech":
            img, label = SustechDataset().get_test_set()


        self.images = img
        self.labels = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        img_path = self.images[index]
        label = self.labels[index]
        img = Image.open(img_path)
        trans_img = self.transform(img)

        return trans_img , label
    


class UniTestLoader(DataLoader):

    def __init__(self , dataset_name , transforms , batch_size , num_worker):

        self.dataset_name = dataset_name
        self.trans = transforms
        self.batch_size = batch_size
        self.num_worker = num_worker
        
        ds = UniTestDataset(self.dataset_name ,transforms)

        self.dl = DataLoader(
            dataset=ds,
            batch_size= batch_size,
            num_workers=self.num_worker,
            pin_memory=True
        )

    def get_loader(self):
        return self.dl
    












