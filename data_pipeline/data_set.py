import sys
sys.path.append("/home/aman/code/research/CV/dia_ret/data_pipeline")
import torch
from torch.utils.data import Dataset , DataLoader

#util import 
from data_load import EyepacsGradingDataset , AptosGradingDataset , IdridGradingDataset , DdrGradingDataset
import random
from PIL import Image

class UnitedTrainingDataset(Dataset):

    def __init__(self , transformation=None , *args):
        self.args = args
        self.image_path = []
        self.labels = []
        self.transformation = transformation

        #concatinating all datasets
        for arg in args:
            img_path , label = self.__getdata(arg)
            self.image_path.extend(img_path)
            self.labels.extend(label)

        #logic for shuffling
        img_path_label_pair = list(zip(self.image_path , self.labels))
        random.shuffle(img_path_label_pair)
        self.image_path , self.labels = zip(*img_path_label_pair)

    def __getdata(self , dataset_name: str) -> list:

        if dataset_name == "eyepacs":
            eyepacs = EyepacsGradingDataset()
            eyepacs_train_img , eyepacs_train_labels = eyepacs.get_train_set()
            
            return eyepacs_train_img , eyepacs_train_labels

        elif dataset_name == "aptos":
            aptos = AptosGradingDataset()
            aptos_train_img , aptos_train_labels = aptos.get_train_set()
            
            return aptos_train_img , aptos_train_labels
           

        elif dataset_name == "ddr":
            ddr = DdrGradingDataset()
            ddr_train_img , ddr_train_labels = ddr.get_train_set()
            
            return ddr_train_img , ddr_train_labels
        
        elif dataset_name == "idrid":
            idrid = IdridGradingDataset()
            idrid_train_img , idrid_train_labels = idrid.get_train_set()
            return idrid_train_img , idrid_train_labels

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):

        img_path = self.image_path[index]
        label = self.labels[index]

        img = Image.open(img_path)

        if self.transformation is not None:
            img = self.transformation(img)

        return img , label


     
        


        