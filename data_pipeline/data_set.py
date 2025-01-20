
import torch
from torch.utils.data import Dataset , DataLoader

#util import 
from data_pipeline.data_load import EyepacsGradingDataset , AptosGradingDataset , IdridGradingDataset , DdrGradingDataset
import random
from PIL import Image
from typing import Tuple , List


class UnitedTrainingDataset(Dataset):

    def __init__(self , *args , transformation=None):
        self.args = args
        self.image_path = []
        self.labels = []
        self.transformation = transformation

        #appending all datasets
        for arg in args:
            img_path , label = self.__getdata(arg)
            self.image_path.extend(img_path)
            self.labels.extend(label)

        #logic for shuffling
        img_path_label_pair = list(zip(self.image_path , self.labels))
        random.shuffle(img_path_label_pair)
        self.image_path , self.labels = map(list , zip(*img_path_label_pair))

    def __getdata(self , dataset_name: str) -> Tuple[List[str] , List[int]]:

        if dataset_name not in ["eyepacs" , "aptos" ,"ddr" , "idrid"]:
            raise ValueError(f"Unknown dataset {dataset_name}")
        

        elif dataset_name.lower() == "eyepacs":
            eyepacs = EyepacsGradingDataset()
            eyepacs_train_img , eyepacs_train_labels = eyepacs.get_train_set()
            
            return eyepacs_train_img , eyepacs_train_labels

        elif dataset_name.lower() == "aptos":
            aptos = AptosGradingDataset()
            aptos_train_img , aptos_train_labels = aptos.get_train_set()
            
            return aptos_train_img , aptos_train_labels
           

        elif dataset_name.lower() == "ddr":
            ddr = DdrGradingDataset()
            ddr_train_img , ddr_train_labels = ddr.get_train_set()
            
            return ddr_train_img , ddr_train_labels
        
        elif dataset_name.lower() == "idrid":
            idrid = IdridGradingDataset()
            idrid_train_img , idrid_train_labels = idrid.get_train_set()
            return idrid_train_img , idrid_train_labels

    def get_paths(self) -> Tuple[List[str] , List[int]]:
        """
        To get combined path of all images of given dataset and corresponding labels
        """

        return self.image_path , self.labels
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        img_path = self.image_path[index]
        label = self.labels[index]

        try:
            img = Image.open(img_path)

            if self.transformation is not None:
                img = self.transformation(img)

            return img , label
        except Exception as e:
            raise RuntimeError(f"error loading img {img_path}")


class UnitedValidationDataset(Dataset):

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
            eyepacs_valid_img , eyepacs_valid_labels = eyepacs.get_valid_set()
            
            return eyepacs_valid_img , eyepacs_valid_labels

        elif dataset_name == "aptos":
            aptos = AptosGradingDataset()
            aptos_valid_img , aptos_valid_labels = aptos.get_valid_set()
            
            return aptos_valid_img , aptos_valid_labels
           

        elif dataset_name == "ddr":
            ddr = DdrGradingDataset()
            ddr_valid_img , ddr_valid_labels = ddr.get_valid_set()
            
            return ddr_valid_img , ddr_valid_labels
        
        elif dataset_name == "idrid":
            idrid = IdridGradingDataset()
            idrid_valid_img , idrid_valid_labels = idrid.get_valid_set()
            return idrid_valid_img , idrid_valid_labels
    def get_paths(self) -> list:
        """
        To get combined path of all images of given dataset and corresponding labels
        """

        return self.image_path , self.labels
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):

        img_path = self.image_path[index]
        label = self.labels[index]

        img = Image.open(img_path)

        if self.transformation is not None:
            img = self.transformation(img)

        return img , label

     
        



