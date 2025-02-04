
import torch
from torch.utils.data import Dataset , DataLoader , WeightedRandomSampler

#util import 
from data_pipeline.data_load import EyepacsGradingDataset , AptosGradingDataset , IdridGradingDataset , DdrGradingDataset
import random
from PIL import Image
from typing import Tuple , List

import numpy as np


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
            ddr = DdrGradingDataset(root_dir="data/ddr")
            ddr_train_img , ddr_train_labels = ddr.get_train_set()
            
            return ddr_train_img , ddr_train_labels
        
        elif dataset_name.lower() == "idrid":
            idrid = IdridGradingDataset()
            idrid_train_img , idrid_train_labels = idrid.get_train_set()
            return idrid_train_img , idrid_train_labels

    def get_paths(self) ->  List[float]:
        """
        To get combined path of all images of given dataset and corresponding labels
        """

        return self.image_path
    
    def get_labels(self) -> List[int]:
        """
        Function to return labels
        """
        return self.labels

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        img_path = self.image_path[index]
        label = self.labels[index]

        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transformation is not None:
                img = np.array(img)
                img = self.transformation(img)

            return img , label
        except (IOError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading {img_path}: {str(e)}")


class UnitedValidationDataset(Dataset):

    def __init__(self ,*args , transformation=None):
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
            ddr = DdrGradingDataset(root_dir="data/ddr")
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

     
        

class UniformTrainDataloader:

    def __init__(
            
            self ,
            dataset_names: List,
            transformation , 
            batch_size: int,
            num_workers: int,
            sampler=True
            
            ):
        
        self.dataset_names = dataset_names
        self.sampler = sampler 
        self.transformation = transformation
        self.batch_size = batch_size
        self.num_workers = num_workers
        training_dataset = UnitedTrainingDataset(*self.dataset_names , transformation=self.transformation)
        
        if sampler:
            from collections import Counter

            labels_np = np.array(training_dataset.get_labels())
            class_counts = Counter(labels_np)

            total_samples = len(labels_np)

            # class weights -> less number of class -> more weightage
            class_weights = {cls: total_samples/count for cls , count in class_counts.items()}

            sample_weight = [class_weights[label] for label in labels_np]
            weight_tensor = torch.DoubleTensor(sample_weight)

            sampler = WeightedRandomSampler(weights=weight_tensor , num_samples=len(weight_tensor) , replacement=True)

            self.sampler = sampler

        self.train_loader = DataLoader(dataset=training_dataset ,sampler=sampler ,batch_size=self.batch_size , pin_memory=True , num_workers=self.num_workers)

    def get_loader(self):
        return self.train_loader
    



