
import torch
from torch.utils.data import Dataset , DataLoader , WeightedRandomSampler

#util import 
from data_pipeline.data_load import EyepacsGradingDataset , AptosGradingDataset , IdridGradingDataset , DdrGradingDataset , MessdrGradingDataset
from data_pipeline.data_load import EyepacsSSLDataset , AptosSSLDataset , IdridSSLDataset , DdrSSLDataset , MessdrSSLDataset
import random
from PIL import Image , ImageFile
from typing import Tuple , List

import numpy as np
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data_pipeline.data_aug import DinowregAug

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnitedTrainingDataset(Dataset):
    def __init__(self, *args, transformation=None, img_size=1024):
        self.args = args
        self.image_path = []
        self.labels = []
        self.dataset_counts = {}
        self.dataset_index = []
        
        self.transformation = transformation
        self.img_size = img_size
        
        for i , arg in enumerate(args):
            img_path, label = self.__getdata(arg)
            self.dataset_counts[arg] = len(img_path)
            # Filter out images with label 5
            filtered_img_path = [path for path, lbl in zip(img_path, label) if lbl != 5]
            filtered_label = [lbl for lbl in label if lbl != 5]
            self.image_path.extend(filtered_img_path)
            self.dataset_index.extend([i] * len(filtered_img_path))
            self.labels.extend(filtered_label)

        img_path_label_index_triple = list(zip(self.image_path, self.labels, self.dataset_index))
        random.shuffle(img_path_label_index_triple)
        random.shuffle(img_path_label_index_triple)
        self.image_path, self.labels, self.dataset_index = map(list, zip(*img_path_label_index_triple))
    
    def __getdata(self, dataset_name: str) -> Tuple[List[str], List[int]]:
        if dataset_name not in ["eyepacs", "aptos", "ddr", "idrid", "messdr"]:
            raise ValueError(f"Unknown dataset {dataset_name}")
        
        elif dataset_name.lower() == "eyepacs":
            eyepacs = EyepacsGradingDataset()
            eyepacs_train_img, eyepacs_train_labels = eyepacs.get_train_set()
            return eyepacs_train_img, eyepacs_train_labels

        elif dataset_name.lower() == "aptos":
            aptos = AptosGradingDataset()
            aptos_train_img, aptos_train_labels = aptos.get_train_set()
            return aptos_train_img, aptos_train_labels

        elif dataset_name.lower() == "ddr":
            ddr = DdrGradingDataset(root_dir="data/ddr")
            ddr_train_img, ddr_train_labels = ddr.get_train_set()
            return ddr_train_img, ddr_train_labels
        
        elif dataset_name.lower() == "idrid":
            idrid = IdridGradingDataset()
            idrid_train_img, idrid_train_labels = idrid.get_train_set()
            return idrid_train_img, idrid_train_labels
        
        elif dataset_name.lower() == "messdr":
            messdr = MessdrGradingDataset()
            messdr_train_img, messdr_train_label = messdr.get_train_set()
            return messdr_train_img, messdr_train_label

    def get_paths(self) -> List[str]:
        """To get combined path of all images of given dataset"""
        return self.image_path
    
    def get_dataset_counts(self):
        return self.dataset_counts
    
    def get_labels(self) -> List[int]:
        """Function to return labels"""
        return self.labels

    def get_num_class(self):
        unique_label = Counter(self.labels)
        return unique_label
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        label = self.labels[index]
        dataset_index = self.dataset_index[index]
        
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transformation is not None:
                trans_img = self.transformation(img)
                return trans_img, label , dataset_index
            return img, label , dataset_index
        except (IOError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading {img_path}: {str(e)}")



class UnitedValidationDataset(Dataset):
    def __init__(self, *args, transformation=None, img_size=1024):
        self.args = args
        self.image_path = []
        self.labels = []
        self.transformation = transformation
        self.img_size = img_size

        # Concatenating all datasets with filtering
        for arg in args:
            img_path, label = self.__getdata(arg)
            # Filter out images with label 5
            filtered_img_path = [path for path, lbl in zip(img_path, label) if lbl != 5]
            filtered_label = [lbl for lbl in label if lbl != 5]
            self.image_path.extend(filtered_img_path)
            self.labels.extend(filtered_label)

        # Logic for shuffling
        img_path_label_pair = list(zip(self.image_path, self.labels))
        random.shuffle(img_path_label_pair)
        self.image_path, self.labels = zip(*img_path_label_pair)
        print("shuffled")
        print("\n")

    def __getdata(self, dataset_name: str) -> list:
        if dataset_name == "eyepacs":
            eyepacs = EyepacsGradingDataset()
            eyepacs_valid_img, eyepacs_valid_labels = eyepacs.get_valid_set()
            return eyepacs_valid_img, eyepacs_valid_labels

        elif dataset_name == "aptos":
            aptos = AptosGradingDataset()
            aptos_valid_img, aptos_valid_labels = aptos.get_valid_set()
            return aptos_valid_img, aptos_valid_labels

        elif dataset_name == "ddr":
            ddr = DdrGradingDataset(root_dir="data/ddr")
            ddr_valid_img, ddr_valid_labels = ddr.get_valid_set()
            return ddr_valid_img, ddr_valid_labels
        
        elif dataset_name == "idrid":
            idrid = IdridGradingDataset()
            idrid_valid_img, idrid_valid_labels = idrid.get_valid_set()
            return idrid_valid_img, idrid_valid_labels
        
        elif dataset_name.lower() == "messdr":
            messdr = MessdrGradingDataset()
            messdr_valid_img, messdr_valid_label = messdr.get_valid_set()
            return messdr_valid_img, messdr_valid_label

    def get_paths(self) -> list:
        """To get combined path of all images of given dataset and corresponding labels"""
        return self.image_path

    def get_labels(self) -> List[int]:
        """Function to return labels"""
        return self.labels
    
    def get_num_class(self):
        unique_label = Counter(self.labels)
        return unique_label
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        label = self.labels[index]
        img = Image.open(img_path)
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        if self.transformation is not None:
            img = self.transformation(img)
        return img, label

     
        

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
            

            labels_np = np.array(training_dataset.get_labels())
            class_counts = Counter(labels_np)

            total_samples = len(labels_np)

            # class weights -> less number of class -> more weightage
            class_weights = {cls: total_samples/count for cls , count in class_counts.items()}
            # class_weights = torch.tensor([total_samples / class_counts[cls] for cls in sorted(class_counts.keys())], dtype=torch.float).to(device)

            sample_weight = [class_weights[label] for label in labels_np]
            weight_tensor = torch.DoubleTensor(sample_weight)

            sampler = WeightedRandomSampler(weights=weight_tensor , num_samples=len(weight_tensor) , replacement=True)

            self.sampler = sampler

        self.train_loader = DataLoader(dataset=training_dataset ,sampler=sampler ,batch_size=self.batch_size , pin_memory=True , num_workers=self.num_workers)

    def get_loader(self):
        return self.train_loader
    
class UniformValidDataloader:

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
        validation_ds = UnitedValidationDataset(*self.dataset_names , transformation=self.transformation)
        
        if sampler:
            

            labels_np = np.array(validation_ds.get_labels())
            class_counts = Counter(labels_np)

            total_samples = len(labels_np)

            # class weights -> less number of class -> more weightage
            class_weights = {cls: total_samples/count for cls , count in class_counts.items()}

            sample_weight = [class_weights[label] for label in labels_np]
            weight_tensor = torch.DoubleTensor(sample_weight)

            sampler = WeightedRandomSampler(weights=weight_tensor , num_samples=len(weight_tensor) , replacement=True)

            self.sampler = sampler

        self.train_loader = DataLoader(dataset=validation_ds ,sampler=sampler ,batch_size=self.batch_size , pin_memory=True , num_workers=self.num_workers)

    def get_loader(self):
        return self.train_loader

class UnitedSSLTrainingDataset(Dataset):

    def __init__(self , *args ,transformation=None , img_size=1024):
        self.args = args
        self.image_path = []
        self.dataset_counts = {}
        self.transformation = transformation
        self.img_size= img_size

        for arg in args:
            img_path = self.__getdata(arg)
            self.image_path.extend(img_path)
            self.dataset_counts[arg] = len(img_path)
            

        random.shuffle(self.image_path)


    def __getdata(self , dataset_name):

        if dataset_name not in ["eyepacs" , "aptos" ,"ddr" , "idrid" , "messdr"]:
            raise ValueError(f"Unknown dataset {dataset_name}")
        
        elif dataset_name.lower() == "eyepacs":
            eyepacs = EyepacsSSLDataset()
            eyepacs_train_img = eyepacs.get_training()
            
            return eyepacs_train_img 

        elif dataset_name.lower() == "aptos":
            aptos = AptosSSLDataset()
            aptos_train_img = aptos.get_training()
            
            return aptos_train_img 
           

        elif dataset_name.lower() == "ddr":
            ddr = DdrSSLDataset(dataset_path="data/ddr")
            ddr_train_img = ddr.get_training()
            
            return ddr_train_img 
        
        elif dataset_name.lower() == "idrid":
            idrid = IdridSSLDataset()
            idrid_train_img  = idrid.get_training()
            return idrid_train_img
        
        elif dataset_name.lower() == "messdr":
            messdr = MessdrSSLDataset()
            messdr_train_img = messdr.get_training()
            return messdr_train_img
        
    def get_paths(self) ->  List[float]:
        """
        To get combined path of all images of given dataset and corresponding labels
        """

        return self.image_path
    def get_dataset_counts(self):
        return self.dataset_counts
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        img_path = self.image_path[index]
        

        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # img = np.array(img)
            
            if self.transformation is not None:
                trans_img = self.transformation(img)
                return trans_img

            return img
        except (IOError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")
        except OSError as e:
            print(f"Warning: Failed to load image {img_path}. Skipping. Error: {e}")
            return None
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading {img_path}: {str(e)}")
        
        


class UnitedSSLValidationDataset(Dataset):
    def __init__(self , *args ,transformation=None , img_size=1024):
        self.args = args
        self.image_path = []
        self.transformation = transformation
        self.img_size= img_size

        for arg in args:
            img_path = self.__getdata(arg)
            self.image_path.extend(img_path)
            

        random.shuffle(self.image_path)


    def __getdata(self , dataset_name):

        if dataset_name not in ["eyepacs" , "aptos" ,"ddr" , "idrid" , "messdr"]:
            raise ValueError(f"Unknown dataset {dataset_name}")
        
        elif dataset_name.lower() == "eyepacs":
            eyepacs = EyepacsSSLDataset()
            eyepacs_train_img = eyepacs.get_validation()
            
            return eyepacs_train_img 

        elif dataset_name.lower() == "aptos":
            aptos = AptosSSLDataset()
            aptos_train_img = aptos.get_validation()
            
            return aptos_train_img 
           

        elif dataset_name.lower() == "ddr":
            ddr = DdrSSLDataset(dataset_path="data/ddr")
            ddr_train_img = ddr.get_validation()
            
            return ddr_train_img 
        
        elif dataset_name.lower() == "idrid":
            idrid = IdridSSLDataset()
            idrid_train_img  = idrid.get_training()
            return idrid_train_img
        
        elif dataset_name.lower() == "messdr":
            messdr = MessdrSSLDataset()
            messdr_train_img = messdr.get_validation()
            return messdr_train_img
        
    def get_paths(self) ->  List[float]:
        """
        To get combined path of all images of given dataset and corresponding labels
        """

        return self.image_path
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        img_path = self.image_path[index]
        

        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # img = np.array(img)
            
            if self.transformation is not None:
                trans_img = self.transformation(img)
                return trans_img

            return img
        except (IOError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")
        except OSError as e:
            print(f"Warning: Failed to load image {img_path}. Skipping. Error: {e}")
            return None
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading {img_path}: {str(e)}")
        


from torch.utils.data.dataloader import default_collate
def custom_collate_fn(batch):
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        return []
        
    return default_collate(filtered_batch)


class SSLTrainLoader:

    def __init__(
            self,
            dataset_names,
            transformation,
            batch_size,
            num_work,
    ):
        
        self.dataset_names = dataset_names
        self.transformation = transformation
        self.batch_size = batch_size
        self.num_workers = num_work
        training_dataset = UnitedSSLTrainingDataset(*self.dataset_names , transformation=self.transformation)
        
        self.train_loader = DataLoader(
            dataset=training_dataset ,
            batch_size=self.batch_size , 
            pin_memory=True , 
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn
            )
    
    def get_loader(self):
        return self.train_loader
    
from torch.utils.data import DataLoader, DistributedSampler

class DistSSLTrainLoader:
    def __init__(self, dataset_names, img_size, batch_size, num_work, world_size=2, rank=0):
        self.dataset_names = dataset_names
        self.transformation = DinowregAug(img_size=img_size)
        self.batch_size = batch_size
        self.num_workers = num_work

        # Create the dataset
        training_dataset = UnitedSSLTrainingDataset(*self.dataset_names, transformation=self.transformation)
        
        # Use DistributedSampler
        self.sampler = DistributedSampler(training_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        # Create DataLoader
        self.train_loader = DataLoader(
            dataset=training_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=self.sampler  # Use DistributedSampler
        )

    def get_loader(self):
        return self.train_loader

    def set_epoch(self, epoch):
        """Ensure data is shuffled differently each epoch."""
        self.sampler.set_epoch(epoch)



class SSLValidLoader:

    def __init__(
            self,
            dataset_names,
            transformation,
            batch_size,
            num_work,
    ):
        
        self.dataset_names = dataset_names
        self.transformation = transformation
        self.batch_size = batch_size
        self.num_workers = num_work
        validation_dataset = UnitedSSLValidationDataset(
            *self.dataset_names,
            transformation=self.transformation
        )

        self.valid_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def get_loader(self):
        return self.valid_loader
    
class DistSSLValidLoader:
    def __init__(self, dataset_names, img_size, batch_size, num_work, world_size=1, rank=0):
        self.dataset_names = dataset_names
        self.transformation = DinowregAug(img_size=img_size)
        self.batch_size = batch_size
        self.num_workers = num_work

        validation_dataset = UnitedSSLValidationDataset(
            *self.dataset_names,
            transformation=self.transformation
        )

        self.sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        self.valid_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=self.sampler
        )

    def get_loader(self):
        return self.valid_loader

    def set_epoch(self, epoch):
        
        self.sampler.set_epoch(epoch)

# --------------------------------------------------------------------------------------------------------------


class UniformTestDataset(Dataset):

    def __init__(self):
        pass


class UniformTestLoader:

    def __init__(self):
        
        pass

    def get_loader(self):

        pass