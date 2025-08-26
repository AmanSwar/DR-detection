import os
from tqdm import tqdm 
from typing import LiteralString
import torch
def add_path(img_name_list: list , path: os.path):
    """
    Combines the complete path to each of the images in a image folder
    
    """
    for i in tqdm(range(len(img_name_list))):
        complete_img_name = os.path.join(path , img_name_list[i])
        if not os.path.exists(complete_img_name):
            print(f"Warning: Image not found: {complete_img_name}")
            continue
        img_name_list[i] = complete_img_name

    
class GradingDataset():

    def __init__(self , dataset_path):
        self.dataset_path = dataset_path
        self._train_image = self._get_img_list('train')
        self._test_image = self._get_img_list('test')
        self._train_label = self._get_labels("train")
        self._test_label = self._get_labels("test")

        self._train_len = int(len(self._train_image) * 0.8)
        self._valid_image = self._train_image[self._train_len: ]
        self._train_image = self._train_image[:self._train_len]
        self._valid_label = self._train_label[self._train_len : ]
        self._train_label = self._train_label[:self._train_len]
        print(self._train_image[:5])
        print(self._valid_image[:5])


    def _get_img_list(self , subset: LiteralString) -> list:
        """
        Function to get image list from data corresponding to subset
        args:
            subset (str) -> subset of dataset  , train , valid , test
        
        return:
            img_names (List) -> list of images 

        """
        pass
        
    def _get_labels(self , subset : LiteralString) -> list:
        pass

    def get_train_set(self) -> list:
        """
        Train images and labels
        """
        return self._train_image , self._train_label

    def get_valid_set(self) -> list:
        """
        Validation Images and labels
        """
        return self._valid_image , self._valid_label

    def get_test_set(self) -> list:
        """
        Test images and labels
        """
        return self._test_image , self._test_label
    

class SSLDataset:

    def __init__(self , dataset_path):

        self.dataset_path = dataset_path
        self._train_imgs = self._get_train_img()
        self._valid_imgs = self._get_valid_img()

    def _get_train_img(self):
        pass

    def _get_valid_img(self):
        pass

    def get_training(self):
        return self._train_imgs

    def get_validation(self):
        return self._valid_imgs

class MemoryEfficientBatchSampler:
    def __init__(
            self, 
            dataset, 
            batch_size, 
            shuffle=True
            ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gradient_accumulation_steps = 4
        
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))
        else:
            indices = torch.arange(len(self.dataset))
            
        mini_batch_size = self.batch_size // self.gradient_accumulation_steps
        for i in range(0, len(indices), mini_batch_size):
            yield indices[i:i + mini_batch_size]




