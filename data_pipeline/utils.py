import os
from tqdm import tqdm 
from typing import LiteralString

def add_path(img_name_list: list , path: os.path):
    """
    Combines the complete path to each of the images in a image folder
    
    """
    for i in tqdm(range(len(img_name_list))):
        complete_img_name = os.path.join(path , img_name_list[i])
        img_name_list[i] = complete_img_name

    
class GradingDataset():

    def __init__(self , dataset_path):
        self.dataset_path = dataset_path
        self._train_image = self._get_img_list('train')
        self._test_image = self._get_img_list('test')
        self._train_label = self._get_labels("train")
        self._test_label = self._get_labels("test")

        self._valid_image = self._train_image[len(self._train_image) * 8 :]
        self._train_image = self._train_image[:len(self._train_image) * 8]
        self._valid_label = self._train_label[len(self._train_label) * 8 : ]
        self._train_label = self._train_label[:len(self._train_image) * 8]


    def _get_img_list(self , subset: LiteralString) -> list:
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