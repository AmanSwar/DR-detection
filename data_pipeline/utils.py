import os
from tqdm import tqdm 
from typing import LiteralString

def add_path(img_name_list , path):
    for img_name in tqdm(img_name_list):
        complete_img_name = os.path.join(path , img_name)
        img_name = complete_img_name

    
class GradingDataset():

    def __init__(self , dataset_path):
        self.dataset_path = dataset_path
        self.__train_image = self.__get_img_list('train')
        self.__test_image = self.__get_img_list('test')
        self.__train_label = self.__get_labels("train")
        self.__test_label = self.__get_labels("test")
        self.__valid_image = self.__train_image[len(self.__train_image) * 8 :]
        self.__train_image = self.__train_image[:len(self.__train_image) * 8]
        self.__valid_label = self.__train_label[len(self.__train_label) * 8 : ]
        self.__train_label = self.__train_label[:len(self.__train_image) * 8]


    def __get_img_list(self , subset: LiteralString) -> list:
        pass
        
    def __get_labels(self , subset : LiteralString) -> list:
        pass

    def get_train_set(self) -> list:
        """
        Train images and labels
        """
        pass

    def get_valid_set(self) -> list:
        """
        Validation Images and labels
        """
        pass

    def get_test_set(self) -> list:
        """
        Test images and labels
        """
        pass