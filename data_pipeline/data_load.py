import os
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader
from typing import LiteralString
from tqdm import tqdm
import pandas as pd

from utils import add_path , GradingDataset


class EyepacsGradingDataset(GradingDataset):
    """
    Creates list of image paths and corresponding labels
    """
    def __init__(self , dataset_path="data/eyepacs"):
        super().__init__()
        
    def __get_img_list(self , subset: LiteralString ) -> list:
        # test vs train directory
        subset_dir = os.listdir(os.path.join(self.dataset_path , subset))
        # image dir for that particular subset
        img_dir = os.path.join(subset_dir , subset)
        # containes all the images name
        img_names = os.listdir(img_dir)

        #root dir 
        add_path(img_name_list=img_names , path=img_dir)

        # for imgs in tqdm(img_names):
        #     complete_img_path = os.path.join(img_dir , imgs)
        #     imgs = complete_img_path

        return img_names
        
    def __get_labels(self , subset: LiteralString) -> list:
        subset_dir = os.path.join(self.dataset_path , subset)
        subset_files = os.listdir(subset_dir)
        label_csv_path = None
        for file in subset_files:
            if file.endswith(".csv"):
                label_csv_path = os.path.join(subset_dir , file)

        
        labels_df = pd.read_csv(label_csv_path)
        label_dic = {labels_df['image'] : labels_df['level']}
        #get original image
        label_inorder = []
        if subset == "train":
            for img_name in self.train_image:
                label_inorder.append(label_dic[img_name])

        if subset == "test":
            for img_name in self.test_image:
                label_inorder.append(label_dic[img_name])

        return label_inorder

    def get_train_set(self) -> list:
        
        return self.__train_image , self.__train_label

    def get_test_set(self) -> list:
        
        return self.__test_image , self.__test_label
    
    def get_valid_set(self) ->list:
        return self.__valid_image , self.__valid_label


class AptosGradingDataset(GradingDataset):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def __get_img_list(self, subset):
        img_dir = os.path.join(self.dataset_path , f"{subset}_images")
        img_names = os.listdir(img_dir)

        add_path(img_name_list=img_names , path=img_dir)
        # for img_name in tqdm(img_names):
        #     # complete path to the image
        #     complete_img_name = os.path.join(img_dir , img_name)
        #     img_name = complete_img_name
        
        return img_names
    
    def __get_labels(self, subset):
        csv_file = None
        for file in os.listdir(self.dataset_path):
            if file == f"{subset}.csv":
                csv_file = file

        labels_df = pd.read_csv(csv_file)
        labels_dic = {labels_df['id_code'] : labels_df['diagnosis']}

        labels_inorder = []
        if subset == "train":
            for img_name in self.train_image:
                labels_dic.append(labels_dic[img_name])

        if subset == "test":
            for img_name in self.test_image:
                labels_inorder.append(labels_dic[img_name])

        return labels_inorder
    
    def get_test_set(self):
        return self.__test_image, self.__test_label
    
    def get_train_set(self):
        return self.__train_image , self.__train_label
    


class DdrGradingDataset(GradingDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        
    
    def __get_img_list(self, subset):
        grading_subset = os.path.join(self.dataset_path , "DR_grading")

        img_dir = os.path.join(grading_subset , subset)
        img_names_list = os.listdir(img_dir)

        add_path(img_name_list=img_names_list , path=img_dir)

        return img_names_list
    

    def __get_labels(self, subset):
        grading_subset = os.path.join(self.dataset_path , "DR_grading")
        files = os.listdir(grading_subset)

        labels_text = None
        for file in files:
            if file == f"{subset}.txt":
                labels_text = file
        col_names = ['imgs' , 'labels']
        labels_df = pd.read_csv(labels_text , sep=' ' , names=col_names)
        labels_dic = {labels_df['imgs'] : labels_df['labels']}

        labels_inorder = []

        if subset == "train":
            for img in tqdm(self.__train_image):
                labels_inorder.append(labels_dic[img])
        
        if subset == "test":
            for img in tqdm(self.__test_image):
                labels_inorder.append(labels_dic[img])

        












        



            


     
         

        
        

        





