import os
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader
from typing import LiteralString
from tqdm import tqdm
import pandas as pd

from data_pipeline.utils import add_path , GradingDataset

base_path = "/"
class EyepacsGradingDataset(GradingDataset):
    """
    Creates list of image paths and corresponding labels
    """
    def __init__(self , dataset_path="data/eyepacs"):
        super().__init__(dataset_path)
        
    def _get_img_list(self , subset: str ) -> list:
        # test vs train directory
        subset_dir = (os.path.join(self.dataset_path , subset))
        # image dir for that particular subset
        img_dir = os.path.join(subset_dir , subset)
        # containes all the images name
        img_names = os.listdir(img_dir)

        #root dir 
        add_path(img_name_list=img_names , path=img_dir)

        return img_names
        
    def _get_labels(self , subset: LiteralString) -> list:
        if subset =="test":
            return []
        subset_dir = os.path.join(self.dataset_path , subset)
        subset_files = os.listdir(subset_dir)
        label_csv_path = None
        for file in subset_files:
            if file.endswith(".csv"):
                label_csv_path = os.path.join(subset_dir , file)

        if label_csv_path == None:
            raise Exception(f"{subset} labels not found")
            
            
        labels_df = pd.read_csv(os.path.join(label_csv_path))
        label_dic = {img_name : label for img_name , label in zip(labels_df['image']  , labels_df['level'])}
        #get original image
        label_inorder = []
        if subset == "train":
            for img_name in self._train_image:
                label_inorder.append(label_dic[img_name.split('/')[-1].rstrip('.jpeg')])

        if subset == "test":
            for img_name in self._test_image:
                label_inorder.append(label_dic[img_name.split('/')[-1].rstrip('.jpeg')])

        return label_inorder

    def get_train_set(self) -> list:
        
        return self._train_image , self._train_label

    def get_test_set(self) -> list:
        
        return self._test_image , self._test_label
    
    def get_valid_set(self) ->list:
        return self._valid_image , self._valid_label


class AptosGradingDataset(GradingDataset):

    def __init__(self, dataset_path="data/aptos"):
        super().__init__(dataset_path)

    def _get_img_list(self, subset):
        img_dir = os.path.join(self.dataset_path , f"{subset}_images")
        img_names = os.listdir(img_dir)
        add_path(img_name_list=img_names , path=img_dir)
        return img_names
    
    def _get_labels(self, subset):
        if subset == "test":
            return []
        csv_file = None
        for file in os.listdir(self.dataset_path):
            if file == f"{subset}.csv":
                csv_file = os.path.join(self.dataset_path , file)

        labels_df = pd.read_csv(csv_file)
        labels_dic = {img_name : labels for img_name , labels in zip(labels_df['id_code'],labels_df['diagnosis'])}

        labels_inorder = []
        if subset == "train":
            for img_name in self._train_image:
                labels_inorder.append(labels_dic[img_name.split('/')[-1].rstrip('.png')])

        if subset == "test":
            for img_name in self._test_image:
                labels_inorder.append(labels_dic[img_name.split('/')[-1].rstrip('.png')])

        return labels_inorder
    
    def get_test_set(self):
        return self._test_image, self._test_label
    
    def get_train_set(self):
        return self._train_image , self._train_label
    
    def get_valid_set(self):
        return self._valid_image , self._valid_label



class DdrGradingDataset(GradingDataset):
    def __init__(self, dataset_path="data/ddr/DDR-dataset"):
        super().__init__(dataset_path)
        self._valid_image = self._get_img_list("valid")
        self._valid_label = self._get_labels("valid")
    
    def _get_img_list(self, subset):
        grading_subset = os.path.join(self.dataset_path , "DR_grading")

        img_dir = os.path.join(grading_subset , subset)
        img_names_list = os.listdir(img_dir)

        add_path(img_name_list=img_names_list , path=img_dir)

        return img_names_list
    

    def _get_labels(self, subset):
        if subset == "test":
            return []
        grading_subset = os.path.join(self.dataset_path , "DR_grading")
        files = os.listdir(grading_subset)

        labels_text = base_path
        for file in files:
            if file == f"{subset}.txt":
                labels_text += os.path.join(grading_subset , file)
        col_names = ['imgs' , 'labels']
        labels_df = pd.read_csv(labels_text , sep=' ' , names=col_names)
        labels_dic = {img_name : label for img_name , label in zip(labels_df['imgs'],labels_df['labels'])}

        labels_inorder = []

        if subset == "train":
            for img in tqdm(self._train_image):
                labels_inorder.append(labels_dic[img.split('/')[-1]])
        
        elif subset == "test":
            for img in tqdm(self._test_image):
                labels_inorder.append(labels_dic[img.split('/')[-1]])

        elif subset == "valid":
            for img in tqdm(self._valid_image):
                labels_inorder.append(labels_dic[img.split('/')[-1]])

        
        return labels_inorder
        
    


class IdridGradingDataset(GradingDataset):

    def __init__(self, dataset_path="data/idrid"):
        super().__init__(dataset_path)

    def _get_img_list(self, subset):
        grading_sub_path = os.path.join(self.dataset_path , "B. Disease Grading")
        img_sub_path = os.path.join(grading_sub_path, "1. Original Images")

        if subset == "train":
            img_dir = os.path.join(img_sub_path , "a. Training Set")
            all_img_name = os.listdir(img_dir)

            add_path(img_name_list=all_img_name , path=img_dir)
        
            return all_img_name
        
        elif subset == "test":

            img_dir = os.path.join(img_sub_path , "b. Testing Set")
            all_img_name = os.listdir(img_dir)

            add_path(img_name_list=all_img_name , path=img_dir)

            return all_img_name
        
    def _get_labels(self, subset):
        
        grading_sub_path = os.path.join(self.dataset_path , "B. Disease Grading")
        labels_dir = os.path.join(grading_sub_path , "2. Groundtruths")
        lables_file = base_path
        for file in os.listdir(labels_dir):
            print(file)
            if f"{subset}ing" in file.lower():
                lables_file += os.path.join(labels_dir , file)
               
        labels_df = pd.read_csv(lables_file)
        labels_dic = {img_name : label for img_name , label in zip(labels_df["Image name"] ,labels_df["Retinopathy grade"])}

        labels_inorder = []

        if subset == "train":
            for img in self._train_image:
                labels_inorder.append(labels_dic[img.split('/')[-1].rstrip('.jpg')])

        elif subset == "test":
            for img in self._test_image:
                labels_inorder.append(labels_dic[img.split('/')[-1].rstrip('.jpg')])

        return labels_inorder
    



        



        

        










        



            


     
         

        
        

        





