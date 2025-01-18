import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset , DataLoader
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os


class DINOAugmentation:
    def __init__(self ,
        img_size: int = 1024,
        global_crop: int = 900 ,
        local_crop: int = 400 ,
        n_crops_local : int = 7
        ):
        self.n_crops_local = n_crops_local
        self.base_transform = A.Compose([
            A.Resize(img_size , img_size),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        ])
        self.global_transform = A.Compose(
            [
                A.RandomResizedCrop(size=(global_crop , global_crop)),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2,
                            contrast_limit=0.2,
                            p=1
                            ),
                        A.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1,
                            p=1
                        ),
                    ] , p=0.8),
                
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3,7) , p=1),
                        A.GaussNoise(p=1)
                    ] , p=0.5
                ),
                ToTensorV2(),
            ]
        )


        self.local_transform = A.Compose(
            [
                A.RandomResizedCrop(size=(local_crop , local_crop)),
                A.HorizontalFlip(p=0.5),

                A.RandomBrightnessContrast(brightness_limit=0.1 , contrast_limit=0.1, p=0.7),
                ToTensorV2(),
            ]
        )


    def __call__(self , image):
        views = []

        if isinstance(image , Image.Image):
            image = np.array(image)


        # base transforms -> apply clahe
        prep_basic = self.base_transform(image=image)['image']

        # global transformation
        for _ in range(2):
            aug = self.global_transform(prep_basic)
            views.append(aug['image'])

        # local transformation
        for _ in range(self.n_crops_local):
            aug = self.local_transform(prep_basic)
            views.append(aug['image'])


        return views


class IJEPAAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5),
            ], p=0.3),
            A.OneOf([
               
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.2),
         
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                A.UnsharpMask(blur_limit=(3, 7), p=0.5),
            ], p=0.3),
       
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=5,
                p=0.3
            ),
    
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                mask_fill_value=0,
                p=0.2
            ),
        ])

    def __call__(self, image):
        return self.transform(image=image)['image']


