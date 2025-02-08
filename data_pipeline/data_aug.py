import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset , DataLoader
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import random



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
            A.Resize(1024 , 1024),
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
            ToTensorV2()
        ])


    def __call__(self, image):
        trans_img =  self.transform(image=image)['image']
        return trans_img.float() / 255.0

class RandomRotate90:
    def __call__(self, img):
        if random.random() < 0.5:
            k = random.randint(0, 3)
            return TF.rotate(img, 90 * k)
        return img

# Custom Transform: Add Gaussian Noise (expects a tensor)
class AddGaussianNoise:
    def __init__(self, mean=0., std_range=(0.01, 0.1)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor):
        std = random.uniform(*self.std_range)
        noise = torch.randn(tensor.size()) * std + self.mean
        return tensor + noise

# Custom Transform: Random Gaussian Blur (expects a tensor)
class RandomGaussianBlur:
    def __init__(self, kernel_sizes=[3, 5, 7], sigma_range=(0.1, 2.0)):
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range

    def __call__(self, tensor):
        if random.random() < 0.5:
            kernel_size = random.choice(self.kernel_sizes)
            sigma = random.uniform(*self.sigma_range)
            # TF.gaussian_blur expects kernel_size to be odd and sigma as a list or tuple
            return TF.gaussian_blur(tensor, kernel_size, [sigma, sigma])
        return tensor

# Custom Transform: Coarse Dropout (expects a tensor)
class CoarseDropout:
    def __init__(self, num_holes_range=(1, 8), hole_height_range=(12, 32), 
                 hole_width_range=(12, 32), fill=0, p=0.5):
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        self.fill = fill
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            num_holes = random.randint(*self.num_holes_range)
            _, height, width = tensor.shape
            for _ in range(num_holes):
                hole_h = random.randint(*self.hole_height_range)
                hole_w = random.randint(*self.hole_width_range)
                # Ensure the hole fits within the image dimensions
                if height - hole_h <= 0 or width - hole_w <= 0:
                    continue
                y = random.randint(0, height - hole_h)
                x = random.randint(0, width - hole_w)
                tensor[:, y:y+hole_h, x:x+hole_w] = self.fill
        return tensor

# Custom Transform: Random Choice with Probabilities
class RandomChoiceWithProbs:
    def __init__(self, transforms, probs):
        self.transforms = transforms
        self.probs = probs

    def __call__(self, x):
        chosen = random.choices(self.transforms, weights=self.probs, k=1)[0]
        return chosen(x)

# Main Augmentation Module
class RetAug:
    def __init__(self, img_size=512):
        self.img_size = img_size

        # Base augmentations on the PIL image (geometric, flip, rotation)
        self.base_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotate90()
        ])

        # View 1: basic augmentations then conversion to tensor.
        self.transform1 = transforms.Compose([
            self.base_transforms,
            transforms.ToTensor()
        ])

        # Lesion simulation – note: these transforms operate on PIL images.
        self.lesion_sim = transforms.RandomApply([
            RandomChoiceWithProbs(
                transforms=[
                    # Make sure your torchvision version supports these on PIL images.
                    transforms.ElasticTransform(alpha=50.0, sigma=7.0),
                    transforms.RandomResizedCrop(
                        size=(img_size, img_size),
                        scale=(16 / img_size, 32 / img_size),
                        ratio=(1.0, 1.0)
                    ),
                    transforms.RandomPerspective(distortion_scale=0.5, p=1.0)
                ],
                probs=[0.3, 0.3, 0.4]
            )
        ], p=0.5)

        # View 2: first apply base transforms, then lesion simulation (still on PIL),
        # then convert to tensor and apply additional pixel-level augmentations.
        self.transform2 = transforms.Compose([
            self.base_transforms,
            self.lesion_sim,        # Apply lesion simulation on PIL image.
            transforms.ToTensor(),  # Convert to tensor.
            transforms.RandomApply([
                transforms.RandomChoice([
                    RandomGaussianBlur(),
                    AddGaussianNoise()
                ])
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ], p=0.8),
            CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(12, 32),
                hole_width_range=(12, 32),
                fill=0,
                p=0.5
            )
        ])

    def __call__(self, image):
        # Convert input to a PIL Image if it isn’t already.
        if not isinstance(image, Image.Image):
            # If it's a numpy array or something else, force it to be a PIL image.
            image = Image.fromarray(np.array(image).astype('uint8'))

        view1 = self.transform1(image)
        view2 = self.transform2(image)
        return view1, view2