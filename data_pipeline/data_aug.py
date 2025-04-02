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
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2
from PIL import ImageFilter, ImageEnhance
import math

#--------------------------------------------------------------------------------------------
# Custom CLAHE transform (applied in LAB color space)
class CLAHE(object):
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # Ensure input is a PIL Image in RGB
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img_np = np.array(img)
        # Convert from RGB to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE to the L-channel
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        # Convert back to RGB
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_clahe)

# Custom RandomRotate90 (rotates by a random 90° multiple)
class RandomRotate90(object):
    def __call__(self, img):
        # Choose a random angle from {90, 180, 270}
        angle = random.choice([90, 180, 270])
        return TF.rotate(img, angle)

# Custom Gaussian blur that randomly chooses a kernel size
class RandomGaussianBlur_dino(object):
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        kernel_size = random.choice([3, 5, 7])
        # torchvision.transforms.GaussianBlur requires kernel_size to be odd and a number or tuple
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=self.sigma)(img)

# Custom GaussNoise transform: adds noise to a PIL image
class GaussNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Convert PIL image to numpy float image in [0,1]
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        np_img = np_img + noise
        np_img = np.clip(np_img, 0, 1)
        np_img = (np_img * 255).astype(np.uint8)
        return Image.fromarray(np_img)


class DINOAugmentation:
    def __init__(self,
                 img_size: int = 224,
                 global_crop: int = 224,
                 local_crop: int = 96,
                 n_crops_local: int = 6):
        self.n_crops_local = n_crops_local

        # Base transform: resize then apply CLAHE
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            CLAHE(clip_limit=3.0, tile_grid_size=(8, 8)),
        ])

        # Global augmentation: random crop, flip, rotate, color jitter, blur/noise, then tensor conversion.
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop((global_crop, global_crop)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([RandomRotate90()], p=0.5),
            transforms.RandomApply([
                transforms.RandomChoice([
                    # Option 1: brightness & contrast jitter (mimicking RandomBrightnessContrast)
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    # Option 2: ColorJitter with saturation and hue as well
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                ])
            ], p=0.8),
            transforms.RandomApply([
                transforms.RandomChoice([
                    RandomGaussianBlur_dino(sigma=(0.1, 2.0)),
                    GaussNoise(mean=0.0, std=0.1)
                ])
            ], p=0.5),
            transforms.ToTensor(),
        ])

        # Local augmentation: random crop, flip, slight brightness/contrast jitter, then tensor conversion.
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop((local_crop, local_crop)),
            transforms.Resize((global_crop, global_crop)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ], p=0.7),
            transforms.ToTensor(),
        ])

    def __call__(self, image):
        # If the image is a numpy array, convert it to a PIL Image.
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Apply the base transform (resize + CLAHE)
        prep_basic = self.base_transform(image)

        views = []
        # Create two global views
        for _ in range(2):
            views.append(self.global_transform(prep_basic))
        # Create n local views
        for _ in range(self.n_crops_local):
            views.append(self.local_transform(prep_basic))

        return views

#--------------------------------------------------------------------------------------------

class GaussianNoiseTransform_ijepa:
    def __init__(self, var_limit=(10.0, 50.0)):
        self.var_limit = var_limit

    def __call__(self, img):
        var = random.uniform(*self.var_limit)
        std = math.sqrt(var) / 255.0
        return img + torch.randn_like(img) * std

class MultiplicativeNoiseTransform_ijepa:
    def __init__(self, multiplier=(0.95, 1.05)):
        self.multiplier = multiplier

    def __call__(self, img):
        return img * random.uniform(*self.multiplier)

class MedianBlurTransform_ijepa:
    def __init__(self, blur_limit=3):
        self.blur_limit = blur_limit

    def __call__(self, img):
        img_pil = TF.to_pil_image(img)
        img_pil = img_pil.filter(ImageFilter.MedianFilter(self.blur_limit))
        return TF.to_tensor(img_pil)

class SharpenTransform_ijepa:
    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.0)):
        self.alpha = alpha
        self.lightness = lightness

    def __call__(self, img):
        alpha = random.uniform(*self.alpha)
        lightness = random.uniform(*self.lightness)
        img_pil = TF.to_pil_image(img)
        enhancer = ImageEnhance.Sharpness(img_pil)
        factor = lightness + alpha * (lightness - 1.0)
        return TF.to_tensor(enhancer.enhance(factor))

class UnsharpMaskTransform_ijepa:
    def __init__(self, blur_limit=(3, 7)):
        self.blur_limit = blur_limit

    def __call__(self, img):
        radius = random.randint(*self.blur_limit)
        img_pil = TF.to_pil_image(img)
        img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=150, threshold=3))
        return TF.to_tensor(img_pil)

class CoarseDropout_ijepa:
    def __init__(self, max_holes=8, max_h=32, max_w=32, min_holes=1, min_h=8, min_w=8, fill_value=0, p=0.2):
        self.max_holes = max_holes
        self.max_h = max_h
        self.max_w = max_w
        self.min_holes = min_holes
        self.min_h = min_h
        self.min_w = min_w
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            C, H, W = img.shape
            num_holes = random.randint(self.min_holes, self.max_holes)
            for _ in range(num_holes):
                h = random.randint(self.min_h, self.max_h)
                w = random.randint(self.min_w, self.max_w)
                y = random.randint(0, H - h)
                x = random.randint(0, W - w)
                img[:, y:y+h, x:x+w] = self.fill_value
        return img

class IJEPAAugmentation:
    def __init__(self , img_size):
        self.pil_transforms = transforms.Compose([
            transforms.Resize(size=(img_size , img_size)),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.1, contrast=0.1)],
            #     p=0.5
            # ),
        ])
        self.tensor_transforms = transforms.Compose([   
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    # MedianBlurTransform_ijepa(3)
                ])
            ], p=0.2),
            transforms.RandomApply([
                transforms.RandomChoice([
                    SharpenTransform_ijepa((0.2, 0.5), (0.5, 1.0)),
                    UnsharpMaskTransform_ijepa((3, 7))
                ])
            ], p=0.3),
            CoarseDropout_ijepa(p=0.2)
        ])

    def __call__(self, image):
        trans_image = self.pil_transforms(image)

        trans_image_1 = TF.to_tensor(trans_image)
        trans_image_2 = self.tensor_transforms(trans_image_1)
        return trans_image_2.float()
    

#--------------------------------------------------------------------------------------------

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
class CoarseDropout_ibot:
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
class IbotRetAug:
    def __init__(self, img_size=512):
        self.img_size = img_size

        self.base_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotate90()
        ])

        
        self.transform1 = transforms.Compose([
            self.base_transforms,
            transforms.ToTensor()
        ])

     
        self.lesion_sim = transforms.RandomApply([
            RandomChoiceWithProbs(
                transforms=[
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
            CoarseDropout_ibot(
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

#--------------------------------------------------------------------------------------------

class RandomRotate90(object):
    """Rotate the PIL image by a random multiple of 90 degrees with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # Choose 90, 180, or 270 degrees (using transpose to avoid interpolation)
            k = random.choice([1, 2, 3])
            for _ in range(k):
                img = img.transpose(Image.ROTATE_90)
        return img

class GaussianBlurTransform(object):
    """
    Apply Gaussian blur with a randomly chosen kernel size.
    The kernel size is chosen as an odd integer between blur_limit[0] and blur_limit[1].
    """
    def __init__(self, blur_limit=(3, 7)):
        self.blur_limit = blur_limit

    def __call__(self, img):
        # Choose an odd kernel size
        possible_kernels = [k for k in range(self.blur_limit[0], self.blur_limit[1]+1) if k % 2 == 1]
        kernel_size = random.choice(possible_kernels)
        # Choose a random sigma (you can tune this range)
        sigma = random.uniform(0.1, 2.0)
        # You can either use the built-in torchvision GaussianBlur (if available)
        # or use a PIL filter:
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        # Alternatively, if you prefer torchvision’s transform (requires kernel_size):
        # return transforms.GaussianBlur(kernel_size=kernel_size, sigma=(sigma, sigma))(img)

class GaussNoiseTransform(object):
    """
    Add random Gaussian noise.
    The variance is randomly chosen from var_limit.
    """
    def __init__(self, var_limit=(10.0, 50.0)):
        self.var_limit = var_limit

    def __call__(self, img):
        # Convert PIL image to numpy array
        np_img = np.array(img).astype(np.float32)
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        std = var ** 0.5
        noise = np.random.normal(0, std, np_img.shape)
        np_img = np_img + noise
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

class ElasticTransform(object):
    """
    Apply an elastic deformation on the image.
    (This implementation uses OpenCV and SciPy.)
    """
    def __init__(self, alpha=50, sigma=7, alpha_affine=10):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine

    def __call__(self, img):
        np_img = np.array(img)
        shape = np_img.shape[:2]

        # Random affine
        center_square = np.float32(shape) // 2
        square_size = min(shape) // 3
        pts1 = np.float32([
            [center_square[1] + square_size, center_square[0] + square_size],
            [center_square[1] - square_size, center_square[0] + square_size],
            [center_square[1], center_square[0] - square_size]
        ])
        pts2 = pts1 + np.random.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        np_img = cv2.warpAffine(np_img, M, (shape[1], shape[0]), borderMode=cv2.BORDER_REFLECT_101)

        # Elastic deformation
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        # map_coordinates expects each channel separately if image is multichannel
        if np_img.ndim == 3:
            channels = []
            for d in range(np_img.shape[2]):
                channel = map_coordinates(np_img[..., d], indices, order=1, mode='reflect').reshape(shape)
                channels.append(channel)
            np_img = np.stack(channels, axis=-1)
        else:
            np_img = map_coordinates(np_img, indices, order=1, mode='reflect').reshape(shape)

        return Image.fromarray(np_img.astype(np.uint8))

class OpticalDistortion(object):
    """
    Apply optical distortion to the image.
    """
    def __init__(self, distort_limit=0.5, shift_limit=0.5):
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit

    def __call__(self, img):
        np_img = np.array(img)
        height, width = np_img.shape[:2]

        # Create meshgrid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Determine distortion factors
        center_x = width / 2
        center_y = height / 2
        # shift factors
        shift_x = random.uniform(-self.shift_limit, self.shift_limit) * width
        shift_y = random.uniform(-self.shift_limit, self.shift_limit) * height
        # distort factors
        distort_x = random.uniform(-self.distort_limit, self.distort_limit)
        distort_y = random.uniform(-self.distort_limit, self.distort_limit)

        # Apply distortions: shift pixels proportionally to their distance from center
        x_distort = x + (x - center_x) * distort_x + shift_x
        y_distort = y + (y - center_y) * distort_y + shift_y

        map_x = x_distort.astype(np.float32)
        map_y = y_distort.astype(np.float32)
        np_img = cv2.remap(np_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(np_img)

class RandomSizedCrop(object):
    """
    Randomly crop a square patch with a height randomly chosen between
    min_max_height[0] and min_max_height[1], then resize it to the given size.
    """
    def __init__(self, min_max_height=(32, 64), size=(1024, 1024)):
        self.min_max_height = min_max_height
        self.size = size

    def __call__(self, img):
        width, height = img.size  # PIL gives (width, height)
        crop_size = random.randint(self.min_max_height[0], min(self.min_max_height[1], height, width))
        if width < crop_size or height < crop_size:
            return img
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        img_cropped = img.crop((left, top, left + crop_size, top + crop_size))
        return img_cropped.resize(self.size, Image.BILINEAR)

class CoarseDropout(object):
    """
    Randomly set rectangular regions in the image to a constant fill value.
    """
    def __init__(self, max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        np_img = np.array(img)
        h, w = np_img.shape[:2]
        n_holes = random.randint(1, self.max_holes)
        for _ in range(n_holes):
            hole_height = random.randint(1, self.max_height)
            hole_width = random.randint(1, self.max_width)
            y = random.randint(0, h - hole_height)
            x = random.randint(0, w - hole_width)
            if np_img.ndim == 3:
                np_img[y:y+hole_height, x:x+hole_width, :] = self.fill_value
            else:
                np_img[y:y+hole_height, x:x+hole_width] = self.fill_value
        return Image.fromarray(np_img)



class DinowregAug:
    def __init__(self, img_size=1024):
        # View 1 pipeline (mimicking the Albumentations view1)
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # CLAHE(clip_limit=3.0, tile_grid_size=(8, 8)),
        ])
        self.view1_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # RandomRotate90(p=0.5),
            # CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.RandomApply([
                transforms.RandomChoice([
                    GaussianBlurTransform(blur_limit=(3, 7)),
                    # GaussNoiseTransform(var_limit=(10.0, 50.0))
                ])
            ], p=0.4),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                #    saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #  std=[0.229, 0.224, 0.225]),
        ])

        # View 2 pipeline (lesion-focused augmentation)
        self.view2_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            # CLAHE(clip_limit=3.0, tile_grid_size=(8, 8)),
            transforms.RandomApply([
                transforms.RandomChoice([
                    # ElasticTransform(alpha=50, sigma=7, alpha_affine=10),
                    # OpticalDistortion(distort_limit=0.5, shift_limit=0.5),
                    RandomSizedCrop(min_max_height=(32, 64), size=(img_size, img_size))
                ])
            ], p=0.5),
            CoarseDropout(max_holes=8, max_height=32, max_width=32,
                          fill_value=0, p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image = transforms.ToPILImage()(image)
            else:
                image = transforms.ToPILImage()(image)

        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        view1 = self.view1_transform(image)
        view2 = self.view2_transform(image)
        return view1, view2
    

#--------------------------------------------------------------------------------------------

scl_trans = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # RandomRotate90(p=0.5),
        # CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.ToTensor(),
    ]
)

#-----------------------------------------------------------------------------------------
def default_simclr_transform():
    return transforms.Compose([
        transforms.ToPILImage(),  # convert tensor image (C x H x W) to PIL Image
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])


class SimCLRAug:

    def __init__(self , img_size):

        self.base_trans = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])


    def __call__(self , img):

        view1 = self.base_trans(img)
        view2 = self.base_trans(img)

        return view1 , view2
    

# -------------------------------------------------------------------------------------

# class MoCoAug:

#     def __init__(self , img_size):

#         self.base_trans = transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#                 CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor()
#             ]
#         )

#     def __call__(self , image):
#         im_q = self.base_trans(image)
#         im_k = self.base_trans(image)

#         return im_q , im_k
    

class MoCoAug:
    def __init__(self, img_size=224):
        # Common transformations
        self.resize_crop = transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0))
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.to_tensor = transforms.ToTensor()
        self.clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
        
        # Color augmentations
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
        )
        self.grayscale = transforms.RandomGrayscale(p=0.2)
        
        # Blur augmentations
        self.gaussian_blur = transforms.GaussianBlur(
            kernel_size=23, sigma=(0.1, 2.0)
        )
        
        # Rotation for fundus images (preserves circular nature)
        self.rotation = transforms.RandomRotation(degrees=180)
        
        # Additional fundus-specific augmentations
        self.random_gamma = lambda x: transforms.functional.adjust_gamma(
            x, gamma=random.uniform(0.7, 1.3)
        )
        
    def __call__(self, image):
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
            
        # Common transforms
        im_q = self.resize_crop(image)
        im_k = self.resize_crop(image)
        
        # Query image - stronger augmentations
        im_q = self.flip(im_q)
        im_q = self.clahe(im_q)
        if random.random() < 0.8:  # Apply with high probability
            im_q = self.color_jitter(im_q)
        if random.random() < 0.2:  # Apply with low probability
            im_q = self.grayscale(im_q)
        if random.random() < 0.5:  # Apply with medium probability
            im_q = self.gaussian_blur(im_q)
        if random.random() < 0.3:  # Apply rotation sometimes
            im_q = self.rotation(im_q)
            
        # Key image - more conservative augmentations
        im_k = self.flip(im_k)
        im_k = self.clahe(im_k)
        if random.random() < 0.3:  # Lower probability than query
            im_k = self.color_jitter(im_k)
        if random.random() < 0.5:  # Apply with medium probability 
            im_k = self.gaussian_blur(im_k)
            
        # Convert to tensor
        im_q = self.to_tensor(im_q)
        im_k = self.to_tensor(im_k)
        
        # Apply gamma correction (after tensor conversion)
        if random.random() < 0.3:
            im_q = self.random_gamma(im_q)
        if random.random() < 0.3:
            im_k = self.random_gamma(im_k)
            
        return im_q, im_k


# class MoCoSingleAug:
#     def __init__(self, img_size):
#         # Training pipeline with augmentations
#         self.base_trans = transforms.Compose([
#             # Resize slightly larger then crop to img_size
#             transforms.Resize(size=(int(img_size * 1.1), int(img_size * 1.1))),
#             transforms.CenterCrop(img_size),  # Crop to final size
#             CLAHE(clip_limit=3.0, tile_grid_size=(8, 8)),  # Your CLAHE transform
#             transforms.RandomHorizontalFlip(p=0.3),  # Random flip
#             transforms.RandomRotation(10),  # Random rotation up to 10 degrees
#             transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color adjustments
#             transforms.ToTensor(),  # Convert PIL Image to Tensor
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],  # ImageNet means
#                 std=[0.229, 0.224, 0.225]    # ImageNet stds
#             )
#         ])
        
        
    
#     def __call__(self, image):
#         """
#         Apply the appropriate transform based on mode.
        
#         Args:
#             image: Input PIL Image
#             evaluation: If True, use eval_trans; else use base_trans
        
#         Returns:
#             Transformed image as a Tensor
#         """
        
#         return self.base_trans(image)


class MoCoSingleAug:
    def __init__(self, img_size, is_training=True):
        # Common transformations
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
        
        if is_training:
            # More aggressive augmentations for training
            self.transform = transforms.Compose([
                # Resize with randomness
                transforms.RandomResizedCrop(
                    size=img_size,
                    scale=(0.8, 1.0),  # Random zoom level
                    ratio=(0.9, 1.1)   # Slight aspect ratio variation
                ),
                CLAHE(clip_limit=3.0, tile_grid_size=(8, 8)),
                transforms.RandomHorizontalFlip(p=0.5),  # Increased flip probability
                transforms.RandomVerticalFlip(p=0.1),    # Add vertical flips (retina is roughly symmetric)
                transforms.RandomRotation(
                    degrees=20,
                    fill=0
                ),
                transforms.RandomAffine(
                    degrees=0,  # No additional rotation
                    translate=(0.05, 0.05),  # Small translations
                    scale=(0.95, 1.05),  # Small scaling
                    fill=0
                ),
                # More aggressive color augmentation
                transforms.ColorJitter(
                    brightness=0.3, 
                    contrast=0.3, 
                    saturation=0.2, 
                    hue=0.1
                ),
                # Random grayscale conversion occasionally
                transforms.RandomGrayscale(p=0.02),
                # Gaussian blur to simulate focus issues
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
                # Random erasing can simulate occlusions or artifacts
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.2, 
                    scale=(0.02, 0.1), 
                    ratio=(0.3, 3.0), 
                    value=0
                ),
                normalize
            ])
        else:
            # Validation/test pipeline
            self.transform = transforms.Compose([
                transforms.Resize(size=(int(img_size * 1.1), int(img_size * 1.1))),
                transforms.CenterCrop(img_size),
                CLAHE(clip_limit=3.0, tile_grid_size=(8, 8)),
                transforms.ToTensor(),
                normalize
            ])
        
    def __call__(self, image):
        return self.transform(image)

import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import functional as F
# from torchvision.transforms.v2 import RandomCutmix, RandomMixup

class DiNOV2Aug:
    def __init__(self, img_size=224, global_crops_scale=(0.7, 1.0), local_crops_scale=(0.3, 0.7), 
                 n_local_crops=8, local_crops_size=96):
        self.img_size = img_size
        self.n_local_crops = n_local_crops
        self.local_crops_size = local_crops_size
        
        # Global crops transformation pipeline (2 crops)
        self.global_crops_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, 
                scale=global_crops_scale,
                ratio=(0.9, 1.1)  # Keep aspect ratio close to original for medical images
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            CLAHE(clip_limit=random.uniform(1.5, 3.0), tile_grid_size=(8, 8)),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                )
            ], p=0.7),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=11, sigma=(0.1, 1.0))
            ], p=0.4),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=90)
            ], p=0.3),
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=15.0, sigma=3.0)
            ], p=0.2),
            transforms.ToTensor(),
            self.random_gamma_transform,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Local crops transformation pipeline (n_local_crops)
        self.local_crops_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size, 
                scale=local_crops_scale,
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            CLAHE(clip_limit=random.uniform(1.5, 3.0), tile_grid_size=(8, 8)),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                )
            ], p=0.7),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 0.8))  # Smaller kernel for local crops
            ], p=0.4),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=90)
            ], p=0.3),
            transforms.ToTensor(),
            self.random_gamma_transform,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    
        self.global_cutout_size = img_size // 16
        self.local_cutout_size = local_crops_size // 16
        
    def random_gamma_transform(self, img):
        """Apply a subtle gamma adjustment with probability 0.3"""
        if random.random() < 0.3:
            return F.adjust_gamma(img, gamma=random.uniform(0.85, 1.15))
        return img
        
    def apply_saliency_masking(self, img, is_global=True, p=0.15):
        """Apply transparent cutout as a form of saliency masking"""
        if random.random() < p and isinstance(img, torch.Tensor):
            h, w = img.shape[1], img.shape[2]
            cutout_size = self.global_cutout_size if is_global else self.local_cutout_size
            mask = torch.ones_like(img)
            
            # Apply 1-3 random cutouts
            num_cutouts = random.randint(1, 3)
            for _ in range(num_cutouts):
                y = random.randint(0, h - cutout_size)
                x = random.randint(0, w - cutout_size)
          
                mask[:, y:y+cutout_size, x:x+cutout_size] = 0.5
                
            return img * mask
        return img
    
    def __call__(self, image):
      
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
            
        # Generate 2 global crops
        global_crops = [self.global_crops_transform(image) for _ in range(2)]
        
        # Apply saliency masking to global crops
        global_crops = [self.apply_saliency_masking(crop, is_global=True) for crop in global_crops]
        
        # Generate local crops
        local_crops = [self.local_crops_transform(image) for _ in range(self.n_local_crops)]
        
        # Apply saliency masking to local crops (with higher probability)
        local_crops = [self.apply_saliency_masking(crop, is_global=False, p=0.2) for crop in local_crops]
        
        # Return all crops as a list
        return global_crops + local_crops


class DiNOSingleAug:
    """
    Single view augmentation for evaluation purposes (linear probe, kNN).
    Uses gentler augmentations since we're preparing for downstream tasks.
    """
    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.1)),  # Slight oversize 
            transforms.CenterCrop(img_size),  # Center crop to remove boundary artifacts
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),  # Enhance contrast
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
            
        return self.transform(image)