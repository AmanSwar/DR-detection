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

# The main augmentation class using torchvision.transforms
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
            transforms.Resize((img_size, img_size)),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1)],
                p=0.5
            ),
            # transforms.RandomApply(
            #     [transforms.ColorJitter(hue=5/360, saturation=0.1, brightness=0.05)],
            #     p=0.3
            # )
        ])
        self.tensor_transforms = transforms.Compose([
            # transforms.RandomApply([
            #     transforms.RandomChoice([
            #         GaussianNoiseTransform((10.0, 50.0)),
            #         MultiplicativeNoiseTransform((0.95, 1.05))
            #     ])
            # ], p=0.3),
            
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    MedianBlurTransform_ijepa(3)
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
        image = self.pil_transforms(image)
        image = TF.to_tensor(image)
        image = self.tensor_transforms(image)
        return image.float().clone()
    

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

