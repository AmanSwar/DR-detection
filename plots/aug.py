from data_pipeline.data_aug import MoCoAug, MoCoSingleAug
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.clone()  # Avoid modifying the original tensor
    tensor.mul_(std).add_(mean)  # Denormalize: (tensor * std) + mean
    return tensor

# Load the image
img_path = 'tests/data/0a74c92e287c.png'
img = Image.open(img_path).convert('RGB')

# Set the image size
img_size = 256

# Initialize transformation
transform = MoCoSingleAug(img_size)

# Apply transformations
train_tensor = transform(img, evaluation=False)  # Training transformation
eval_tensor = transform(img, evaluation=True)    # Evaluation transformation

# Transformation for the original image
orig_trans = transforms.Compose([
    transforms.Resize(size=(int(img_size * 1.1), int(img_size * 1.1))),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
])
orig_tensor = orig_trans(img)

# Prepare images for display/save
orig_img = orig_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format for matplotlib
eval_img = denormalize(eval_tensor).clamp(0, 1).permute(1, 2, 0).numpy()  # Denormalize and clamp
train_img = denormalize(train_tensor).clamp(0, 1).permute(1, 2, 0).numpy()

# Visualize the images side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image (resized and cropped)
axes[0].imshow(orig_img)
axes[0].set_title('Original (Resized & Cropped)')
axes[0].axis('off')

# Evaluation transformed image
axes[1].imshow(eval_img)
axes[1].set_title('Evaluation Transformed')
axes[1].axis('off')

# Training transformed image
axes[2].imshow(train_img)
axes[2].set_title('Training Transformed')
axes[2].axis('off')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('plots/transformed_images.png', bbox_inches='tight')
print("Transformed images saved to 'transformed_images.png'")