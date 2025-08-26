import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from data_pipeline.data_aug import MoCoAug , MoCoSingleAug
import os
import matplotlib.pyplot as plt
import os
os.environ['QT_DEBUG_PLUGINS'] = '0'

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean  # Reverse normalization
    tensor = torch.clamp(tensor, 0, 1)  # Ensure values stay in [0,1]
    return tensor

# Function to convert tensor to numpy array for display
def tensor_to_image(tensor):
    tensor = denormalize(tensor)
    image = tensor.permute(1, 2, 0).numpy()  # Convert from C x H x W to H x W x C
    return image

# Load the retinal fundus image
image_path = 'tests/data/0a74c92e287c.png'  
image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode

# Instantiate the augmentor
augmentor = MoCoSingleAug(img_size=224)

# Apply augmentations to get query and key images
im_q, im_k = augmentor(image)

# Resize the original image to match the augmented image size (224x224)
resize = transforms.Resize((224, 224))
original_resized = resize(image)
original_np = np.array(original_resized) / 255.0  # Convert to numpy array in [0,1]

# Create a figure with three subplots to display the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display the original image
axes[0].imshow(original_np)
axes[0].set_title('Original')
axes[0].axis('off')

# Display the query augmented image
axes[1].imshow(tensor_to_image(im_q))
axes[1].set_title('Query Augmentation')
axes[1].axis('off')

# Display the key augmented image
axes[2].imshow(tensor_to_image(im_k))
axes[2].set_title('Key Augmentation')
axes[2].axis('off')

# Save the plot instead of showing it
plt.tight_layout()
plt.savefig('augmented_images.png')
plt.close()