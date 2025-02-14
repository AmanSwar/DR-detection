import matplotlib.pyplot as plt
import torch
def show_img(train_ds):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i, ax in enumerate(axes.flatten()):
        # Assume each train_ds[i] returns multiple views, so we select one
        views = train_ds[i]
        # If views is a list/tuple, choose the first view.
        img = views[0] if isinstance(views, (list, tuple)) else views

        # If the image is a PyTorch tensor, convert it to a numpy array in (H, W, C) order.
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()

        ax.imshow(img)
        ax.axis('off') 

    plt.tight_layout()
    plt.show()



def show_multi_img(train_ds):
    views = train_ds[0]  # This returns a list of views

    # Decide how many views you want to show (here, 4)
    n_views_to_show = 4

    # Create a grid to display the images (e.g., 2 rows x 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Loop over the first 4 views and display them
    for i, ax in enumerate(axes.flatten()):
        view = views[i]  # select the i-th view

        # If the view is a tensor (from transforms.ToTensor), convert it to a NumPy array
        if torch.is_tensor(view):
            # Convert from (C, H, W) to (H, W, C)
            view = view.permute(1, 2, 0).cpu().numpy()
        
        ax.imshow(view)
        ax.axis("off")

    plt.tight_layout()
    plt.show()