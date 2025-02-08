from model.iBOT import MaskedViT , CustomiBOT
from data_pipeline.data_aug import RetAug
from tests.dl_test import ssltestdataset
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
trans = RetAug()
train_ds = ssltestdataset(trans=trans)
train_dl = DataLoader(dataset=train_ds , batch_size=3 , pin_memory=True)

def test_retuaug():
    import matplotlib.pyplot as plt
    import torchvision
    for batch in train_dl:
        view1, view2 = batch  # Unpack the tuple
        
        # Create grids for both views
        grid1 = torchvision.utils.make_grid(view1, nrow=3)
        grid2 = torchvision.utils.make_grid(view2, nrow=3)
        
        # Convert tensors to numpy arrays for display
        grid1_np = grid1.permute(1, 2, 0).cpu().numpy()
        grid2_np = grid2.permute(1, 2, 0).cpu().numpy()
        
        # Plot them side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(grid1_np)
        axes[0].axis('off')
        axes[0].set_title("View 1")
        axes[1].imshow(grid2_np)
        axes[1].axis('off')
        axes[1].set_title("View 2")
        plt.show()
        break


def test_VIT():

    faltu_tensor = torch.randn(size=(3 , 512 , 512))
    model = CustomiBOT()
    for batch in train_dl:
        view1 , view2 = batch

    

        g_feat , l_feat , mask = model(view1 , view2)

    print(g_feat , l_feat , mask)




        



if __name__ == "__main__":

    # test_retuaug()
    test_VIT()



    