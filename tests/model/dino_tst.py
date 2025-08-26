from tests.model.sample_data import SslDs
from data_pipeline.data_aug import DINOAugmentation
from torch.utils.data import DataLoader
from train.dino_swin import single_gpu_train

transform = DINOAugmentation(img_size=224)
data_ds = SslDs(trans=transform)
data_ld = DataLoader(dataset=data_ds , batch_size=4 , pin_memory=True)


single_gpu_train(
    dataloader=data_ld,
    n_epochs=30,
    batch_size=4
)