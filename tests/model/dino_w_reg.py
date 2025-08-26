from tests.model.sample_data import SslDs
from data_pipeline.data_aug import DinowregAug
from torch.utils.data import DataLoader
from train.dinowreg_swin import train_single_gpu

transform = DinowregAug(img_size=256)
data_ds = SslDs(trans=transform)
data_ld = DataLoader(dataset=data_ds , batch_size=8 , pin_memory=True)

train_single_gpu(data_ld , b_size=16 , max_epoch=30)