from data_pipeline.data_aug import IbotRetAug
from tests.model.sample_data import SslDs
from torch.utils.data import DataLoader

from model.iBOT_swin import CustomiBOT , train_single_gpu

transform = IbotRetAug(img_size=224)
data_ds = SslDs(trans=transform)
data_ld = DataLoader(dataset=data_ds , batch_size=4 , pin_memory=True)

ibot_model = CustomiBOT(
        embed_dim=1024,
        momentum=0.996
    )
train_single_gpu(ibot_model , data_ld , 30)
