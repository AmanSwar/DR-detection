from tests.model.sample_data import SslDs
from data_pipeline.data_aug import IJEPAAugmentation
from torch.utils.data import DataLoader
from model.DRijepa_swin import Trainer , create_DRijepa


transform = IJEPAAugmentation(img_size=224)
data_ds = SslDs(trans=transform)
data_ld = DataLoader(dataset=data_ds , batch_size=4 , pin_memory=True)

# train_single_gpu(data_ld=data_ld , batch_size=4)

model, loss_fn = create_DRijepa(
        img_size=224,
        model_name='swin_base_patch4_window7_224',
        n_box=6,
        dropout=0.1
    )

trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=data_ld,
        max_ep=30
    )

trainer.train()