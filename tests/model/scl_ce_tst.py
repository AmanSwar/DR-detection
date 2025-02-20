from data_pipeline.data_set import UniformTrainDataloader , UniformValidDataloader
from model.custom_scl import train
from data_pipeline.data_aug import scl_trans

dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid" , "messdr"]
trainloader = UniformTrainDataloader(
    dataset_names=dataset_names,
    transformation=scl_trans,
    batch_size=64,
    num_workers=2,
    sampler=True
).get_loader()
validloader = UniformValidDataloader(
        dataset_names=dataset_names,
        transformation=scl_trans,
        batch_size=64,
        num_workers=2,
        sampler=True
    ).get_loader()


train(30 , trainloader , validloader)
