import torch
from data_pipeline.data_aug import IJEPAAugmentation , DINOAugmentation
from data_pipeline.data_set import UniformTrainDataloader , UnitedValidationDataset


def test_ijepa_aug():

    batched_img = torch.randn(size=(8 , 1024 , 1024))

    augmentation = IJEPAAugmentation()

    aug_img = augmentation(batched_img)

    # check for uniform sizes



def test_trainloader():
    dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid"]
    train_ld = UniformTrainDataloader(
        dataset_names=dataset_names,
        transformation=IJEPAAugmentation(),
        batch_size=8,
        num_workers=1,
        sampler=True
    )

    data_ld = train_ld.get_loader()
    for batch in data_ld:
        batched_imgs = batch[0]

        for i in batched_imgs:

            print(i.shape)
        
        

if __name__ == "__main__":

    test_trainloader()