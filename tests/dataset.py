import torch
import torch.nn as nn
from torch.utils.data import DataLoader ,WeightedRandomSampler
from collections import Counter
import numpy as np

from data_pipeline.data_set import UnitedTrainingDataset , UnitedValidationDataset , UniformTrainDataloader , UnitedSSLTrainingDataset
from data_pipeline.data_set import SSLTrainLoader
from data_pipeline.data_aug import IJEPAAugmentation , DINOAugmentation



BATCH_SIZE = 32





# all_data_set = UnitedTrainingDataset(
#     "eyepacs",
#     "aptos",
#     "ddr",
#     "idrid",
#     transformation=IJEPAAugmentation(),
# )


# all_validation_set = UnitedValidationDataset(
#     "eyepacs",
#     "aptos",
#     "ddr",
#     "idrid",
#     transformation=IJEPAAugmentation(),
# )

# labels_np = np.array(all_data_set.get_labels())
# class_counts = Counter(labels_np)

# total_samples = len(labels_np)
# class_weight = {cls : total_samples/count for cls , count in class_counts.items()}
# sample_weight = [class_weight[label] for label in labels_np]

# weight_tensor = torch.DoubleTensor(sample_weight)
# sampler = WeightedRandomSampler(weights=weight_tensor , num_samples=len(weight_tensor) , replacement=True)


# all_train_data_loader = DataLoader(dataset=all_data_set ,sampler=sampler ,batch_size=BATCH_SIZE , pin_memory=True , num_workers=0)
# all_valid_data_loader = DataLoader(dataset=all_validation_set ,sampler=sampler ,batch_size=BATCH_SIZE , pin_memory=True , num_workers=0)

dataset_names = ["eyepacs" , "aptos" , "ddr" , "idrid"]
# uniform_data_ld = UniformTrainDataloader(
#     dataset_names=dataset_names,
#     transformation=IJEPAAugmentation(),
#     batch_size=BATCH_SIZE,
#     num_workers=2,
#     sampler=True
# )

# data_ld = uniform_data_ld.get_loader()

# # for img , label in data_ld:
# #     print(img)
# #     print(label)
# #     print("\n")

train_ds = UnitedSSLTrainingDataset("eyepacs" , "aptos" , "ddr" , "idrid" , "messdr")

print(len(train_ds))


ssl_train_ld = SSLTrainLoader(dataset_names=dataset_names , transformation=IJEPAAugmentation(img_size=256) , batch_size=8 , num_work=2)

data_ld = ssl_train_ld.get_loader()
for img in data_ld:
    print(img.shape)
    print("\n")
    break





