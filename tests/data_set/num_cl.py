from data_pipeline.data_set import UnitedTrainingDataset , UnitedValidationDataset , UnitedSSLTrainingDataset
from data_pipeline.data_eval import UniTestDataset
# dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
dataset_names = ["messdr"]
ds = UniTestDataset("lmao" , transform=None)

print(len(ds))