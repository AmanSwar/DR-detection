import os
from data_pipeline.utils import GradingDataset , add_path
import pandas as pd
from tqdm import tqdm



 
ddr_root_dir = "data/ddr"
subset = "train"
# for dirpath , dirnames , filenames in os.walk(ddr_root_dir):

#     if "DR_grading" in dirpath:
#         print(dirnames)
        

col_names = ["imgs" , "labels"]
labels_df = pd.read_csv("data/ddr/DDR-dataset/DR_grading/train.txt" , sep=' ' , names=col_names)
labels_dic = {img_name : label for img_name , label in zip(labels_df['imgs'] , labels_df['labels'])}

print(labels_df["imgs"][0])
print("007-0004-000.jpg" in labels_df["imgs"].values)
print("007-0004-000.jpg" in labels_df["imgs"])
img_path = "data/ddr/DDR-dataset/DR_grading/train/007-0004-000.jpg"
print(labels_dic[img_path.split('/')[-1]])
