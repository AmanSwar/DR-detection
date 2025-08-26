from data_pipeline.data_set import UnitedSSLTrainingDataset , UnitedTrainingDataset
import os
import random
from typing import List
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import defaultdict

dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
dataset = UnitedSSLTrainingDataset(*dataset_names)
counts = dataset.get_dataset_counts()
total_images = len(dataset.get_paths())
print(f"Total images: {total_images}")

labels = ["EyePACS", "APTOS", "DDR", "IDRiD", "MESSIDOR 2"]
sizes = [counts.get(name, 0) for name in dataset_names]
print(sizes)

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=140)
plt.title("Dataset Composition in UnitedSSLTrainingDataset")
plt.axis('equal')
plt.show()