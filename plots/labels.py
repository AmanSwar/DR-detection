from collections import Counter
from prettytable import PrettyTable 
from data_pipeline.data_set import UnitedTrainingDataset , UniformTrainDataloader
from data_pipeline.data_aug import MoCoSingleAug
dataset_names = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
# dataset = UnitedTrainingDataset(*dataset_names)

# # Get the labels
# labels = dataset.get_labels()

# # Count occurrences of each label
# label_counts = Counter(labels)

# # Get total number of images
# total_images = len(dataset)

# # Create a table
# table = PrettyTable()
# table.field_names = ["Label", "Count", "Percentage"]

# # Add rows to the table
# for label, count in sorted(label_counts.items()):
#     percentage = (count / total_images) * 100
#     table.add_row([label, count, f"{percentage:.2f}%"])

# # Add total row
# table.add_row(["Total", total_images, "100.00%"])

# # Print the table
# print("Label Distribution in UnitedTrainingDataset:")
# print(table)

# # Optional: Print additional dataset information
# print("\nDataset Composition:")
# for dataset_name, count in dataset.get_dataset_counts().items():
#     print(f"{dataset_name}: {count} images")


transformation =  MoCoSingleAug(img_size=256)
batch_size = 64       
num_workers = 0       

# Initialize the dataloader
dataloader = UniformTrainDataloader(
    dataset_names=dataset_names,
    transformation=transformation,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=True
)

train_loader = dataloader.get_loader()

dataset = train_loader.dataset
all_labels = dataset.get_labels()
unique_labels = sorted(set(all_labels))

# Get one batch from the dataloader
for images, labels in train_loader:
    batch_labels = labels.tolist()  # Convert tensor to list
    batch_label_counts = Counter(batch_labels)  # Count occurrences of each label
    break  # Only need one batch

# Create a table to display the distribution
table = PrettyTable()
table.field_names = ["Label", "Count", "Percentage"]

# Populate the table with counts for all possible labels
for label in unique_labels:
    count = batch_label_counts[label]  # Returns 0 if label not in batch
    percentage = (count / batch_size) * 100
    table.add_row([label, count, f"{percentage:.2f}%"])

# Add a total row
total_count = sum(batch_label_counts.values())  # Should equal batch_size
table.add_row(["Total", total_count, "100.00%"])

# Print the table
print(f"Label Distribution in a Batch of Size {batch_size}:")
print(table)
