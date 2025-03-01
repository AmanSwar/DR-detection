import torch
total_memory = torch.cuda.get_device_properties(0).total_memory
available_memory = torch.cuda.mem_get_info()[0]

total_memory_gb = total_memory / (1024**3)
available_memory_gb = available_memory / (1024**3)

print(f"Total GPU memory: {total_memory_gb:.2f} GB")
print(f"Available GPU memory: {available_memory_gb:.2f} GB")
