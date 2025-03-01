import torch

def get_gpu_info():
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        available_memory = torch.cuda.mem_get_info(i)[0]  # in bytes
        gpu_info.append({
            "name": gpu.name,
            "index": i,
            "total_memory": gpu.total_memory / (1024**3),  # Convert to GB
            "available_memory": available_memory / (1024**3)  # Convert to GB
        })
    return gpu_info

gpu_list = get_gpu_info()
for gpu in gpu_list:
    print(f"GPU {gpu['index']}: {gpu['name']}")
    print(f"  Total Memory: {gpu['total_memory']:.2f} GB")
    print(f"  Available Memory: {gpu['available_memory']:.2f} GB")
