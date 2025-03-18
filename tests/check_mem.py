import torch
import gc

def get_gpu_info():
    """
    Retrieves information about available GPUs, including their memory.
    """
    gpu_info = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            device = torch.device(f'cuda:{i}')
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # in GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3) # in GB
            cached_memory = torch.cuda.memory_reserved(device) / (1024 ** 3) # in GB
            free_memory = total_memory - allocated_memory
            gpu_info.append({
                'index': i,
                'name': gpu_name,
                'total_memory_gb': total_memory,
                'allocated_memory_gb': allocated_memory,
                'cached_memory_gb': cached_memory,
                'free_memory_gb': free_memory,
            })
    else:
        gpu_info.append({'error': 'CUDA is not available.'})
    return gpu_info

def print_gpu_info():
    """
    Prints the GPU information in a formatted way.
    """
    gpu_info = get_gpu_info()
    for gpu in gpu_info:
        if 'error' in gpu:
            print(gpu['error'])
            return
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  Total Memory: {gpu['total_memory_gb']:.2f} GB")
        print(f"  Allocated Memory: {gpu['allocated_memory_gb']:.2f} GB")
        print(f"  Cached Memory: {gpu['cached_memory_gb']:.2f} GB")
        print(f"  Free Memory: {gpu['free_memory_gb']:.2f} GB")
        print("-" * 20)

if __name__ == "__main__":
    print_gpu_info()