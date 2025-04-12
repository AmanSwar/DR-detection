import openvino as ov
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from tqdm import tqdm
import time
import psutil
import os

# Print OpenVINO version
print(f"OpenVINO version: {ov.__version__}")


model_path = "model.onnx"
core = ov.Core()
model = core.read_model(model_path)

# Helper function for softmax
def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 2. Define evaluation function with timing and memory measurements
def evaluate_model(compiled_model, data_loader, precision_name="Unknown"):
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Get output tensor
    output_tensor = compiled_model.outputs[0]
    
    # Warmup (important for accurate timing)
    for images, _ in [next(iter(data_loader))]:
        compiled_model(images.numpy())
    
    # Measure inference time
    total_time = 0
    sample_count = 0
    batch_times = []
    
    for images, labels in tqdm(data_loader):
        images_np = images.numpy()
        
        # Time the inference
        start_time = time.time()
        results = compiled_model(images_np)[output_tensor]
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        sample_count += images.shape[0]
        batch_times.append(inference_time)
        
        probs = softmax(results, axis=1)
        preds = np.argmax(results, axis=1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
    
    # Calculate metrics
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    try:
        all_probs = np.array(all_probs)
        auc_macro_ovr = roc_auc_score(
            all_labels, all_probs, multi_class='ovr', average='macro'
        )
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc_macro_ovr = 0.0
    
    # Calculate memory usage
    end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    memory_used = end_memory - start_memory
    
    # Calculate timing statistics
    avg_time_per_batch = total_time / len(data_loader)
    avg_time_per_sample = total_time / sample_count
    fps = sample_count / total_time
    
    # Calculate timing variation (stability)
    std_dev = np.std(batch_times)
    
    print(f"\n{precision_name} Performance:")
    print(f"  Memory Usage: {memory_used:.2f} MB")
    print(f"  Average Inference Time: {avg_time_per_sample*1000:.2f} ms per sample")
    print(f"  Throughput: {fps:.2f} FPS")
    print(f"  Timing Stability (StdDev): {std_dev*1000:.2f} ms")
    
    return {
        'qwk': qwk,
        'f1_weighted': f1_weighted,
        'auc_macro_ovr': auc_macro_ovr,
        'memory_mb': memory_used,
        'avg_time_ms': avg_time_per_sample * 1000,  # Convert to ms
        'fps': fps,
        'timing_std_dev': std_dev * 1000  # Convert to ms
    }

from data_pipeline.data_set import UniformValidDataloader
from data_pipeline.data_aug import MoCoSingleAug

ds_name = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
val_transform = MoCoSingleAug(img_size=256)
val_loader = UniformValidDataloader(
        dataset_names=ds_name,
        transformation=val_transform,
        batch_size=32,
        num_workers=3
    ).get_loader()


print("Evaluating baseline FP32 model...")
compiled_model_fp32 = core.compile_model(model, device_name="CPU")
baseline_metrics = evaluate_model(compiled_model_fp32, val_loader, "FP32")

print("\nTesting full INT8 model...")
compiled_model_int8 = core.compile_model(
    model, 
    device_name="CPU", 
    config={"INFERENCE_PRECISION_HINT": "INT8"}
)
int8_metrics = evaluate_model(compiled_model_int8, val_loader, "INT8")

print("\nTesting FP16 model...")
compiled_model_fp16 = core.compile_model(
    model, 
    device_name="CPU", 
    config={"INFERENCE_PRECISION_HINT": "FP16"}
)
fp16_metrics = evaluate_model(compiled_model_fp16, val_loader, "FP16")

# 4. Print comprehensive comparison table
print("\n" + "="*80)
print("COMPREHENSIVE PERFORMANCE COMPARISON")
print("="*80)
print(f"{'Metric':<20} | {'FP32 (Baseline)':<20} | {'FP16':<20} | {'INT8':<20}")
print("-"*80)

# Accuracy metrics
print(f"{'QWK':<20} | {baseline_metrics['qwk']:<20.4f} | {fp16_metrics['qwk']:<20.4f} | {int8_metrics['qwk']:<20.4f}")
print(f"{'F1 Weighted':<20} | {baseline_metrics['f1_weighted']:<20.4f} | {fp16_metrics['f1_weighted']:<20.4f} | {int8_metrics['f1_weighted']:<20.4f}")
print(f"{'AUC (Macro-OvR)':<20} | {baseline_metrics['auc_macro_ovr']:<20.4f} | {fp16_metrics['auc_macro_ovr']:<20.4f} | {int8_metrics['auc_macro_ovr']:<20.4f}")

# Performance metrics
print("-"*80)
print(f"{'Memory (MB)':<20} | {baseline_metrics['memory_mb']:<20.2f} | {fp16_metrics['memory_mb']:<20.2f} | {int8_metrics['memory_mb']:<20.2f}")
print(f"{'Inference (ms/img)':<20} | {baseline_metrics['avg_time_ms']:<20.2f} | {fp16_metrics['avg_time_ms']:<20.2f} | {int8_metrics['avg_time_ms']:<20.2f}")
print(f"{'Throughput (FPS)':<20} | {baseline_metrics['fps']:<20.2f} | {fp16_metrics['fps']:<20.2f} | {int8_metrics['fps']:<20.2f}")
print(f"{'Timing StdDev (ms)':<20} | {baseline_metrics['timing_std_dev']:<20.2f} | {fp16_metrics['timing_std_dev']:<20.2f} | {int8_metrics['timing_std_dev']:<20.2f}")

# 5. Calculate relative performance changes
print("-"*80)
fp16_speed_up = baseline_metrics['avg_time_ms'] / fp16_metrics['avg_time_ms']
int8_speed_up = baseline_metrics['avg_time_ms'] / int8_metrics['avg_time_ms']
fp16_qwk_change = (fp16_metrics['qwk'] / baseline_metrics['qwk'] - 1) * 100
int8_qwk_change = (int8_metrics['qwk'] / baseline_metrics['qwk'] - 1) * 100
fp16_memory_reduction = (1 - fp16_metrics['memory_mb'] / baseline_metrics['memory_mb']) * 100
int8_memory_reduction = (1 - int8_metrics['memory_mb'] / baseline_metrics['memory_mb']) * 100

print(f"{'Speed-up vs FP32':<20} | {'1.00x':<20} | {fp16_speed_up:<19.2f}x | {int8_speed_up:<19.2f}x")
print(f"{'QWK Change':<20} | {'0.00%':<20} | {fp16_qwk_change:<19.2f}% | {int8_qwk_change:<19.2f}%")
print(f"{'Memory Reduction':<20} | {'0.00%':<20} | {fp16_memory_reduction:<19.2f}% | {int8_memory_reduction:<19.2f}%")
print("="*80)

# 6. Provide recommendation
print("\nRECOMMENDATION:")
if int8_qwk_change > -1.0:  # Less than 1% QWK loss
    print("✅ INT8 quantization is recommended: Significant performance improvement with minimal accuracy loss")
elif fp16_qwk_change > -0.5:  # Less than 0.5% QWK loss
    print("✅ FP16 precision is recommended: Good balance between performance and accuracy")
else:
    print("❌ Use FP32 baseline: Quantization has too much impact on model accuracy for this task")