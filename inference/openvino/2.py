import openvino as ov
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from tqdm import tqdm
from data_pipeline.data_set import UniformValidDataloader
from data_pipeline.data_aug import MoCoSingleAug

# 1. First, load your ONNX model
model_path = "model.onnx"
core = ov.Core()
model = core.read_model(model_path)

# 2. Define a calibration data loader
def calibration_data_generator(data_loader, num_samples=100):
    count = 0
    for images, labels in data_loader:
        yield {"images": images.numpy()}
        count += 1
        if count >= num_samples:
            break
def prepare_data_for_calibration(data_loader, num_samples=100):
    calibration_data = []
    count = 0
    for images, _ in data_loader:
        calibration_data.append(images.numpy())
        count += 1
        if count >= num_samples:
            break
    return calibration_data



ds_name = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
val_transform = MoCoSingleAug(img_size=256)
val_loader = UniformValidDataloader(
        dataset_names=ds_name,
        transformation=val_transform,
        batch_size=32,
        num_workers=3
    ).get_loader()
# Create a subset of your validation data for calibration
calibration_dataset = torch.utils.data.Subset(val_loader.dataset, indices=range(100))
calibration_dataloader = torch.utils.data.DataLoader(
    calibration_dataset, batch_size=8, shuffle=False
)

calibration_data = prepare_data_for_calibration(calibration_dataloader)

# 3. Define a metric function to evaluate model performance
def evaluate_model(compiled_model, data_loader):
    all_labels = []
    all_preds = []
    all_probs = []
    
    output_layer = compiled_model.output(0)
    
    for images, labels in tqdm(data_loader):
        images_np = images.numpy()
        results = compiled_model([images_np])[output_layer]
        probs = softmax(results, axis=1)
        preds = np.argmax(results, axis=1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
    
    # Calculate metrics
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Calculate AUC
    try:
        all_probs = np.array(all_probs)
        auc_macro_ovr = roc_auc_score(
            all_labels, all_probs, multi_class='ovr', average='macro'
        )
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc_macro_ovr = 0.0
    
    return {
        'qwk': qwk,
        'f1_weighted': f1_weighted,
        'auc_macro_ovr': auc_macro_ovr
    }

# Helper function for softmax
def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 4. Baseline evaluation with FP32 model
print("Evaluating baseline FP32 model...")
compiled_model_fp32 = core.compile_model(model, "CPU")
# baseline_metrics = evaluate_model(compiled_model_fp32, val_loader)
# print(f"Baseline metrics: {baseline_metrics}")

print("Applying INT8 quantization...")

# Get model inputs
inputs = model.inputs

# Simple INT8 quantization using direct compilation with INT8 hint
compiled_model_int8 = core.compile_model(
    model, 
    device_name="CPU", 
    config={"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "INT8"}
)

# 6. Evaluate the INT8 model
print("Evaluating INT8 model...")
int8_metrics = evaluate_model(compiled_model_int8, val_loader)
print(f"INT8 metrics: {int8_metrics}")

# 7. For a more controlled approach, try per-tensor quantization using nGraph passes
# This allows more control over which parts get quantized
print("Trying per-tensor quantization for better control...")

# We can use low precision transformations API if available
# First, identify sensitive nodes in the model
sensitive_ops = []
for op in model.get_ops():
    op_name = op.get_friendly_name()
    if any(name in op_name.lower() for name in ["attention", "class", "grade_head"]):
        sensitive_ops.append(op_name)
        print(f"Found sensitive operation: {op_name}")

# Try selective precision using the available API
try:
    # In newer OpenVINO versions, set precision directly
    from openvino.runtime import properties
    
    # Create configuration with mixed precision hints
    config = {
        "OPTIMIZATION_CAPABILITIES": "FP16,INT8",
        "PERFORMANCE_HINT": "LATENCY"
    }
    
    # Compile with optimization capabilities
    mixed_model = core.compile_model(model, "CPU", config)
    
    # Evaluate the mixed precision model
    print("Evaluating mixed precision model...")
    mixed_metrics = evaluate_model(mixed_model, val_loader)
    print(f"Mixed precision metrics: {mixed_metrics}")
    
    # Save the best model based on QWK performance
    if mixed_metrics['qwk'] > int8_metrics['qwk']:
        best_model = mixed_model
        best_metrics = mixed_metrics
        print("Mixed precision model performed better")
    else:
        best_model = compiled_model_int8
        best_metrics = int8_metrics
        print("Full INT8 model performed better")
    
except Exception as e:
    print(f"Error with mixed precision: {e}")
    best_model = compiled_model_int8
    best_metrics = int8_metrics
    print("Using INT8 model as fallback")

# 8. Print performance comparison
print("\nPerformance Comparison:")
# print(f"FP32 Baseline - QWK: {baseline_metrics['qwk']:.4f}, F1: {baseline_metrics['f1_weighted']:.4f}, AUC: {baseline_metrics['auc_macro_ovr']:.4f}")
print(f"Best Optimized - QWK: {best_metrics['qwk']:.4f}, F1: {best_metrics['f1_weighted']:.4f}, AUC: {best_metrics['auc_macro_ovr']:.4f}")

# try:
#     # For OpenVINO 2022+
#     ov.save_model(best_model, "dr_model_optimized.xml", model_path="dr_model_optimized.bin")
#     print("Model saved successfully")
# except Exception as e:
#     print(f"Error saving model: {e}")
#     # For older OpenVINO versions
#     try:
#         best_model.export("dr_model_optimized.xml")
#         print("Model exported using legacy method")
#     except:
#         print("Could not save the optimized model")