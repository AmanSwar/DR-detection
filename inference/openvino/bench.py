import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
from openvino.runtime import Core
from data_pipeline.data_aug import MoCoSingleAug
from data_pipeline.data_set import UniformValidDataloader
val_transform = MoCoSingleAug(img_size=256)
# Assuming val_loader is your validation dataloader
# Replace with your actual dataloader instantiation if needed
# Example: val_loader = get_val_loader()
ds_name = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
val_loader = UniformValidDataloader(
        dataset_names=ds_name,
        transformation=val_transform,
        batch_size=32,
        num_workers=3
    ).get_loader()



def validate_openvino(compiled_model, dataloader, num_classes=5):
    """
    Validate the model using OpenVINO inference and compute performance metrics.
    
    Args:
        exec_net: OpenVINO executable network
        dataloader: Validation data loader yielding (images, labels, ...)
        num_classes: Number of classes (default=5 as per your code)
    
    Returns:
        Dictionary with performance metrics and inference time
    """
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    inference_time = 0.0
    
    # Get input and output blob names
    input_node = compiled_model.input(0)
    output_node = compiled_model.output(0)
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            # Handle dataloader structure
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data
            
            # Convert to numpy for OpenVINO
            images_np = images.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Measure inference time
            # Measure inference time using synchronous inference
            start_time = time.time()
            # Perform inference using the compiled model and input data
            results = compiled_model({input_node: images_np})
            end_time = time.time()
            inference_time += end_time - start_time

            # Extract logits from the results dictionary using the output node
            logits = results[output_node]

            
            # Convert to torch tensors for loss computation
            logits_torch = torch.from_numpy(logits)
            labels_torch = torch.from_numpy(labels_np)
            loss = loss_fn(logits_torch, labels_torch)
            running_loss += loss.item()
            
            # Compute probabilities and predictions
            probs = torch.softmax(logits_torch, dim=1).numpy()
            preds = np.argmax(logits, axis=1)
            
            all_labels.extend(labels_np)
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    # Compute metrics
    avg_loss = running_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Confusion matrix-based metrics
    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = []
    specificity = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)
    
    # AUC calculation
    try:
        auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc_macro_ovr = 0.0
    
    return {
        'avg_loss': avg_loss,
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'qwk': qwk,
        'avg_sensitivity': avg_sensitivity,
        'avg_specificity': avg_specificity,
        'auc_macro_ovr': auc_macro_ovr,
        'inference_time': inference_time
    }

# Main comparison script
def compare_models():
    # Initialize OpenVINO Inference Engine
    ie = Core()
    
    # Define model files (adjust paths/names as per your files)
    models = {
        # 'FP32': ('fp32_model.xml', 'fp32_model.bin'),  # Assuming original weights are in OpenVINO FP32
        'FP16': ('fp16_model.xml', 'fp16_model.bin'),
        'INT8': ('int8_model.xml', 'int8_model.bin')   # Assuming INT8 follows similar naming
    }
    
    results = {}
    
    # Process each model
    for model_name, (xml_file, bin_file) in models.items():
        print(f"\nProcessing {model_name} model...")
        
        # Check if files exist
        if not (os.path.exists(xml_file) and os.path.exists(bin_file)):
            print(f"Warning: {xml_file} or {bin_file} not found. Skipping {model_name}.")
            continue
        
        # Load the model
        # Load and compile the model using the new API
        model = ie.read_model(model=xml_file, weights=bin_file)
        compiled_model = ie.compile_model(model=model, device_name='CPU')

        
        # Measure size
        size = os.path.getsize(bin_file)  # Size of .bin file (weights)
        
        # Run validation and measure inference time
        metrics = validate_openvino(compiled_model, val_loader)
        
        # Store results
        results[model_name] = {
            'size': size,
            'inference_time': metrics['inference_time'],
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'qwk': metrics['qwk'],
            'avg_sensitivity': metrics['avg_sensitivity'],
            'avg_specificity': metrics['avg_specificity'],
            'auc_macro_ovr': metrics['auc_macro_ovr']
        }
    
    # Display results
    print("\n=== Model Comparison ===")
    for model_name, res in results.items():
        print(f"\nModel: {model_name}")
        print(f"Size: {res['size'] / 1024 / 1024:.2f} MB")
        print(f"Inference Time: {res['inference_time']:.4f} seconds")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print(f"F1 Weighted: {res['f1_weighted']:.4f}")
        print(f"QWK: {res['qwk']:.4f}")
        print(f"Avg Sensitivity: {res['avg_sensitivity']:.4f}")
        print(f"Avg Specificity: {res['avg_specificity']:.4f}")
        print(f"AUC Macro OvR: {res['auc_macro_ovr']:.4f}")
    
    # Compare to FP32 (performance hit and improvements)
    # if 'FP32' in results:
    #     fp32_metrics = results['FP32']
    #     print("\n=== Performance Hit and Improvements Relative to FP32 ===")
    #     for model_name in ['FP16', 'INT8']:
    #         if model_name not in results:
    #             continue
    #         res = results[model_name]
    #         size_reduction = (fp32_metrics['size'] - res['size']) / fp32_metrics['size'] * 100
    #         speedup = fp32_metrics['inference_time'] / res['inference_time'] if res['inference_time'] > 0 else float('inf')
    #         print(f"\n{model_name} vs FP32:")
    #         print(f"Size Reduction: {size_reduction:.2f}%")
    #         print(f"Speedup: {speedup:.2f}x")
    #         print(f"Accuracy Drop: {fp32_metrics['accuracy'] - res['accuracy']:.4f}")
    #         print(f"F1 Weighted Drop: {fp32_metrics['f1_weighted'] - res['f1_weighted']:.4f}")
    #         print(f"QWK Drop: {fp32_metrics['qwk'] - res['qwk']:.4f}")

if __name__ == "__main__":
    # Ensure val_loader is defined before running
    # val_loader = your_dataloader_function()
    compare_models()