import torch
import torch.nn as nn
import timm
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score
import logging
import os
import time
import subprocess # For benchmark_app
import openvino


from data_pipeline.data_set import UniformValidDataloader , UniformTrainDataloader

class LesionAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(LesionAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        assert kernel_size % 2 == 1, "Kernel size must be odd for spatial attention"
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x_channel = x * channel_att
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_input))
        return x_channel * spatial_att

class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradeConsistencyHead(nn.Module):
    def __init__(self, feature_dim, num_grades=5, dropout_rate=0.4):
        super(GradeConsistencyHead, self).__init__()
        self.grade_predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate), # Dropout is disabled in eval mode
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate), # Dropout is disabled in eval mode
            nn.Linear(256, num_grades)
        )
        self.ordinal_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_grades - 1)
        )

    def forward(self, x):
        logits = self.grade_predictor(x)
        ordinal_thresholds = self.ordinal_encoder(x)
        return logits, ordinal_thresholds

class EnhancedDRClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True): # freeze_backbone might not matter after loading state_dict
        super(EnhancedDRClassifier, self).__init__()
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0) # Set pretrained=False if loading weights
        self.feature_dim = self.backbone.num_features
        self.attention = LesionAttentionModule(self.feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.grade_head = GradeConsistencyHead(self.feature_dim, num_grades=num_classes)
       

    def forward(self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None):
        # Standard forward pass needed for training
        features = self.backbone.forward_features(x)
        attended_features = self.attention(features)
        h = torch.mean(attended_features, dim=(2, 3)) # Global Average Pooling after attention

        logits = self.classifier(h)
        grade_logits, grade_thresholds = self.grade_head(h) # Corrected: grade_head returns two outputs

        # The validation function only uses the main `logits`.
        # Let's return outputs consistently. The validation fn below needs adjustment if it uses grade_outputs
        # For simplicity for ONNX export and validation, let's assume we primarily need 'logits'.
        # If 'grade_logits' are also needed for validation, adjust the wrapper and validation fn.

        # Domain logic only needed for training with DANN
        domain_logits = None
        if alpha > 0:
            reversed_features = GradientReversal.apply(h, alpha)
            # Assuming domain_classifier exists if alpha > 0
            # domain_logits = self.domain_classifier(reversed_features)
            pass # Placeholder as domain_classifier was commented out

        # Note: The original validation code only uses `logits` derived from `self.classifier`.
        # It passes `grade_outputs` to a loss function which is commented out.
        # Let's focus on exporting what's needed for the provided validation metrics.
        # Return type needs to be consistent for tracing/export.
        # We'll use a wrapper for export.

        # Returning only main logits for standard inference/validation based on the provided validate func usage
        # If other outputs are needed, adjust the wrapper below.
        return logits #, (grade_logits, grade_thresholds), domain_logits

# --- Helper: Inference Wrapper for ONNX Export ---
# This wrapper simplifies the forward pass for inference, returning only the necessary output(s).
class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Call the original model's forward but ensure alpha=0 and ignore unnecessary outputs/logic
        features = self.model.backbone.forward_features(x)
        attended_features = self.model.attention(features)
        h = torch.mean(attended_features, dim=(2, 3))
        logits = self.model.classifier(h)
        # NOTE: If your validation *truly* needs grade_logits, return them here too,
        # e.g., grade_logits, _ = self.model.grade_head(h); return logits, grade_logits
        # Update ONNX export output_names and OpenVINO processing accordingly.
        # Based on the provided validate function, only `logits` are used for metrics.
        return logits

# --- Validation Function (Slightly modified for OpenVINO) ---
logging.basicConfig(level=logging.INFO) # For OpenVINO/NNCF messages

def validate(model, dataloader, device, epoch=0, num_epochs=1, num_classes=5, is_openvino=False, compiled_model=None):
    if not is_openvino:
        model.eval()
    else:
        assert compiled_model is not None, "Compiled OpenVINO model must be provided"
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    loss_fn = nn.CrossEntropyLoss()

    # Use torch.no_grad() for the entire loop, safe for both PyTorch and OpenVINO
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation Epoch {epoch+1}/{num_epochs}")
        for i, batch_data in pbar:
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data

            if is_openvino:
                input_data = images.cpu().numpy()
                results = compiled_model(input_data)  # Synchronous inference
                logits_np = results[output_layer]
                logits = torch.from_numpy(logits_np).to(device)
                labels = labels.to(device)
            else:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)

            # Metrics calculation remains unchanged
            loss = loss_fn(logits, labels)
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            pbar.set_postfix({"Loss": running_loss / (i + 1)})
    
    # Rest of the function (metrics calculation) remains the same
    # Calculate Metrics
    avg_loss = running_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    try:
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
    except Exception as e:
        logging.warning(f"Could not calculate confusion matrix based metrics: {e}")
        cm = None; avg_sensitivity = 0.0; avg_specificity = 0.0

    try:
        present_classes = np.unique(all_labels)
        if len(present_classes) == num_classes and all_probs.shape[1] == num_classes:
             auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        elif len(present_classes) > 1 and all_probs.shape[1] == num_classes:
             logging.warning(f"Only {len(present_classes)}/{num_classes} classes present. AUC might be unreliable.")
             try: auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro', labels=present_classes)
             except ValueError: auc_macro_ovr = 0.0
        else:
             logging.warning("Not enough classes/probability shape mismatch for AUC.")
             auc_macro_ovr = 0.0
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        auc_macro_ovr = 0.0

    # Logging Results
    print(f"\n--- Validation Results ---")
    print(f"Mode: {'OpenVINO' if is_openvino else 'PyTorch'}")
    print(f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1(W): {f1_weighted:.4f}, QWK: {qwk:.4f}")
    print(f"Avg Sensitivity: {avg_sensitivity:.4f}, Avg Specificity: {avg_specificity:.4f}, AUC(Macro-OvR): {auc_macro_ovr:.4f}")
    print(f"Confusion Matrix:\n{cm}\n" if cm is not None else "Confusion Matrix: N/A")

    return {
        "loss": avg_loss, "accuracy": acc, "f1_weighted": f1_weighted, "qwk": qwk,
        "avg_sensitivity": avg_sensitivity, "avg_specificity": avg_specificity,
        "auc_macro_ovr": auc_macro_ovr
    }


# def create_calibration_data_transform():
#     def transform_fn(data_item):
#         """Extract image data for quantization calibration."""
#         # Handle different types of data items returned by the dataloader
#         if isinstance(data_item, tuple):
#             if len(data_item) == 3:
#                 images, _, _ = data_item
#             else:
#                 images, _ = data_item
#         else:
#             raise ValueError(f"Unexpected dataloader item format: {type(data_item)}")
        
#         # Ensure images are numpy arrays
#         if isinstance(images, torch.Tensor):
#             images_np = images.cpu().numpy()
#         else:
#             raise ValueError(f"Expected images to be torch.Tensor, got {type(images)}")
        
#         # Return dictionary mapping input name to numpy array
#         input_name = "input"  # Must match ONNX model input name
#         return {input_name: images_np.astype(np.float32)}
    
#     return transform_fn
import numpy as np

def transform_fn(data_item):
    # Assuming data_item is a tuple/list where the first element is the batch of images
    if isinstance(data_item, (tuple, list)):
        images = data_item[0]  # Extract the images from the data_item
    else:
        raise ValueError(f"Unexpected data_item type: {type(data_item)}")
    
    # Iterate over the batch and yield one image at a time
    for img in images:
        # Convert the image to a NumPy array and ensure it's in the correct format
        img_np = img.cpu().numpy().astype(np.float32)
        # Add a batch dimension of 1 to match the model input shape [1, 3, 256, 256]
        input_name = ov_model_fp32.inputs[0].get_any_name()  # Replace with your model's input name
        yield {input_name: img_np[None, ...]}  # Shape becomes [1, 3, 256, 256]

# --- Dummy Data Loading and Preprocessing ---
# !! Replace this with your actual data loading and preprocessing !!
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


IMG_SIZE = 256 
val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

ds_name = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
val_dataloader = UniformValidDataloader(
        dataset_names=ds_name,
        transformation=val_transform,
        batch_size=32,
        num_workers=3
    ).get_loader()


calib_dataloader = UniformTrainDataloader(
    dataset_names=ds_name,
    transformation=val_transform,
    batch_size=1,
    num_workers=3
).get_loader()


if __name__ == "__main__":

    CHECKPOINT_PATH = "good_chkpt/fine_3_local/best_best_clinical_model.pth"
    NUM_CLASSES = 5
    BATCH_SIZE = 1 
    IMG_SIZE = 256
    INPUT_SHAPE = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE) 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Output directories
    OV_MODEL_DIR = "ov_models"
    os.makedirs(OV_MODEL_DIR, exist_ok=True)
    ONNX_MODEL_PATH = os.path.join(OV_MODEL_DIR, "enhanced_dr_classifier.onnx")
    OV_FP32_MODEL_XML = os.path.join(OV_MODEL_DIR, "enhanced_dr_fp32.xml")
    OV_INT8_MODEL_XML = os.path.join(OV_MODEL_DIR, "enhanced_dr_int8.xml")

    # Load PyTorch Model
    print("\n--- Loading PyTorch Model ---")
    pytorch_model = EnhancedDRClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    # Load checkpoint
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE , weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Handle potential DataParallel prefixes
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        load_result = pytorch_model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded checkpoint from {CHECKPOINT_PATH}")

        if load_result.missing_keys:
            print(f"Warning: Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Info: Ignored keys: {load_result.unexpected_keys}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit()

    pytorch_model.eval()

    # Run PyTorch baseline validation
    print("\n--- Running PyTorch Baseline Validation ---")
    baseline_metrics = validate(pytorch_model, val_dataloader, DEVICE, num_classes=NUM_CLASSES, is_openvino=False)

    # Export to ONNX
    print("\n--- Exporting Model to ONNX ---")
    inference_model = InferenceWrapper(pytorch_model).to(DEVICE)
    inference_model.eval()
    dummy_input = torch.randn(INPUT_SHAPE, device=DEVICE)

    try:
        torch.onnx.export(
            inference_model,
            dummy_input,
            ONNX_MODEL_PATH,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        print(f"Model successfully exported to {ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        exit()

    # Convert to OpenVINO IR (FP32)
    print("\n--- Converting ONNX to OpenVINO FP32 IR ---")
    try:
        import openvino as ov
        ov_model_fp32 = ov.convert_model(ONNX_MODEL_PATH)
        ov.save_model(ov_model_fp32, OV_FP32_MODEL_XML, compress_to_fp16=False)
        print(f"Model successfully converted to OpenVINO FP32 IR: {OV_FP32_MODEL_XML}")
    except Exception as e:
        print(f"Error converting to OpenVINO FP32 IR: {e}")
        exit()

    # Validate OpenVINO FP32 Model
    print("\n--- Running OpenVINO FP32 Validation ---")
    try:
        core = ov.Core()
        compiled_model_fp32 = core.compile_model(OV_FP32_MODEL_XML, "CPU")
        fp32_metrics = validate(None, val_dataloader, torch.device('cpu'),
                                num_classes=NUM_CLASSES, is_openvino=True, compiled_model=compiled_model_fp32)
        del compiled_model_fp32
    except Exception as e:
        print(f"Error during OpenVINO FP32 validation: {e}")
        fp32_metrics = None

    # Quantize to OpenVINO INT8 with fixed transform function
    print("\n--- Quantizing Model to OpenVINO INT8 ---")
    try:
        import nncf
        
        # Load the FP32 model for quantization
        ov_model_to_quantize = core.read_model(OV_FP32_MODEL_XML)
        
        # Create calibration dataset with fixed transform function
        calibration_dataset = nncf.Dataset(calib_dataloader, transform_fn)
        quantized_model = nncf.quantize(
            ov_model_to_quantize,
            calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            subset_size=len(calib_dataloader.dataset)
        )
        ov.save_model(quantized_model, OV_INT8_MODEL_XML)
        print(f"Model successfully quantized and saved to {OV_INT8_MODEL_XML}")
        del ov_model_to_quantize, quantized_model
        
    except Exception as e:
        print(f"Error during OpenVINO INT8 quantization: {e}")
        import traceback
        traceback.print_exc()
        OV_INT8_MODEL_XML = None

    # Validate OpenVINO INT8 Model
    if OV_INT8_MODEL_XML and os.path.exists(OV_INT8_MODEL_XML):
        print("\n--- Running OpenVINO INT8 Validation ---")
        try:
            compiled_model_int8 = core.compile_model(OV_INT8_MODEL_XML, "CPU")
            int8_metrics = validate(None, val_dataloader, torch.device('cpu'),
                                  num_classes=NUM_CLASSES, is_openvino=True, compiled_model=compiled_model_int8)
            del compiled_model_int8
        except Exception as e:
            print(f"Error during OpenVINO INT8 validation: {e}")
            int8_metrics = None
    else:
        print("\n--- Skipping OpenVINO INT8 Validation (Quantization failed or skipped) ---")
        int8_metrics = None


    # --- 8. Performance Benchmarking (using benchmark_app) ---
    print("\n--- Running Performance Benchmarking (benchmark_app) ---")
    print("Note: benchmark_app provides throughput and latency estimates.")

    def run_benchmark(model_path, model_name):
        if not model_path or not os.path.exists(model_path):
             print(f"Skipping benchmark for {model_name}: Model file not found at {model_path}")
             return None
        # benchmark_app command (adjust path if not in system PATH)
        # Use -d CPU and potentially -hint throughput or latency
        # -shape can be specified if model has dynamic shapes or you want to override
        command = f"benchmark_app -m \"{model_path}\" -d CPU -api async -hint throughput" # Throughput focus
        # command = f"benchmark_app -m \"{model_path}\" -d CPU -api sync -niter 100 -hint latency" # Latency focus
        print(f"\nBenchmarking {model_name}...")
        print(f"Command: {command}")
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(result.stdout)
            # Extract throughput (example parsing, may need adjustment based on benchmark_app version)
            throughput = None
            for line in result.stdout.splitlines():
                if "Throughput:" in line:
                    try:
                        throughput = float(line.split(":")[1].strip().split(" ")[0])
                        print(f"Extracted Throughput for {model_name}: {throughput:.2f} FPS")
                        break
                    except: pass # Handle parsing errors
            return throughput
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark_app for {model_name}: {e}")
            print(f"Stderr: {e.stderr}")
            return None
        except FileNotFoundError:
             print("Error: benchmark_app not found. Ensure OpenVINO environment is set up correctly and benchmark_app is in PATH.")
             return None

    fp32_throughput = run_benchmark(OV_FP32_MODEL_XML, "OpenVINO FP32")
    int8_throughput = run_benchmark(OV_INT8_MODEL_XML, "OpenVINO INT8")


    # --- 9. Summarize Results ---
    print("\n\n--- Optimization Summary ---")
    print("-" * 50)
    print("Metric        | PyTorch (Baseline) | OpenVINO FP32     | OpenVINO INT8")
    print("-" * 50)
    metrics_keys = ["accuracy", "f1_weighted", "qwk", "auc_macro_ovr", "avg_sensitivity", "avg_specificity"]
    all_metrics = {"PyTorch": baseline_metrics, "OV_FP32": fp32_metrics, "OV_INT8": int8_metrics}

    for key in metrics_keys:
        p_val = f"{all_metrics['PyTorch'].get(key, 'N/A'):.4f}" if all_metrics['PyTorch'] else "N/A"
        fp32_val = f"{all_metrics['OV_FP32'].get(key, 'N/A'):.4f}" if all_metrics['OV_FP32'] else "N/A"
        int8_val = f"{all_metrics['OV_INT8'].get(key, 'N/A'):.4f}" if all_metrics['OV_INT8'] else "N/A"
        print(f"{key:<14}| {p_val:<18} | {fp32_val:<17} | {int8_val}")

    print("-" * 50)
    print("Performance     |                    |                   |")
    print("-" * 50)
    fp32_perf = f"{fp32_throughput:.2f} FPS" if fp32_throughput else "N/A"
    int8_perf = f"{int8_throughput:.2f} FPS" if int8_throughput else "N/A"
    print(f"{'Throughput':<14}| N/A                | {fp32_perf:<17} | {int8_perf}")
    print("-" * 50)