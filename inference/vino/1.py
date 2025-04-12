import torch
import numpy as np
import openvino as ov
import os
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

import timm
import torch.nn as nn
from torch.nn import functional as F

# Your model definition classes here
class LesionAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(LesionAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        # Spatial attention
        assert kernel_size % 2 == 1, "Kernel size must be odd for spatial attention"
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x_channel = x * channel_att

        # Spatial Attention
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
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_grades)
        )
        # Ordinal regression part
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
    def __init__(self, num_classes=5, freeze_backbone=True):
        super(EnhancedDRClassifier, self).__init__()
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0)
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
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)
        )
        
        self.register_buffer('prototypes', torch.zeros(num_classes, self.feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        
    def forward(self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None):
        features = self.backbone.forward_features(x)
        attended_features = self.attention(features)
        h = torch.mean(attended_features, dim=(2, 3))
        
        logits = self.classifier(h)
        grade_logits, ordinal_thresholds = self.grade_head(h)
        
        # For inference optimization, we'll exclude the prototype updating and domain adaptation
        return logits, (grade_logits, ordinal_thresholds), None

    def inference_forward(self, x):
        # Create a simpler forward pass for inference
        features = self.backbone.forward_features(x)
        attended_features = self.attention(features)
        h = torch.mean(attended_features, dim=(2, 3))
        
        logits = self.classifier(h)
        grade_logits, ordinal_thresholds = self.grade_head(h)
        
        return logits, grade_logits

# Create a class for inference-only model
class DRInferenceModel(nn.Module):
    def __init__(self, full_model):
        super(DRInferenceModel, self).__init__()
        self.full_model = full_model
        
    def forward(self, x):
        with torch.no_grad():
            logits, grade_logits = self.full_model.inference_forward(x)
            class_probs = F.softmax(logits, dim=1)
            grade_probs = F.softmax(grade_logits, dim=1)
            return class_probs, grade_probs

# Main optimization script
def main():
    device = torch.device("cpu")  # Use CPU for OpenVINO optimization
    
    # Path to your checkpoint
    checkpoint_path = "good_chkpt/fine_3_local/best_clinical_checkpoint.pth" 
    output_dir = "openvino_optimized_model"
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Loading PyTorch model...")
    # Initialize model
    model = EnhancedDRClassifier(num_classes=5, freeze_backbone=False).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) 
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create inference-only model wrapper
    inference_model = DRInferenceModel(model)
    
    # Set model to evaluation mode
    inference_model.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess image with proper batch dimension
    image_path = "data/aptos/train_images/001639a390f0.png"
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    dummy_input = input_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
    
    print("2. Tracing model to TorchScript...")
    try:
        traced_model = torch.jit.trace(inference_model, dummy_input)
        torch_script_path = os.path.join(output_dir, "model_traced.pt")
        traced_model.save(torch_script_path)
        print(f"TorchScript model saved to {torch_script_path}")
    except Exception as e:
        print(f"Error in TorchScript tracing: {e}")
        torch_script_path = None
    
    print("3. Converting to OpenVINO IR format...")
    ov_model_path = os.path.join(output_dir, "model.xml")
    
    try:
        if torch_script_path:
            ov_model = ov.convert_model(torch_script_path, input=[ov.preprocess.InputTensorInfo(
                dummy_input.shape, 
                element_type=ov.Type.f32,
                layout=ov.Layout("NCHW")
            )])
        else:
            ov_model = ov.convert_model(
                inference_model,
                example_input=dummy_input,
                input=[ov.preprocess.InputTensorInfo(
                    dummy_input.shape, 
                    element_type=ov.Type.f32,
                    layout=ov.Layout("NCHW")
                )]
            )
        
        ov.save_model(ov_model, ov_model_path)
        print(f"OpenVINO IR model saved to {ov_model_path}")
    except Exception as e:
        print(f"Error in OpenVINO conversion: {e}")
        print("Using ONNX as intermediary format...")
        
        onnx_path = os.path.join(output_dir, "model.onnx")
        try:
            # Export to ONNX with proper batch dimension
            torch.onnx.export(
                inference_model,
                dummy_input,  # Already has batch dimension
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["class_probs", "grade_probs"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "class_probs": {0: "batch_size"},
                    "grade_probs": {0: "batch_size"}
                }
            )
            print(f"ONNX model saved to {onnx_path}")
            
            ov_model = ov.convert_model(onnx_path)
            ov.save_model(ov_model, ov_model_path)
            print(f"OpenVINO IR model saved to {ov_model_path}")
        except Exception as e:
            print(f"Error in ONNX conversion: {e}")
            raise
    
    print("4. Optimizing OpenVINO model for CPU...")
    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")
    
    compiled_model = core.compile_model(ov_model, "CPU")
    optimized_path = os.path.join(output_dir, "model_optimized.xml")
    ov.save_model(compiled_model, optimized_path, compress_to_fp16=True)
    print(f"Optimized OpenVINO model saved to {optimized_path}")
    
    print("5. Testing inference performance...")
    infer_request = compiled_model.create_infer_request()
    
    # Use numpy array with batch dimension
    input_tensor = dummy_input.numpy()
    
    print("Warming up...")
    for _ in range(10):
        infer_request.infer(inputs=[input_tensor])
    
    num_iterations = 100
    print(f"Running {num_iterations} iterations for benchmark...")
    start_time = time.time()
    for _ in tqdm(range(num_iterations)):
        infer_request.infer(inputs=[input_tensor])
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_iterations * 1000
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"Throughput: {1000 / avg_inference_time:.2f} FPS")
    
    # Update inference script with proper batch dimension handling
    inference_script_path = os.path.join(output_dir, "inference.py")
    with open(inference_script_path, 'w') as f:
        f.write('''
import openvino as ov
import numpy as np
import time
from PIL import Image
import torch
from torchvision import transforms
import argparse

def load_and_preprocess_image(image_path, input_size=(224, 224)):
    """Load and preprocess an image for model inference"""
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor.numpy()

def run_inference(model_path, image_path):
    """Run inference using OpenVINO optimized model"""
    core = ov.Core()
    print(f"Loading model from {model_path}")
    compiled_model = core.compile_model(model_path, "CPU")
    
    input_layer = compiled_model.inputs[0]
    output_layers = compiled_model.outputs
    
    input_tensor = load_and_preprocess_image(image_path)
    
    infer_request = compiled_model.create_infer_request()
    
    start_time = time.time()
    results = infer_request.infer(inputs=[input_tensor])
    end_time = time.time()
    
    class_probs = results[output_layers[0]][0]  # Adjust indexing based on output
    grade_probs = results[output_layers[1]][0]  # Adjust indexing based on output
    
    predicted_class = np.argmax(class_probs)
    predicted_grade = np.argmax(grade_probs)
    
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {class_probs}")
    print(f"Predicted grade: {predicted_grade}")
    print(f"Grade probabilities: {grade_probs}")
    
    return {
        "predicted_class": predicted_class,
        "class_probs": class_probs,
        "predicted_grade": predicted_grade,
        "grade_probs": grade_probs,
        "inference_time_ms": (end_time - start_time) * 1000
    }

def main():
    parser = argparse.ArgumentParser(description='Run inference with OpenVINO optimized DR model')
    parser.add_argument('--model', type=str, default='model_optimized.xml', help='Path to OpenVINO model XML file')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    
    args = parser.parse_args()
    results = run_inference(args.model, args.image)

if __name__ == "__main__":
    main()
''')
    

    print(f"Inference helper script saved to {inference_script_path}")
    print("\nOptimization complete! Your model has been optimized for CPU inference using OpenVINO.")
    print("\nTo run inference on a new image, use:")
    print(f"python {inference_script_path} --model {output_dir}/model_optimized.xml --image your_image.jpg")

if __name__ == "__main__":
    main()