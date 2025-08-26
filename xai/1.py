import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
import os
from matplotlib.colors import LinearSegmentedColormap
import timm
from captum.attr import IntegratedGradients, GradientShap
# import matplotlib
# os.environ['QT_DEBUG_PLUGINS'] = '0'
# matplotlib.use('Qt5Agg')
# Import your XaiVisual class - it's already defined in your paste.txt
from xai.visualizer import XaiVisual


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
        # Multiply gradient by -alpha, pass None for alpha's gradient
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
        # Ordinal regression part (predicts thresholds/cumulative logits)
        self.ordinal_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_grades - 1) # Predict K-1 thresholds for K classes
        )

    def forward(self, x):
        logits = self.grade_predictor(x)
        # Ensure ordinal thresholds are monotonically increasing (optional but good practice)
        # Here, we directly predict them. Can be post-processed if needed.
        ordinal_thresholds = self.ordinal_encoder(x)
        return logits, ordinal_thresholds
    


class EnhancedDRClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        super(EnhancedDRClassifier, self).__init__()
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # moco_state_dict = checkpoint['model_state_dict']
        # config = checkpoint['config']
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0)
        
        # backbone_state_dict = {k.replace('query_encoder.', ''): v for k, v in moco_state_dict.items() if k.startswith('query_encoder.')}
        # self.backbone.load_state_dict(backbone_state_dict)
        
        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

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
        
        # self._initialize_weights()
        
    def _initialize_weights(self):
        for module in [self.classifier, self.grade_head.grade_predictor, self.domain_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d): 
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x, alpha=0.0, get_attention=False, update_prototypes=False, labels=None):
        features = self.backbone.forward_features(x)
        attended_features = self.attention(features)
        h = torch.mean(attended_features, dim=(2, 3))
        
        logits = self.classifier(h)
        grade_probs = self.grade_head(h)
        
        if update_prototypes and labels is not None:
            with torch.no_grad():
                for i, label in enumerate(labels):
                    self.prototypes[label] = self.prototypes[label] * (self.prototype_counts[label] / (self.prototype_counts[label] + 1)) + \
                                           h[i] * (1 / (self.prototype_counts[label] + 1))
                    self.prototype_counts[label] += 1
        
        if alpha > 0:
            reversed_features = GradientReversal.apply(h, alpha)
            domain_logits = self.domain_classifier(reversed_features)
        else:
            domain_logits = None
        
        if get_attention:
            return logits, grade_probs, domain_logits, h, attended_features
        return logits, grade_probs, domain_logits
    



# Model wrapper for Captum compatibility
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        logits, _, _, h, _ = self.model(x, get_attention=True)
        return logits

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        # Correctly access the last conv layer in ConvNeXt architecture
        if target_layer is None:
            # ConvNeXt architecture - access the last block in the last stage
            self.target_layer = model.backbone.stages[-1].blocks[-1]
        else:
            self.target_layer = target_layer
            
        self.hooks = []
        self.gradients = None
        self.activations = None
        self.register_hooks()

        
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        logits, _, _, _, _ = self.model(input_tensor, get_attention=True)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        
        # Backward pass
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights and activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Create CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        self.remove_hooks()
        
        return cam, target_class, torch.softmax(logits, dim=1)[0, target_class].item()

# Function to preprocess images
def preprocess_image(image_path, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor, original_image

# Integrated Gradients
def compute_integrated_gradients(model_wrapper, input_tensor, target_class=None, steps=50):
    logits = model_wrapper(input_tensor)
    
    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()
    
    ig = IntegratedGradients(model_wrapper)
    baseline = torch.zeros_like(input_tensor)
    attributions = ig.attribute(input_tensor, baseline, target=target_class, n_steps=steps)
    
    return attributions, target_class, torch.softmax(logits, dim=1)[0, target_class].item()

# SHAP values
def compute_shap_values(model_wrapper, input_tensor, target, background=None, n_samples=50):
    if background is None:
        background = torch.zeros((10,) + input_tensor.shape[1:], device=input_tensor.device)
    
    gs = GradientShap(model_wrapper)
    shap_values = gs.attribute(input_tensor, background, target=target, n_samples=n_samples)
    
    return shap_values

# Monte Carlo Dropout for uncertainty estimation
def monte_carlo_dropout(model, input_tensor, n_samples=30):
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    
    model.eval()  # Set model to evaluation mode
    model.apply(enable_dropout)  # But enable dropout
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits, _, _, _, _ = model(input_tensor, get_attention=True)
            predictions.append(logits)
    
    predictions = torch.stack(predictions)
    mean_pred = torch.mean(predictions, dim=0)
    std_pred = torch.std(predictions, dim=0)
    
    return mean_pred, std_pred

# Attention map visualization (specific to your model's attention mechanism)
def visualize_attention_map(model, input_tensor):
    model.eval()
    with torch.no_grad():
        _, _, _, _, attended_features = model(input_tensor, get_attention=True)
    
    # Get attention map from the attended features
    attention_map = torch.mean(attended_features, dim=1).squeeze().cpu().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    attention_map = cv2.resize(attention_map, (input_tensor.shape[3], input_tensor.shape[2]))
    
    return attention_map

# Main function to run all XAI analyses
def run_xai_analysis(model, image_path, output_dir="xai_results", class_names=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if class_names is None:
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    visualizer = XaiVisual(class_names=class_names)
    
    # # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(next(model.parameters()).device)
    
    # Create a wrapper for Captum
    model_wrapper = ModelWrapper(model)
    
    # Dictionary to store all results
    results = {
        'original_image': original_image
    }
    
    # 1. Run GradCAM
    print("Computing GradCAM...")
    gradcam = GradCAM(model)
    cam, pred_class, confidence = gradcam.generate_cam(input_tensor)
    results['gradcam'] = cam
    results['prediction'] = pred_class
    results['confidence'] = confidence
    
    visualizer.plot_gradcam(
        original_image, 
        cam, 
        prediction=pred_class, 
        confidence=confidence,
        save_path=os.path.join(output_dir, "gradcam.png")
    )
    
    # 2. Run Integrated Gradients
    print("Computing Integrated Gradients...")
    attributions, _, _ = compute_integrated_gradients(model_wrapper, input_tensor, target_class=pred_class)
    results['integrated_gradients'] = attributions
    
    visualizer.plot_integrated_gradients(
        original_image,
        attributions,
        save_path=os.path.join(output_dir, "integrated_gradients.png")
    )
    
    # 3. Run SHAP
    print("Computing SHAP values...")
    shap_values = compute_shap_values(model_wrapper, input_tensor, target=pred_class)
    results['shap_values'] = shap_values
    
    visualizer.plot_shap_values(
        original_image,
        shap_values,
        save_path=os.path.join(output_dir, "shap.png")
    )
    
    # 4. Run Monte Carlo Dropout for uncertainty
    print("Computing uncertainty...")
    mean_pred, std_pred = monte_carlo_dropout(model, input_tensor)
    results['uncertainty'] = std_pred
    
    visualizer.plot_uncertainty(
        original_image,
        mean_pred,
        std_pred,
        save_path=os.path.join(output_dir, "uncertainty.png")
    )
    
    # 5. Visualize model's attention mechanism
    # print("Visualizing attention map...")
    # attention_map = visualize_attention_map(model, input_tensor)
    # plt.imshow(original_image)
    # plt.imshow(attention_map, cmap='jet', alpha=0.5)  
    # plt.show()
        
    # 6. Create summary visualization
    print("Creating summary visualization...")
    visualizer.create_summary_visualization(
        results,
        save_path=os.path.join(output_dir, "summary.png")
    )
    
    print(f"XAI analysis complete. Results saved to {output_dir}")
    return results

def free_gpu_memory():
    """Release unused GPU memory to avoid fragmentation."""
    torch.cuda.empty_cache()
    
# Call this at the beginning of your run_xai_analysis function



# Example usage
if __name__ == "__main__":
    import argparse
    free_gpu_memory()
    parser = argparse.ArgumentParser(description='Run XAI analysis on a DR classifier model')
    parser.add_argument('--image', type=str, default="data/aptos/train_images/001639a390f0.png", help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='xai_results', help='Directory to save results')
    parser.add_argument('--checkpoint', type=str, default="good_chkpt/fine_3_local/best_best_clinical_model.pth", 
                      help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = EnhancedDRClassifier(num_classes=5, freeze_backbone=False).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device , weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Class names for DR grades
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    
    # Run XAI analysis
    results = run_xai_analysis(model, args.image, args.output_dir, class_names)
