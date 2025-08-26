import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
import os
from matplotlib.colors import LinearSegmentedColormap
import timm
# Make sure captum is installed if you uncomment those sections: pip install captum
# from captum.attr import IntegratedGradients, GradientShap
import matplotlib

# Comment out or remove this line if you don't need Qt5Agg specifically
# os.environ['QT_DEBUG_PLUGINS'] = '0'
# Use a non-interactive backend like 'Agg' if you *only* want to save figures
# and never display them interactively during the script run.
matplotlib.use('Agg') # Use Agg backend for saving figures without showing windows

# Import your XaiVisual class - assume it's defined elsewhere or in paste.txt
# from xai.visualizer import XaiVisual # Assuming this class exists

# --- Your other class definitions (LesionAttentionModule, etc.) go here ---
# (Keep the class definitions as you provided them)
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
        self.backbone = timm.create_model("convnext_small", pretrained=False, num_classes=0) # Use num_classes=0 to get features

        # backbone_state_dict = {k.replace('query_encoder.', ''): v for k, v in moco_state_dict.items() if k.startswith('query_encoder.')}
        # self.backbone.load_state_dict(backbone_state_dict)

        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

        # Ensure self.feature_dim reflects the actual output dimension of the backbone
        # For convnext_small, num_features is typically 768
        self.feature_dim = self.backbone.num_features # Correctly get feature dimension
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
            nn.Linear(256, 5) # Assuming 5 domains or adapt as needed
        )

        self.register_buffer('prototypes', torch.zeros(num_classes, self.feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))

        # self._initialize_weights() # Call this if needed

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
        features = self.backbone.forward_features(x) # Use forward_features to get the feature map
        attended_features = self.attention(features)
        # Global average pooling after attention
        h = torch.mean(attended_features, dim=(2, 3))

        logits = self.classifier(h)
        grade_outputs = self.grade_head(h) # Changed variable name for clarity

        if update_prototypes and labels is not None:
            with torch.no_grad():
                for i, label in enumerate(labels):
                    label_idx = label.item() # Ensure label is an index
                    self.prototypes[label_idx] = self.prototypes[label_idx] * (self.prototype_counts[label_idx] / (self.prototype_counts[label_idx] + 1)) + \
                                               h[i] * (1 / (self.prototype_counts[label_idx] + 1))
                    self.prototype_counts[label_idx] += 1

        domain_logits = None
        if alpha > 0:
            reversed_features = GradientReversal.apply(h, alpha)
            domain_logits = self.domain_classifier(reversed_features)

        if get_attention:
            # Return attended_features which holds the spatial attention map before pooling
            return logits, grade_outputs, domain_logits, h, attended_features
        return logits, grade_outputs, domain_logits


# --- ModelWrapper and other XAI methods (GradCAM, IG, SHAP, MCDropout) ---
# (Keep these as they were, uncomment if needed)
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Adjust this based on what your XAI method expects (e.g., just logits)
        logits, _, _, _, _ = self.model(x, get_attention=True)
        return logits

# --- Placeholder for XaiVisual if not imported ---
class XaiVisual:
    def __init__(self, class_names=None):
        self.class_names = class_names if class_names else ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
        self.cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'green', 'yellow', 'red']) # Example colormap

    def _overlay_heatmap(self, image, heatmap, alpha=0.5, cmap='jet'):
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # Resize heatmap to match image dimensions if needed
        if heatmap.shape[:2] != image.shape[:2]:
             heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        overlayed_image = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlayed_image

    def plot_gradcam(self, original_image, cam, prediction=None, confidence=None, save_path=None):
        plt.figure(figsize=(8, 6))
        overlay = self._overlay_heatmap(original_image, cam)
        plt.imshow(overlay)
        title = "Grad-CAM"
        if prediction is not None:
            title += f"\nPredicted: {self.class_names[prediction]}"
        if confidence is not None:
            title += f" (Conf: {confidence:.2f})"
        plt.title(title)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Grad-CAM saved to {save_path}")
        else:
            plt.show()
        plt.close()

    # Add stubs for other methods if you plan to use them
    def plot_integrated_gradients(self, original_image, attributions, save_path=None):
        print("Plotting Integrated Gradients (stub - implement visualization)")
        # Basic sum over channels and absolute value visualization
        attr_viz = np.sum(np.abs(attributions.squeeze().cpu().numpy()), axis=0)
        attr_viz = (attr_viz - np.min(attr_viz)) / (np.max(attr_viz) - np.min(attr_viz) + 1e-8)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(attr_viz, cmap='inferno')
        plt.title("Integrated Gradients Attribution")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Integrated Gradients saved to {save_path}")
        else:
            plt.show()
        plt.close()


    def plot_shap_values(self, original_image, shap_values, save_path=None):
        print("Plotting SHAP (stub - implement visualization)")
         # Basic sum over channels and absolute value visualization
        shap_viz = np.sum(np.abs(shap_values.squeeze().cpu().numpy()), axis=0)
        shap_viz = (shap_viz - np.min(shap_viz)) / (np.max(shap_viz) - np.min(shap_viz) + 1e-8)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(shap_viz, cmap='viridis')
        plt.title("SHAP Values Attribution")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"SHAP saved to {save_path}")
        else:
            plt.show()
        plt.close()


    def plot_uncertainty(self, original_image, mean_pred, std_pred, save_path=None):
        print("Plotting Uncertainty (stub - implement visualization)")
        # Example: Plot image and bar chart of std deviations per class
        mean_probs = torch.softmax(mean_pred, dim=1).squeeze().cpu().numpy()
        std_devs = std_pred.squeeze().cpu().numpy()
        pred_class = np.argmax(mean_probs)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f"Predicted: {self.class_names[pred_class]} (Mean Prob: {mean_probs[pred_class]:.2f})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.bar(self.class_names, std_devs)
        plt.ylabel("Standard Deviation (Uncertainty)")
        plt.title("Prediction Uncertainty (MC Dropout)")
        plt.xticks(rotation=45, ha='right')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Uncertainty plot saved to {save_path}")
        else:
            plt.show()
        plt.close()


    def create_summary_visualization(self, results, save_path=None):
        print("Creating Summary Visualization (stub - implement layout)")
        # Example layout (adapt based on available results)
        num_plots = len(results) # Count how many items are in results
        if num_plots <= 1: # Need at least original image + 1 result
             print("Not enough results to create a summary plot.")
             return

        # Dynamically create subplot grid (e.g., 2 columns)
        cols = 2
        rows = (num_plots + cols -1) // cols # Calculate rows needed

        plt.figure(figsize=(6 * cols, 5 * rows))
        plot_index = 1

        if 'original_image' in results:
            plt.subplot(rows, cols, plot_index)
            plt.imshow(results['original_image'])
            title = "Original Image"
            if 'prediction' in results and 'confidence' in results:
                 pred_class = results['prediction']
                 conf = results['confidence']
                 title += f"\nPred: {self.class_names[pred_class]} (Conf: {conf:.2f})"
            plt.title(title)
            plt.axis('off')
            plot_index += 1

        if 'gradcam' in results:
             plt.subplot(rows, cols, plot_index)
             overlay = self._overlay_heatmap(results['original_image'], results['gradcam'])
             plt.imshow(overlay)
             plt.title("Grad-CAM")
             plt.axis('off')
             plot_index += 1

        # Add similar blocks for 'integrated_gradients', 'shap_values', 'uncertainty', 'attention_map_viz' if they exist in results
        if 'integrated_gradients' in results:
            plt.subplot(rows, cols, plot_index)
            attr_viz = np.sum(np.abs(results['integrated_gradients'].squeeze().cpu().numpy()), axis=0)
            attr_viz = (attr_viz - np.min(attr_viz)) / (np.max(attr_viz) - np.min(attr_viz) + 1e-8)
            plt.imshow(attr_viz, cmap='inferno')
            plt.title("Integrated Gradients")
            plt.axis('off')
            plot_index += 1

        if 'shap_values' in results:
            plt.subplot(rows, cols, plot_index)
            shap_viz = np.sum(np.abs(results['shap_values'].squeeze().cpu().numpy()), axis=0)
            shap_viz = (shap_viz - np.min(shap_viz)) / (np.max(shap_viz) - np.min(shap_viz) + 1e-8)
            plt.imshow(shap_viz, cmap='viridis')
            plt.title("SHAP")
            plt.axis('off')
            plot_index += 1

        if 'attention_map_viz' in results: # Check for the visualization map
            plt.subplot(rows, cols, plot_index)
            plt.imshow(results['attention_map_viz']) # Display the pre-rendered overlay
            plt.title("Model Attention")
            plt.axis('off')
            plot_index += 1

        # Add Uncertainty plot if available (might need specific handling)
        if 'uncertainty' in results and 'mean_pred' in results: # Need both for context
             plt.subplot(rows, cols, plot_index)
             mean_probs = torch.softmax(results['mean_pred'], dim=1).squeeze().cpu().numpy()
             std_devs = results['uncertainty'].squeeze().cpu().numpy()
             plt.bar(self.class_names, std_devs)
             plt.ylabel("Std Dev")
             plt.title("Uncertainty")
             plt.xticks(rotation=45, ha='right')
             # plot_index += 1 # Only increment if plot added

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Summary visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()


# --- Preprocessing and other helper functions ---
def preprocess_image(image_path, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image.resize((224,224))) # Resize original too for consistent overlay size
    input_tensor = transform(image).unsqueeze(0)

    return input_tensor, original_image

# Attention map visualization (specific to your model's attention mechanism)
def visualize_attention_map(model, input_tensor):
    model.eval()
    with torch.no_grad():
        # Ensure get_attention=True returns the attended_features
        # Check your EnhancedDRClassifier forward method's return values when get_attention=True
        _, _, _, _, attended_features = model(input_tensor, get_attention=True)

    # Calculate attention map from the attended features
    # This assumes attended_features is the output *after* spatial attention
    # If LesionAttentionModule returns x_channel * spatial_att, the spatial_att part is what we need
    # Let's recalculate spatial_att for visualization if needed, or assume attended_features directly reflects it.
    # Assuming attended_features = features * channel_att * spatial_att.
    # We visualize the effect, so averaging channels of attended_features is reasonable.

    attention_map = torch.mean(attended_features, dim=1).squeeze().cpu().numpy()

    # Normalize for visualization
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

    # Resize to match input image dimensions (e.g., 224x224)
    attention_map = cv2.resize(attention_map, (input_tensor.shape[3], input_tensor.shape[2]), interpolation=cv2.INTER_LINEAR)

    return attention_map


# Main function to run all XAI analyses
def run_xai_analysis(model, image_path, output_dir="xai_results", class_names=None):
    os.makedirs(output_dir, exist_ok=True)

    if class_names is None:
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    # Initialize the visualizer (make sure the class definition is available)
    visualizer = XaiVisual(class_names=class_names)

    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(next(model.parameters()).device)

    # --- Initialize results dictionary ---
    # (Uncomment parts below as you implement/uncomment the corresponding XAI methods)
    results = {
        'original_image': original_image
    }

    # --- Run the XAI methods you need ---
    # (Example: If you uncomment GradCAM computation, add results['gradcam'] = cam etc.)

    # # 1. Run GradCAM (Example - uncomment if needed)
    # print("Computing GradCAM...")
    # model_wrapper = ModelWrapper(model) # Needed for Captum methods like GradCAM usually
    # gradcam = GradCAM(model) # Ensure GradCAM class is defined/imported
    # cam, pred_class, confidence = gradcam.generate_cam(input_tensor)
    # results['gradcam'] = cam
    # results['prediction'] = pred_class
    # results['confidence'] = confidence
    # visualizer.plot_gradcam(original_image, cam, prediction=pred_class, confidence=confidence, save_path=os.path.join(output_dir, "gradcam.png"))

    # # 2. Run Integrated Gradients (Example - uncomment if needed)
    # print("Computing Integrated Gradients...")
    # # Ensure compute_integrated_gradients function is defined/imported
    # attributions, ig_pred_class, _ = compute_integrated_gradients(model_wrapper, input_tensor, target_class=results.get('prediction')) # Use predicted class if available
    # results['integrated_gradients'] = attributions
    # if 'prediction' not in results: results['prediction'] = ig_pred_class # Store prediction if not already done
    # visualizer.plot_integrated_gradients(original_image, attributions, save_path=os.path.join(output_dir, "integrated_gradients.png"))

    # # 3. Run SHAP (Example - uncomment if needed)
    # print("Computing SHAP values...")
    # # Ensure compute_shap_values function is defined/imported
    # shap_values = compute_shap_values(model_wrapper, input_tensor, target=results.get('prediction')) # Use predicted class
    # results['shap_values'] = shap_values
    # visualizer.plot_shap_values(original_image, shap_values, save_path=os.path.join(output_dir, "shap.png"))

    # # 4. Run Monte Carlo Dropout (Example - uncomment if needed)
    # print("Computing uncertainty...")
    # # Ensure monte_carlo_dropout function is defined/imported
    # mean_pred, std_pred = monte_carlo_dropout(model, input_tensor)
    # results['mean_pred'] = mean_pred # Store for context if needed
    # results['uncertainty'] = std_pred
    # visualizer.plot_uncertainty(original_image, mean_pred, std_pred, save_path=os.path.join(output_dir, "uncertainty.png"))


    # 5. Visualize model's attention mechanism and SAVE it
    print("Visualizing and saving attention map...")
    attention_map = visualize_attention_map(model, input_tensor)
    results['attention_map'] = attention_map # Store the raw map data

    # Create the plot overlay
    plt.figure(figsize=(6, 6)) # Create a new figure
    plt.imshow(original_image)
    plt.imshow(attention_map, cmap='jet', alpha=0.5) # Overlay heatmap
    plt.axis('off') # Hide axes
    plt.title('Model Attention Map Overlay')

    # Define save path
    attention_map_save_path = os.path.join(output_dir, "attention_map_overlay.png")

    # Save the figure
    plt.savefig(attention_map_save_path, bbox_inches='tight', pad_inches=0)
    plt.close() # Close the figure to free memory
    print(f"Attention map overlay saved to {attention_map_save_path}")
    results['attention_map_path'] = attention_map_save_path # Store path
    # Also create the overlay image array for summary plot if needed
    results['attention_map_viz'] = visualizer._overlay_heatmap(original_image, attention_map)


    # 6. Create summary visualization (IF other results are generated)
    if len(results) > 3: # Check if more than just original, attention map, and path are present
        print("Creating summary visualization...")
        visualizer.create_summary_visualization(
            results,
            save_path=os.path.join(output_dir, "summary.png")
        )
    else:
        print("Skipping summary visualization as only attention map was generated.")

    print(f"XAI analysis complete. Results saved to {output_dir}")
    return results

def free_gpu_memory():
    """Release unused GPU memory to avoid fragmentation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Example usage
if __name__ == "__main__":
    import argparse
    free_gpu_memory() # Call at the start
    """
    0 -> 002c21358ce6
    1 -> 00cb6555d108
    2 -> 000c1434d8d7
    3 -> 0104b032c141
    4 -> 001639a390f0
    """
    parser = argparse.ArgumentParser(description='Run XAI analysis on a DR classifier model')
    parser.add_argument('--image', type=str, default="data/aptos/train_images/001639a390f0.png", help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='xai_results_saved', help='Directory to save results') # Changed default name
    parser.add_argument('--checkpoint', type=str, default="good_chkpt/fine_3_local/best_best_clinical_model.pth",
                        help='Path to model checkpoint')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    # Set freeze_backbone=False if you fine-tuned the backbone
    model = EnhancedDRClassifier(num_classes=5, freeze_backbone=False).to(device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
         print(f"Error: Checkpoint file not found at {args.checkpoint}")
         exit()
    if not os.path.exists(args.image):
         print(f"Error: Image file not found at {args.image}")
         exit()

    try:
        # Try loading with map_location first
        checkpoint = torch.load(args.checkpoint, map_location=device , weights_only=False)
        # Check if 'model_state_dict' exists, otherwise assume the whole file is the state dict
        if "model_state_dict" in checkpoint:
             model.load_state_dict(checkpoint["model_state_dict"])
        else:
             model.load_state_dict(checkpoint) # Directly load if no key
        print("Checkpoint loaded successfully.")
    except Exception as e:
         print(f"Error loading checkpoint: {e}")
         # Consider adding `weights_only=True` if trust source and face issues
         # try:
         #     checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
         #     # Load state dict as before
         # except Exception as e2:
         #      print(f"Error loading checkpoint even with weights_only=True: {e2}")
         #      exit()
         exit()


    model.eval()

    # Class names for DR grades
    class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

    # Run XAI analysis
    results = run_xai_analysis(model, args.image, args.output_dir, class_names)

    free_gpu_memory() # Call at the end