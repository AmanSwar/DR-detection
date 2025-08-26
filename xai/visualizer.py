import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap
import torch
from torchvision import transforms
import shap


class XaiVisual:

    def __init__(self , class_names=None):
        self.class_names = class_names or [f'Class {i}' for i in range(5)]
        colors = [(0.6, 0.6, 0.6, 0), (1, 0, 0, 1)]
        self.heatmap_cmap = LinearSegmentedColormap.from_list('custom', colors)

    def prep_image(self,  img_tensor):
        if isinstance(img_tensor, torch.Tensor):
            image = img_tensor.cpu().numpy()
            if image.shape[0] == 1 or image.shape[0] == 3:  # CHW format
                image = np.transpose(image, (1, 2, 0))
            if image.shape[2] == 1:  # Grayscale to RGB
                image = np.repeat(image, 3, axis=2)
            # Normalize to [0,1] range
            image = (image - image.min()) / (image.max() - image.min())
        return image
    

    def plot_gradcam(self, original_image, gradcam_mask, prediction=None, confidence=None, save_path="xai/saves/gradcam", fig_size=(12, 4)):
        # image = self.preprocess_image(original_image)
        image = original_image
        plt.figure(figsize=fig_size)
        
        """taken from claude"""
        # Original image
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        if prediction is not None and confidence is not None:
            plt.xlabel(f'Prediction: {self.class_names[prediction]}\nConfidence: {confidence:.2f}')
        plt.axis('off')
        
        # Grad-CAM heatmap
        plt.subplot(132)
        plt.imshow(gradcam_mask, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        # Overlay
        plt.subplot(133)
        plt.imshow(image)
        plt.imshow(gradcam_mask, cmap='jet', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_integrated_gradients(self, original_image, attributions, save_path="xai/saves/intgrad", fig_size=(12, 4)):
        # image = self.preprocess_image(original_image)
        image = original_image
        attr_map = attributions.sum(dim=1).squeeze().cpu().numpy()

        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())

        plt.figure(figsize=fig_size)
        
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(attr_map, cmap='seismic')
        plt.title('Attribution Map')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(image)
        plt.imshow(attr_map, cmap='seismic', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

        
    def plot_shap_values(self, original_image, shap_values, save_path="xai/saves/shap", fig_size=(12, 4)):
        image = original_image
        
        # Handle torch.Tensor or numpy array input
        if isinstance(shap_values, torch.Tensor):
            shap_img = torch.abs(shap_values).sum(dim=1).squeeze().cpu().numpy()  # Sum over channels
        else:
            shap_img = np.abs(shap_values).sum(axis=1).squeeze()  # Sum over channels for numpy array
        
        # Normalize to [0, 1] for visualization
        shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
        
        plt.figure(figsize=fig_size)
        
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(shap_img, cmap='hot')
        plt.title('SHAP Values')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(image)
        plt.imshow(shap_img, cmap='hot', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    def plot_uncertainty(self, original_image, mean_pred, std_pred, save_path="xai/saves/uncertainty", fig_size=(12, 4)):
        """
        Visualize the original image, prediction uncertainty, and class probabilities.
        
        Args:
            original_image: Input image tensor or numpy array.
            mean_pred: Mean predictions tensor of shape (1, num_classes).
            std_pred: Standard deviation of predictions, shape (1, num_classes).
            save_path: Path to save the plot (optional).
            fig_size: Figure size as (width, height).
        """
        plt.figure(figsize=fig_size)

        # Subplot 1: Original image with predicted class
        plt.subplot(131)
        plt.imshow(original_image)  # Ensure original_image is in a displayable format (e.g., numpy, HxWxC)
        pred_class = torch.argmax(mean_pred).item()
        plt.title(f'Prediction: {self.class_names[pred_class]}')
        plt.axis('off')

        # Subplot 2: Uncertainty bar plot
        plt.subplot(132)
        uncertainties = std_pred.squeeze().cpu().numpy()  # Shape: (num_classes,)
        y_pos = np.arange(len(self.class_names))
        plt.barh(y_pos, uncertainties, align='center')
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Uncertainty (Std Dev)')
        plt.title('Prediction Uncertainty')

        # Subplot 3: Class probabilities
        plt.subplot(133)
        probs = torch.softmax(mean_pred, dim=1).squeeze().cpu().numpy()  # Shape: (num_classes,)
        plt.barh(y_pos, probs, align='center')
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Probability')
        plt.title('Class Probabilities')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_clinical_correlation(self, attention_map, feature_map, correlation,save_path="xai/saves/corr", fig_size=(12, 4)):
        """
        Plot correlation between model attention and clinical features
        """
        plt.figure(figsize=fig_size)
        
        plt.subplot(131)
        plt.imshow(attention_map, cmap='jet')
        plt.title('Model Attention')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(feature_map, cmap='gray')
        plt.title('Clinical Features')
        plt.axis('off')
        
        plt.subplot(133)
        plt.scatter(attention_map.flatten(), feature_map.flatten(), alpha=0.1)
        plt.xlabel('Model Attention')
        plt.ylabel('Clinical Feature Intensity')
        plt.title(f'Correlation: {correlation:.2f}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def create_summary_visualization(self, results, save_path="xai/saves/summary"):
        """
        Create a comprehensive summary visualization of all XAI results

        Args:
            results: Dictionary containing all XAI results
            save_path: Path to save the visualization
        """
        # Adjust figsize and subplot layout if removing a plot
        # For example, use 2 rows, 2 cols if removing uncertainty and clinical
        num_plots = 1 # Start with original image
        if 'gradcam' in results: num_plots += 1
        if 'integrated_gradients' in results: num_plots += 1
        # SHAP is commented out in the original code
        # if 'shap_values' in results: num_plots += 1
        if 'clinical_correlation' in results: num_plots += 1

        # Determine grid size (e.g., 2x2, 2x3)
        if num_plots <= 4:
            grid_rows, grid_cols = 2, 2
            figsize = (10, 10)
        else:
            grid_rows, grid_cols = 2, 3
            figsize = (15, 10)

        plt.figure(figsize=figsize)
        plot_index = 1

        # Original image
        plt.subplot(grid_rows, grid_cols, plot_index)
        plot_index += 1
        plt.imshow((results['original_image']))
        plt.title('Original Image')
        if 'prediction' in results and 'confidence' in results:
            plt.xlabel(f'Prediction: {self.class_names[results["prediction"]]}\n'
                       f'Confidence: {results["confidence"]:.2f}')
        plt.axis('off')

        # Grad-CAM
        if 'gradcam' in results:
            plt.subplot(grid_rows, grid_cols, plot_index)
            plot_index += 1
            plt.imshow(results['original_image']) # Show original for context
            plt.imshow(results['gradcam'], cmap='jet', alpha=0.6) # Overlay
            plt.title('Grad-CAM Overlay')
            plt.axis('off')

        # Integrated Gradients
        if 'integrated_gradients' in results:
            plt.subplot(grid_rows, grid_cols, plot_index)
            plot_index += 1
            # Normalize attribution map for visualization
            attr_map = results['integrated_gradients'].sum(dim=1).squeeze().cpu().numpy()
            attr_norm = np.abs(attr_map) # Use absolute value for overlay intensity
            attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min() + 1e-8)

            plt.imshow(results['original_image']) # Show original for context
            # Use seismic for attributions, hot/cool for positive/negative emphasis
            plt.imshow(attr_map, cmap='seismic', alpha=0.6) # Overlay
            plt.title('Integrated Gradients Overlay')
            plt.axis('off')

        # SHAP Values (If uncommented later - apply similar overlay logic)
        # if 'shap_values' in results:
        #     plt.subplot(grid_rows, grid_cols, plot_index)
        #     plot_index += 1
        #     shap_img = torch.abs(results['shap_values']).sum(dim=1).squeeze().cpu().numpy()
        #     shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
        #     plt.imshow(results['original_image'])
        #     plt.imshow(shap_img, cmap='hot', alpha=0.6)
        #     plt.title('SHAP Values Overlay')
        #     plt.axis('off')

        # ----- Uncertainty Plot Removed -----
        # The 'uncertainty' value (std_pred) is per-class, not spatial.
        # The `plot_uncertainty` function already creates the correct bar chart visualization.
        # Trying to imshow it here is incorrect.

        # Clinical Correlation (If available)
        if 'clinical_correlation' in results:
            plt.subplot(grid_rows, grid_cols, plot_index)
            plot_index += 1
            # Assuming clinical_correlation is a spatial map like attention
            plt.imshow(results['original_image']) # Show original for context
            plt.imshow(results['clinical_correlation'], cmap='viridis', alpha=0.6) # Overlay
            plt.title('Clinical Correlation Overlay')
            plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


