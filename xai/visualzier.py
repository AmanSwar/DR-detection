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
        image = self.preprocess_image(original_image)

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
        image = self.preprocess_image(original_image)
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
        """
        Plot SHAP value visualization
        """
        image = self.preprocess_image(original_image)
        
        plt.figure(figsize=fig_size)
        
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        shap_img = np.abs(shap_values).mean(axis=-1)
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
        Plot uncertainty visualization
        """
        image = self.preprocess_image(original_image)
        
        plt.figure(figsize=fig_size)
        
        # Original image with prediction
        plt.subplot(131)
        plt.imshow(image)
        pred_class = torch.argmax(mean_pred).item()
        plt.title(f'Prediction: {self.class_names[pred_class]}')
        plt.axis('off')
        
        # Uncertainty bar plot
        plt.subplot(132)
        probs = torch.softmax(mean_pred, dim=1).squeeze().cpu().numpy()
        uncertainties = std_pred.squeeze().cpu().numpy()
        
        y_pos = np.arange(len(self.class_names))
        plt.barh(y_pos, probs, xerr=uncertainties, align='center')
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Probability')
        plt.title('Prediction Uncertainty')
        
        # Uncertainty heatmap
        plt.subplot(133)
        uncertainty_map = std_pred.mean(dim=1).squeeze().cpu().numpy()
        uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min())
        plt.imshow(image)
        plt.imshow(uncertainty_map, cmap='Reds', alpha=0.5)
        plt.title('Uncertainty Map')
        plt.axis('off')
        
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
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(231)
        plt.imshow(self.preprocess_image(results['original_image']))
        plt.title('Original Image')
        if 'prediction' in results and 'confidence' in results:
            plt.xlabel(f'Prediction: {self.class_names[results["prediction"]]}\n'
                      f'Confidence: {results["confidence"]:.2f}')
        plt.axis('off')
        
        # Grad-CAM
        if 'gradcam' in results:
            plt.subplot(232)
            plt.imshow(results['gradcam'], cmap='jet')
            plt.title('Grad-CAM')
            plt.axis('off')
        
        # Integrated Gradients
        if 'integrated_gradients' in results:
            plt.subplot(233)
            attr_map = results['integrated_gradients'].sum(dim=1).squeeze().cpu().numpy()
            plt.imshow(attr_map, cmap='seismic')
            plt.title('Integrated Gradients')
            plt.axis('off')
        
        # SHAP Values
        if 'shap_values' in results:
            plt.subplot(234)
            shap_img = np.abs(results['shap_values']).mean(axis=-1)
            plt.imshow(shap_img, cmap='hot')
            plt.title('SHAP Values')
            plt.axis('off')
        
        # Uncertainty
        if 'uncertainty' in results:
            plt.subplot(235)
            uncertainty_map = results['uncertainty'].mean(dim=1).squeeze().cpu().numpy()
            plt.imshow(uncertainty_map, cmap='Reds')
            plt.title('Uncertainty Map')
            plt.axis('off')
        
        # Clinical Correlation
        if 'clinical_correlation' in results:
            plt.subplot(236)
            plt.imshow(results['clinical_correlation'], cmap='viridis')
            plt.title('Clinical Correlation')
            plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


