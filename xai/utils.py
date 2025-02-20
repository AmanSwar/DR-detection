import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from captum.attr import IntegratedGradients, DeepLift
import shap
import numpy as np

class XAIanal:

    def __init__(self , model , target_layer):

        self.model = model
        self.target_layer = target_layer
        self.grad_cam = GradCAM(model=model , target_layers=target_layer)
        self.int_grad = IntegratedGradients(model)


    def generate_gradcam(self , inp_tensor):

        cam_mask = self.grad_cam(inp_tensor)
        return cam_mask
    
    def gen_integrated_gradients(self , inp_tensor , target_class):

        attributions = self.int_grad.attribute(
            inp_tensor,
            target=target_class,
            n_steps=50

        )

        return attributions
    
    def gen_shap_values(self, inp_tensor , bg_tensor):

        explainer = shap.DeepExplainer(self.model , bg_tensor)
        shap_values = explainer.shap_values(inp_tensor)

        return shap_values
    
    def uncertainty_quant(self , inp_tens , num_samples = 30):
        self.model.train() #for enabling dropouts

        pred_arr = []

        for _ in range(num_samples):
            with torch.no_grad():

                pred = self.model(inp_tens)
                pred_arr.append(pred)

        pred_stack = torch.stack(pred_arr)

        mean_pred = torch.mean(pred_stack , dim=0)
        std_pred = torch.std(pred , dim=0)

        return mean_pred , std_pred
    

    def gen_counterfactual(self , inp_tensor , target_class , epsilon=0.1 , num_steps=100):
        counterfactual = inp_tensor.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=0.01)
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            output = self.model(counterfactual)
            loss = nn.CrossEntropyLoss()(output, target_class)
            loss += epsilon * torch.norm(counterfactual - inp_tensor)
            loss.backward()
            optimizer.step()
            
        return counterfactual


    def clinical_feature_correlation(self, input_tensor, feature_map):
        """Correlate model attention with clinical features"""
        attention_map = self.generate_gradcam(input_tensor)
        correlation = np.corrcoef(
            attention_map.flatten(),
            feature_map.flatten()
        )[0,1]
        return correlation
    
def init_xai_anal(model):

    target_layer = model.features[-1]  
    analyzer = XAIanal(model, target_layer)
    return analyzer

def analyze_single_image(analyzer, image_tensor, target_class):
    # Generate Grad-CAM
    gradcam_heatmap = analyzer.generate_gradcam(image_tensor)
    
    # Generate Integrated Gradients
    ig_attributions = analyzer.generate_integrated_gradients(image_tensor, target_class)
    
    # Get uncertainty estimates
    mean_pred, uncertainty = analyzer.uncertainty_quantification(image_tensor)
    
    # Generate counterfactual
    counterfactual = analyzer.generate_counterfactual(image_tensor, target_class)
    
    return {
        'gradcam': gradcam_heatmap,
        'integrated_gradients': ig_attributions,
        'uncertainty': uncertainty,
        'counterfactual': counterfactual
    }

def batch_analysis(analyzer, image_batch, target_classes):
    results = []
    for img, target in zip(image_batch, target_classes):
        img_results = analyze_single_image(analyzer, img.unsqueeze(0), target)
        results.append(img_results)
    return results

def realtime_analysis(analyzer, image_tensor, confidence_threshold=0.8):
    """
    Generate explanations based on model confidence
    """
    mean_pred, uncertainty = analyzer.uncertainty_quantification(image_tensor)
    confidence = torch.max(torch.softmax(mean_pred, dim=1))
    
    # Basic explanation for all cases
    explanations = {
        'gradcam': analyzer.generate_gradcam(image_tensor)
    }
    
    # If confidence is low, generate more detailed explanations
    if confidence < confidence_threshold:
        target_class = torch.argmax(mean_pred)
        explanations.update({
            'integrated_gradients': analyzer.generate_integrated_gradients(
                image_tensor, target_class
            ),
            'uncertainty': uncertainty,
            'counterfactual': analyzer.generate_counterfactual(
                image_tensor, target_class
            )
        })
    
    return explanations

