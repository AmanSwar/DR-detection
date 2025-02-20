from inference.model import ConvexNet
import  xai.utils as xai
from data_pipeline.data_set import UniformTestLoader
from xai.visualizer import XaiVisual

import torch
        
model = ConvexNet()
analyzer = xai.init_xai_anal(model)
visualizer = XaiVisual(class_names=['No DR' , 'Mild' , 'Moderate' , "Severe"  ,"Proliferative DR"])
def analyze_and_visualize(image_tensor, save_dir='results/'):
    target_class = 1 # class I am interested in
    results = xai.analyze_single_image(analyzer, image_tensor, target_class)
    
    # Add original image and prediction info
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output).item()
        conf = torch.softmax(output, dim=1).max().item()
    
    results.update({
        'original_image': image_tensor,
        'prediction': pred,
        'confidence': conf
    })
    
    # Create individual visualizations
    visualizer.plot_gradcam(
        image_tensor, 
        results['gradcam'],
        prediction=pred,
        confidence=conf,
        save_path=f'{save_dir}/gradcam.png'
    )
    
    visualizer.plot_integrated_gradients(
        image_tensor,
        results['integrated_gradients'],
        save_path=f'{save_dir}/integrated_gradients.png'
    )
    
    # Create summary visualization
    visualizer.create_summary_visualization(
        results,
        save_path=f'{save_dir}/summary.png'
    )
    
    return results

test_loader = UniformTestLoader().get_loader()
all_results = []

for images, targets in test_loader:
    batch_results = xai.batch_analysis(analyzer, images, targets)
    all_results.extend(batch_results)