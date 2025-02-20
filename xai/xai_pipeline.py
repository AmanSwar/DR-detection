from inference.model import ConvexNet
import  xai.utils as xai
from data_pipeline.data_set import UniformTestLoader

        
model = ConvexNet()
analyzer = xai.init_xai_anal(model)


test_loader = UniformTestLoader().get_loader()
all_results = []

for images, targets in test_loader:
    batch_results = xai.batch_analysis(analyzer, images, targets)
    all_results.extend(batch_results)