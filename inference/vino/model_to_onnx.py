import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score
import numpy as np
import torch.nn as nn
import torch
import torch.nn as nn
from inference.model import EnhancedDRClassifier



class InferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        logits, _, _ = self.model(x, alpha=0.0)
        return logits

device = torch.device("cpu")
model = EnhancedDRClassifier(num_classes=5, freeze_backbone=False).to(device)
checkpoint = torch.load("good_chkpt/fine_3_local/best_best_clinical_model.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

inference_model = InferenceModel(model)

dummy_input = torch.randn(1, 3, 256, 256)

# Export to ONNX
torch.onnx.export(
    inference_model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=11  # Compatible with OpenVINO
)


