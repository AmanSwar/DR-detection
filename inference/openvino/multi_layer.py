import onnx
from onnx2torch import convert
from torch.fx import symbolic_trace
import torch
# Path to your ONNX model
onnx_model_path = "model.onnx"

# Option 1: Directly convert from the file path
model = convert(onnx_model_path)
traced_model = torch.jit.script(model)

print(traced_model.print_readable())