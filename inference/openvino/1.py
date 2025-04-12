from openvino.tools import mo
import nncf
from nncf import TargetDevice, QuantizationPreset  # Updated imports
from openvino.runtime import Core, save_model
from data_pipeline.data_set import UniformValidDataloader
from torchvision import transforms
from nncf.quantization.advanced_parameters import (
    AdvancedQuantizationParameters,
    QuantizationParameters,
    OverflowFix
)

model_path = "model.onnx"

# Convert to OpenVINO IR (FP16)
ir_model = mo.convert_model(model_path, compress_to_fp16=True)
save_model(ir_model, "fp16_model.xml")

# Initialize OpenVINO Core
core = Core()
model = core.read_model("fp16_model.xml")
print("Model layers:")
for op in model.get_ops():
    print(f"- {op.friendly_name}")
# Define dataset transformation
def transform_fn(data_item):
    images, _ = data_item
    return images.numpy()

# Prepare calibration dataset
ds_name = ["eyepacs", "aptos", "ddr", "idrid", "messdr"]
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_loader = UniformValidDataloader(
    dataset_names=ds_name,
    transformation=val_transform,
    batch_size=1,
    num_workers=3
).get_loader()

calibration_dataset = nncf.Dataset(data_loader, transform_fn)

# # INT8 Quantization
# int8_model = nncf.quantize(
#     model,
#     calibration_dataset,
#     preset=QuantizationPreset.PERFORMANCE,
#     target_device=TargetDevice.CPU,
#     subset_size=300
# )
# save_model(int8_model, "int8_model.xml")
QuantizationPreset.
# INT4 Quantization (Fixed)
print("Starting INT4 quantization")
int4_model = nncf.quantize(
    model,
    calibration_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    advanced_parameters=AdvancedQuantizationParameters(
        weights_quantization_params=QuantizationParameters(num_bits=4),
        activations_quantization_params=QuantizationParameters(num_bits=8),
        smooth_quant_alphas=None
    ),
    ignored_scope=nncf.IgnoredScope(
        # Example based on actual layer names:
        names=["/model/stem.*", "/model/classifier.*"],
        # types=[]  # Remove if no ConvTranspose layers exist
    )
) 

save_model(int4_model, "int4_model.xml")