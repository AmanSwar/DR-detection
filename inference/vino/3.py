from openvino.runtime import Core
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score
import numpy as np
import torch.nn as nn
import torch



class OpenVINOInferenceModel(nn.Module):
    def __init__(self, ir_path, device="CPU", perf_hint=None):
        super().__init__()
        self.ie = Core()
        self.net = self.ie.read_network(model=ir_path + ".xml", weights=ir_path + ".bin")
        config = {"PERFORMANCE_HINT": perf_hint} if perf_hint else {}
        self.exec_net = self.ie.load_network(network=self.net, device_name=device, config=config)
        self.input_blob = next(iter(self.net.input_info))
        self.output_blob = next(iter(self.net.outputs))
    
    def forward(self, x):
        input_data = x.cpu().numpy()
        res = self.exec_net.infer(inputs={self.input_blob: input_data})
        output = res[self.output_blob]
        return torch.from_numpy(output)
    


def get_calibration_data(dataloader, num_samples=100):
    calibration_data = []
    for i, batch_data in enumerate(dataloader):
        if len(batch_data) == 3:
            images, _, _ = batch_data
        else:
            images, _ = batch_data
        calibration_data.extend(images.cpu().numpy())
        if len(calibration_data) >= num_samples:
            break
    return calibration_data[:num_samples]

calibration_data = get_calibration_data(dataloader)


def validate(model, dataloader, device, epoch, num_epochs, wandb_run=None,
             lambda_consistency=0.1, ordinal_weight=0.3, num_classes=5):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    inference_time = 0.0  # Track inference time only
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for _, batch_data in tqdm(enumerate(dataloader)):
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Measure inference time
            start_time = time.time()
            if isinstance(model, OpenVINOInferenceModel):
                logits = model(images)
            else:
                logits, _, _ = model(images, alpha=0.0, update_prototypes=False)
            end_time = time.time()
            inference_time += end_time - start_time
            
            loss = loss_fn(logits, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    
    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = []
    specificity = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)
    
    try:
        auc_macro_ovr = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except:
        auc_macro_ovr = 0.0
    
    # Report results
    avg_inference_time = inference_time / len(dataloader) * 1000  # Convert to milliseconds
    print(f"Validation - Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, "
          f"F1(W): {f1_weighted:.4f}, QWK: {qwk:.4f}")
    print(f"Avg Sensitivity: {avg_sensitivity:.4f}, Avg Specificity: {avg_specificity:.4f}, "
          f"AUC(Macro-OvR): {auc_macro_ovr:.4f}")
    print(f"Average Inference Time per Batch: {avg_inference_time:.2f} ms")
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "qwk": qwk,
        "sensitivity": avg_sensitivity,
        "specificity": avg_specificity,
        "auc": auc_macro_ovr,
        "inference_time": avg_inference_time
    }
