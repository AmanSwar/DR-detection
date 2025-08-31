# A Research Repository for Diabetic Retinopathy Detection

This repository contains a collection of PyTorch implementations for various deep learning models and self-supervised learning methods for Diabetic Retinopathy (DR) detection. This repository was used for research and experimentation, and it contains a wide range of scripts for training, fine-tuning, and evaluating different models.

Diabetic Retinopathy (DR) is a leading cause of preventable blindness worldwide. This repository explores various deep learning techniques to build a robust and generalizable DR detection system. The focus is on leveraging self-supervised learning to learn meaningful representations from retinal fundus images, and then fine-tuning these models for the downstream task of DR classification.

## 🌟 Key Features

- **Multiple Self-Supervised Learning (SSL) Methods**: Implementations of various SSL methods, including:
    - Momentum Contrast (MoCo)
    - DINO (self-distillation with no labels)
    - iBOT (Image-based Joint-embedding Transformer)
    - iJEPA (Image-based Joint-Embedding Predictive Architecture)
    - SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)
- **Various Deep Learning Models**: Implementations of different backbone architectures, such as:
    - ConvNeXt
    - Vision Transformer (ViT)
    - Swin Transformer
- **Multi-Task Fine-tuning**: The repository includes scripts for fine-tuning the pre-trained models on the DR classification task.
- **Explainable AI (XAI)**: The repository includes a suite of XAI techniques to provide insights into the model's decision-making process, including:
    - Attention Maps
    - Integrated Gradients
    - SHAP
    - Uncertainty Estimation (Monte Carlo Dropout)
- **Optimized for Deployment**: The repository includes scripts for optimizing the models for real-time CPU deployment using OpenVINO.

## 🏗️ System Architecture

The general pipeline used in this repository follows a two-stage process:

1.  **Self-Supervised Pre-training**: A backbone model (e.g., ConvNeXt, ViT) is pre-trained on a large dataset of unlabeled retinal fundus images using one of the implemented SSL methods. This allows the model to learn robust and generalizable visual features.
2.  **Supervised Fine-tuning**: The pre-trained model is then fine-tuned on a smaller dataset of labeled images for the task of DR classification.

This modular design allows for independent improvements to each component.

![RetinaSys Architecture](assets/training.drawio.png)

## 👁️ Diabetic Retinopathy Grades

The system is trained to classify retinal fundus images into five distinct grades of severity, following established clinical standards, from a healthy retina (No DR) to the most advanced stage (Proliferative DR).

![DR Grades](assets/DRgrades.png)
![Attention Mapes](assets/attention_map.png)

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+
- PyTorch
- OpenCV
- Albumentations
- Einops
- And other packages listed in `requirements.txt`.

### 2. Installation

First, clone the repository to your local machine:

```
git clone <repository-url>
cd <repository-name>
```

Next, it is highly recommended to create a dedicated virtual environment to manage dependencies and avoid conflicts:

```
python -m venv .venv
source .venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Dataset Preparation

This project's robustness comes from its training on a diverse collection of public datasets (EyePACS, DDR, APTOS, IDRID, MESSIDOR-2, etc.). Please download these datasets from their original sources. For compatibility with our data loaders, structure them as follows:

```
data/
├── aptos/
├── ddr/
├── eyepacs/
├── idrid/
└── mesdr/
```

## 🔬 Experiments

The `experimental_script` directory contains numerous scripts for running various experiments. Here are some examples:

-   `train_ijepa.py`: Train a model using the iJEPA self-supervised learning method.
-   `train_drijepa.py`: A custom version of iJEPA for DR detection.
-   `train_iBOT.py`: Train a model using the iBOT self-supervised learning method.
-   `train_dino.py`: Train a model using the DINO self-supervised learning method.
-   `finetune/new_finetune.py`: A script for fine-tuning a pre-trained model.

Please refer to the scripts in the `experimental_script` directory for more details on how to run the experiments.

## 📊 Results

The models trained using the scripts in this repository can achieve strong performance across multiple evaluation criteria, including high specificity and Quadratic Weighted Kappa (QWK).

#### Training & Validation Curves
![MoCo Loss](assets/moco_train_loss.png)
![Finetuning Curves](assets/train_epoch.png)

## 🧠 Explainable AI (XAI) Analysis

A correct prediction is useful, but an explainable one is trustworthy. To build clinical trust and facilitate model debugging, this repository provides clear visual explanations for its predictions. These visualizations highlight the specific pathological features (e.g., microaneurysms, hard exudates) that drive the model's diagnostic reasoning, creating a powerful feedback loop for clinicians.

![XAI Analysis](assets/all_combined.png)