import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from model.DRijepa import DRIjepa ,Patchify, TransformerEncoder, IJEPALoss, create_DRijepa, Trainer

@pytest.fixture
def dummy_data():
    """Creates a batch of dummy images for testing."""
    batch_size = 2
    img_size = 2048
    return torch.randn(batch_size, 3, img_size, img_size)

@pytest.fixture
def model():
    """Creates a small version of the DRIjepa model for testing."""
    return create_DRijepa(img_size=128, patch_size=16, embed_dim=64, encoder_depth=2, pred_depth=1, n_heads=2)

def test_patchify(dummy_data):
    """Test patch embedding output shape."""
    patchify = Patchify(img_size=2048, patch_size=32, in_chan=3, embed_dim=1024)
    patches = patchify(dummy_data)
    assert patches.shape[0] == dummy_data.shape[0]  # Batch size
    assert patches.shape[1] > 0  # Ensure non-zero patches

def test_transformer_encoder():
    """Test transformer encoder output shape."""
    encoder = TransformerEncoder(dim=64, depth=2, heads=2, mlp_dim=128)
    x = torch.randn(2, 100, 64)  # Batch=2, Tokens=100, Dim=64
    output = encoder(x)
    assert output.shape == x.shape  # Output should match input shape

def test_drijepa_forward(dummy_data, model):
    """Test forward pass of DRIjepa."""
    model, _ = model
    pred_feat, target_feat = model(dummy_data)
    assert pred_feat.shape == target_feat.shape
    assert pred_feat.shape[-1] == model.embed_dim  # Feature dimension must match embedding dim

def test_loss_function():
    """Test the loss function output."""
    loss_fn = IJEPALoss()
    pred_feat = torch.randn(2, 6, 1024)
    target_feat = torch.randn(2, 6, 1024)
    loss = loss_fn(pred_feat, target_feat)
    assert loss.item() > 0  # Ensure loss is computed

def test_trainer(dummy_data, model):
    """Test a single training step."""
    model, loss_fn = model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(dummy_data)
    train_loader = DataLoader(dataset, batch_size=2)
    trainer = Trainer(model, loss_fn, train_loader, optimizer, max_ep=1)
    trainer.train_epoch(1)

if __name__ == "__main__":
    pytest.main()
