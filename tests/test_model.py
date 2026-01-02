import pytest
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import VictimResNet

@pytest.fixture(scope="module")
def model():
    # 40 output classes for CelebA
    m = VictimResNet(num_classes=40)
    m.eval()
    return m

def test_output_shape(model):
    batch = 2
    x = torch.randn(batch, 3, 224, 224)
    y = model(x)
    assert y.shape == (batch, 40), f"Expected (2, 40), got {y.shape}"

def test_normalization_layer(model):
    # Pass a raw image (all 1s). The internal layer should shift values.
    x = torch.ones(1, 3, 224, 224)
    # Hook into the model to see what enters the ResNet body
    # (Checking logical existence of normalization)
    assert hasattr(model, 'normalize'), "Normalization layer missing"