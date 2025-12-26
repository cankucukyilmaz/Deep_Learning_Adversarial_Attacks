import pytest
import torch
import sys
import os
from torchvision import models

# Ensure src is in the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import VictimResNet

# --- FIXTURES ---
# Logic: Loading a neural network is computationally expensive.
# We do it once per module to keep the feedback loop tight.
@pytest.fixture(scope="module")
def n_classes():
    return 10

@pytest.fixture(scope="module")
def victim_model(n_classes):
    """Initializes the model once for all tests in this file."""
    model = VictimResNet(num_classes=n_classes, freeze_body=False)
    model.eval() # Default to eval mode
    return model

# --- TESTS ---

def test_dimensionality_alignment(victim_model, n_classes):
    """
    Hypothesis: The model maps R^(B,3,224,224) -> R^(B, N).
    """
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = victim_model(dummy_input)
    
    expected_shape = (batch_size, n_classes)
    
    assert output.shape == expected_shape, \
        f"Dimension Logic Error: Expected {expected_shape}, got {output.shape}"

def test_pretrained_weights_are_loaded(victim_model):
    """
    Hypothesis: The model is not initialized with pure random noise.
    Verification: Compare weights against a fresh, untrained ResNet.
    """
    # Load a raw, untrained ResNet for comparison
    untrained_model = models.resnet50(weights=None)
    
    # Extract weights from the first convolutional layer
    # Note: victim_model.model accesses the underlying ResNet inside our wrapper
    trained_w = victim_model.model.conv1.weight.data
    random_w = untrained_model.conv1.weight.data
    
    # Logic: If weights are identical, our loading mechanism failed.
    # We use L1 distance to measure difference.
    difference = torch.sum(torch.abs(trained_w - random_w))
    
    assert difference.item() > 0.0, \
        "Initialization Error: Model weights match random initialization."

def test_gradient_propagation(victim_model):
    """
    Hypothesis: The computational graph supports backpropagation (learning).
    """
    # We must switch to train mode to ensure gradients are tracked if needed, 
    # though technically layers like Dropout/BatchNorm change behavior.
    victim_model.train()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    output = victim_model(dummy_input)
    
    # Create a scalar loss to backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check the first layer's gradients
    grad = victim_model.model.conv1.weight.grad
    
    assert grad is not None, "Gradient Error: Gradients are None (Graph broken)."
    assert torch.norm(grad).item() > 0.0, "Gradient Error: Gradients are zero (No learning signal)."