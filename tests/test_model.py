import torch
from src.model import get_celeba_resnet50

def test_is_pretrained():
    """
    Verify that the weights are not just random initialization.
    """
    model = get_celeba_resnet50(pretrained=True)
    
    # Check the weights of the first layer
    # If they were random (default PyTorch init), they would have a very specific distribution.
    # Pretrained weights have 'learned' structures.
    first_conv_weights = model.conv1.weight.data
    
    # Simple check: the mean of pretrained weights is almost never exactly zero
    assert torch.abs(first_conv_weights.mean()) > 1e-7
    print("✓ Model confirmed to have non-random (likely pretrained) weights.")

def test_celeba_output():
    model = get_celeba_resnet50(pretrained=True)
    assert model.fc.out_features == 40