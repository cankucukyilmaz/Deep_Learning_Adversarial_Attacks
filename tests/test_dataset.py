import pytest
import torch
from src.dataset import CelebADataset
from torchvision import transforms

def test_label_transformation():
    # Setup paths (ensure these match your actual local structure)
    attr_path = "data/external/list_attr_celeba.txt"
    root_dir = "data/raw/"
    
    # Define a simple transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Initialize dataset
    dataset = CelebADataset(root_dir=root_dir, attr_file=attr_path, transform=transform)
    
    # Get the first sample
    image, labels = dataset[0]
    
    # 1. Check for binary values 0 or 1
    # We use torch.unique to see all distinct values in the label tensor
    unique_labels = torch.unique(labels)
    for val in unique_labels:
        assert val in [0.0, 1.0], f"Expected label values {0, 1}, but found {val}"
    
    # 2. Check shapes
    assert image.shape == (3, 224, 224), f"Incorrect image shape: {image.shape}"
    assert isinstance(labels, torch.Tensor), "Labels should be a torch.Tensor"

def test_dataset_length():
    attr_path = "data/external/list_attr_celeba.txt"
    root_dir = "data/raw/"
    dataset = CelebADataset(root_dir=root_dir, attr_file=attr_path)
    
    # CelebA should have 202,599 images
    assert len(dataset) > 0, "Dataset should not be empty"