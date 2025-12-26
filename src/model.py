import torch
import torch.nn as nn
from torchvision import models

class VictimResNet(nn.Module):
    """
    A wrapper around ResNet50 to serve as the Victim Model.
    
    Attributes:
        num_classes (int): Dimension of the output vector.
        model (nn.Module): The underlying architecture.
    """
    def __init__(self, num_classes, freeze_body=False):
        super(VictimResNet, self).__init__()
        
        # Load Pretrained Weights (Deterministic starting point)
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # Logic: Freeze body if we want to isolate the feature extractor
        if freeze_body:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace the Head (Hypothesis Space)
        # We map R^2048 -> R^num_classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)