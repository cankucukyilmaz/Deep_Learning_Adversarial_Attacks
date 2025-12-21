import torch
import torch.nn as nn
from torchvision import models

def get_resnet_baseline(num_classes=1, pretrained=True):
    # 1. Load the architecture with ImageNet weights
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    
    # 2. Modify the final layer (Fully Connected)
    # ResNet18's last layer is 'fc', originally for 1000 ImageNet classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes) 
    
    return model