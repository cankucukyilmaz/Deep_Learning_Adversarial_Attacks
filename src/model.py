import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torchvision import models

def get_celeba_resnet50(pretrained=True):
    # Load ResNet-50 
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # ResNet-50 has 2048 input features at the final (fc) layer
    num_ftrs = model.fc.in_features
    
    # Change the output to 40 for CelebA attributes
    model.fc = nn.Linear(num_ftrs, 40)
    
    return model