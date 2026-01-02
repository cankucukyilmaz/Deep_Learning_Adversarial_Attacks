import torch
import torch.nn as nn
from torchvision import models

class Normalize(nn.Module):
    """
    Internal Normalization Layer.
    Allows the attacker to work in [0, 1] space while ResNet gets [-2, 2].
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class VictimResNet(nn.Module):
    def __init__(self, num_classes=40, freeze_body=False):
        super(VictimResNet, self).__init__()
        
        # 1. Normalize Layer (ImageNet stats)
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 2. Pretrained ResNet
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        if freeze_body:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 3. Modify Head for Multi-Label (Output 40 logits)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.normalize(x)
        return self.model(x)