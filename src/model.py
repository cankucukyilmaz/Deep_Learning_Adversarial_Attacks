import torch
import torch.nn as nn
from torchvision import models

class Normalize(nn.Module):
    """
    Standard ImageNet normalization layer.
    
    Logic:
    Deep Learning models (ResNet) expect inputs with mean ~0 and std ~1.
    Images exist in [0, 1]. This layer bridges that gap inside the model,
    allowing the attacker to optimize directly on valid pixel values.
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # We register these as buffers because they are state, but not trained weights.
        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class VictimResNet(nn.Module):
    def __init__(self, num_classes, freeze_body=False):
        """
        Args:
            num_classes (int): The number of attributes/classes in your specific dataset.
            freeze_body (bool): If True, locks the pretrained weights to save memory/time.
        """
        super(VictimResNet, self).__init__()
        
        # 1. Internal Normalization (ImageNet Standards)
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 2. Load Pretrained Structure
        # 'DEFAULT' loads the best available ImageNet weights
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # 3. Logic: Freeze Body
        # If we are studying the defense of the head, we might keep the body static.
        if freeze_body:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 4. Modify the Hypothesis Space (The Head)
        # ResNet50 penultimate layer has 2048 features.
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # Expects x in range [0, 1]
        x = self.normalize(x)
        return self.model(x)