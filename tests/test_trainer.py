import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import VictimResNet
from src.trainer import AdversarialTrainer

def test_training_loop():
    # Dummy data: 10 images, 40 attributes each
    x = torch.rand(10, 3, 224, 224)
    y = torch.randint(0, 2, (10, 40)).float() # Binary labels
    
    loader = DataLoader(TensorDataset(x, y), batch_size=5)
    model = VictimResNet(num_classes=40)
    trainer = AdversarialTrainer(model, loader, loader)
    
    # Run 1 epoch
    loss = trainer.train_epoch(attack_epsilon=0.01, attack_steps=1)
    assert loss > 0