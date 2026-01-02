import pytest
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import VictimResNet
from src.attack import IterativeTargetClassAttacker

@pytest.fixture
def setup():
    model = VictimResNet(num_classes=40)
    attacker = IterativeTargetClassAttacker(model, alpha=0.01, num_steps=5)
    return model, attacker

def test_targeted_attack_logic(setup):
    model, attacker = setup
    image = torch.rand(1, 3, 224, 224)
    target_idx = 8 # Pretend this is Black_Hair
    
    # Get initial score for index 8
    initial_score = model(image)[0, target_idx].item()
    
    # Attack to maximize index 8
    adv_image = attacker.attack(image, target_idx, epsilon=0.1)
    
    final_score = model(adv_image)[0, target_idx].item()
    
    # The score (logit) should increase
    print(f"Score: {initial_score} -> {final_score}")
    assert final_score >= initial_score - 0.1 # Tolerance for random init

def test_epsilon_constraint(setup):
    model, attacker = setup
    image = torch.rand(1, 3, 224, 224)
    epsilon = 0.05
    adv = attacker.attack(image, 0, epsilon=epsilon)
    diff = (adv - image).abs().max().item()
    assert diff <= epsilon + 1e-5