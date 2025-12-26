import pytest
import torch
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import VictimResNet
from src.attack import IterativeTargetClassAttacker

# --- FIXTURES ---

@pytest.fixture(scope="module")
def setup_system():
    """
    Sets up the Model and a default Attacker once for the whole test file.
    """
    n_classes = 10
    # We use a dummy model structure. 
    # Note: We don't need real ImageNet weights to test the *logic* of the attack math.
    model = VictimResNet(num_classes=n_classes)
    model.eval()
    
    # Default parameters for testing
    attacker = IterativeTargetClassAttacker(model, alpha=0.02, num_steps=10)
    
    return model, attacker, n_classes

# --- TESTS ---

def test_epsilon_constraint_dynamic(setup_system):
    """
    Hypothesis: The attack must respect the epsilon passed at RUNTIME.
    """
    model, attacker, _ = setup_system
    
    # Create a random image [0, 1]
    image = torch.rand(1, 3, 224, 224)
    target = torch.tensor([1])
    
    # Test 1: Small Epsilon
    eps_small = 0.05
    adv_small = attacker.attack(image, target, epsilon=eps_small)
    diff_small = (adv_small - image).abs().max().item()
    
    # Test 2: Large Epsilon
    eps_large = 0.2
    adv_large = attacker.attack(image, target, epsilon=eps_large)
    diff_large = (adv_large - image).abs().max().item()
    
    # Assertions (with small floating point tolerance)
    assert diff_small <= eps_small + 1e-5, f"Violated small epsilon: {diff_small} > {eps_small}"
    assert diff_large <= eps_large + 1e-5, f"Violated large epsilon: {diff_large} > {eps_large}"
    
    # Logic check: Larger epsilon should allow (but not guarantee) larger perturbations
    # Ideally, if the model is robust-ish, the attack will use the space it's given.
    print(f"\n[Check] Small Perturbation: {diff_small:.4f}, Large Perturbation: {diff_large:.4f}")

def test_pixel_physical_validity(setup_system):
    """
    Hypothesis: Adversarial images must remain valid images [0, 1].
    """
    model, attacker, _ = setup_system
    
    # Start with a very bright image (pixels near 1.0)
    # If we add positive noise, it might overshoot 1.0 without clamping
    image = torch.ones(1, 3, 224, 224) * 0.95
    target = torch.tensor([0])
    
    # Use a large step size to force overshoot potential
    attacker.alpha = 0.1 
    
    adv_image = attacker.attack(image, target, epsilon=0.2)
    
    assert adv_image.max().item() <= 1.0, "Upper bound violation: Pixels > 1.0"
    assert adv_image.min().item() >= 0.0, "Lower bound violation: Pixels < 0.0"

def test_attack_directionality(setup_system):
    """
    Hypothesis: The attack should INCREASE the probability of the target class.
    
    Note: Since we use a mostly random model here, we can't guarantee a label flip.
    But mathematically, Gradient Descent MUST increase the target logit 
    (or decrease target loss) if the step size is small enough.
    """
    model, attacker, n_classes = setup_system
    
    # Reset alpha to something reasonable for this test
    attacker.alpha = 0.01
    
    image = torch.rand(1, 3, 224, 224)
    
    # Pick a target that is NOT the current predicted class (if possible)
    initial_out = model(image)
    current_pred = initial_out.argmax(dim=1)
    target = (current_pred + 1) % n_classes
    
    # Measure initial probability of the TARGET
    initial_prob = torch.softmax(initial_out, dim=1)[0, target].item()
    
    # Run Attack
    adv_image = attacker.attack(image, target, epsilon=0.1)
    
    # Measure final probability
    final_out = model(adv_image)
    final_prob = torch.softmax(final_out, dim=1)[0, target].item()
    
    print(f"\n[Logic] Target Prob: {initial_prob:.4f} -> {final_prob:.4f}")
    
    # Verification: Did we move in the right direction?
    # We use >= because sometimes we start at a local optimum or flat gradient.
    assert final_prob >= initial_prob - 1e-6, \
        "Logic Error: Attack moved AWAY from the target class."

def test_batch_processing(setup_system):
    """
    Hypothesis: The attack should handle a batch of images (B > 1) correctly.
    """
    model, attacker, _ = setup_system
    
    batch_size = 4
    images = torch.rand(batch_size, 3, 224, 224)
    # Create different targets for each image in the batch
    targets = torch.tensor([0, 1, 2, 3])
    
    adv_images = attacker.attack(images, targets, epsilon=0.1)
    
    assert adv_images.shape == (batch_size, 3, 224, 224), \
        f"Batch Shape Error: Expected {(batch_size, 3, 224, 224)}, got {adv_images.shape}"