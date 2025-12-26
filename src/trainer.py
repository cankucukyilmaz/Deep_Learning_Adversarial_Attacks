import torch
import torch.nn as nn
import torch.optim as optim
from .attack import IterativeTargetClassAttacker

class AdversarialTrainer:
    """
    The Defender. 
    This class manages the training loop, injecting adversarial examples 
    into the batch to robustify the model.
    """
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.parameters()).device
        
        # Optimizer (Stochastic Gradient Descent or Adam)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, attack_epsilon=0.1, alpha=0.01, attack_steps=10):
        """
        Runs one epoch of Adversarial Training.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Initialize the attacker
        attacker = IterativeTargetClassAttacker(
            self.model, 
            alpha=alpha, 
            num_steps=attack_steps
        )

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # --- THE DEFENSE LOGIC ---
            # 1. Generate Attack Targets
            # For targeted training, we usually pick a random WRONG class to defend against,
            # or we use Untargeted attacks (maximize loss). 
            # If your project strictly uses Targeted BIM, we pick a random target index != label.
            
            # Simple logic: Target is (label + 1) % num_classes
            # This ensures we are always defending against "shifting" attacks
            target_labels = (labels + 1) % self.model.model.fc.out_features
            
            # 2. Generate Adversarial Examples
            # We treat the model as fixed/frozen while generating the attack
            self.model.eval() 
            adv_images = attacker.attack(images, target_labels, epsilon=attack_epsilon)
            self.model.train() # Switch back to training mode
            
            # 3. Train on Adversarial Images (Robustness)
            # You can also mix clean + adv images here (50/50 split is common)
            
            self.optimizer.zero_grad()
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels) # We want model to predict TRUE label for ADV image
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        accuracy = 100. * correct / total
        return total_loss / len(self.train_loader), accuracy

    def evaluate(self):
        """Standard validation loop on CLEAN data."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total