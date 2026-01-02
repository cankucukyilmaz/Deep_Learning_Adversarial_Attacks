import torch
import torch.nn as nn
import torch.optim as optim
from .attack import IterativeTargetClassAttacker
import random

class AdversarialTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.parameters()).device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Multi-Label Loss
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, attack_epsilon=0.03, attack_steps=5):
        self.model.train()
        total_loss = 0
        
        attacker = IterativeTargetClassAttacker(self.model, alpha=0.01, num_steps=attack_steps)

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # --- DEFENSE STRATEGY ---
            # We pick a random attribute index (0-39) to attack for this batch.
            # This teaches the model to resist changes on ANY variable.
            target_idx = random.randint(0, 39)
            
            self.model.eval()
            adv_images = attacker.attack(images, target_idx, epsilon=attack_epsilon)
            self.model.train()
            
            self.optimizer.zero_grad()
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)