import torch
import torch.nn as nn

class IterativeTargetClassAttacker:
    def __init__(self, model, alpha=0.01, num_steps=10):
        self.model = model
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = next(model.parameters()).device

    def attack(self, image, target_attribute_index, epsilon=0.1):
        """
        Attacks a specific attribute (variable) to make the model predict "True".
        
        Args:
            target_attribute_index (int): The index (0-39) to attack.
        """
        original_image = image.clone().detach().to(self.device)
        x_adv = original_image.clone().detach().requires_grad_(True)
        
        # Binary Cross Entropy with Logits
        loss_fn = nn.BCEWithLogitsLoss()
        
        self.model.eval()

        for step in range(self.num_steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            # Output shape: (B, 40)
            output = self.model(x_adv)
            
            # --- MULTI-LABEL LOGIC ---
            # Extract only the column corresponding to the target attribute
            target_logits = output[:, target_attribute_index] # Shape: (B)
            
            # We want the model to predict 1.0 (True) for this attribute
            target_labels = torch.ones_like(target_logits)
            
            # Calculate loss only on this specific attribute
            loss = loss_fn(target_logits, target_labels)
            
            loss.backward()
            
            if x_adv.grad is None:
                break
                
            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                # Gradient Descent: Minimize loss (make it look like target)
                x_adv.data = x_adv.data - (self.alpha * grad_sign)
                
                # Projection
                diff = x_adv.data - original_image
                diff = torch.clamp(diff, -epsilon, epsilon)
                x_adv.data = original_image + diff
                x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
                
            x_adv.requires_grad_(True)

        return x_adv.detach()