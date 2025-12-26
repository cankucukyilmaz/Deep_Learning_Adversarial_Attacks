import torch
import torch.nn as nn

class IterativeTargetClassAttacker:
    """
    Implements the Basic Iterative Method (BIM) for Targeted Attacks.
    """
    def __init__(self, model, alpha=0.01, num_steps=10):
        """
        Args:
            model: The VictimResNet instance.
            alpha: The step size for each iteration.
            num_steps: How many times we iterate.
        """
        self.model = model
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = next(model.parameters()).device

    def attack(self, image, target_class_index, epsilon=0.1):
        """
        Args:
            image: Tensor (B, 3, H, W) in range [0, 1].
            target_class_index: Int or Tensor containing the label we want to force.
            epsilon (float): The radius of the L_inf ball. 
                             Passed here so you can vary it for plotting curves.
            
        Returns:
            Adversarial Image Tensor (B, 3, H, W) in range [0, 1].
        """
        original_image = image.clone().detach().to(self.device)
        x_adv = original_image.clone().detach().requires_grad_(True)
        
        # Ensure target is a tensor
        if not isinstance(target_class_index, torch.Tensor):
            target = torch.tensor([target_class_index], device=self.device)
        else:
            target = target_class_index.to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        self.model.eval()

        for step in range(self.num_steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            # Forward Pass
            output = self.model(x_adv)
            loss = loss_fn(output, target)
            loss.backward()
            
            if x_adv.grad is None:
                break
                
            # Update Step
            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                # Move towards target (Gradient Descent)
                x_adv.data = x_adv.data - (self.alpha * grad_sign)
                
                # Constraint 1: Epsilon Ball (Dynamic Epsilon)
                diff = x_adv.data - original_image
                
                # Logic: We clip the noise to be within [-epsilon, +epsilon]
                diff = torch.clamp(diff, -epsilon, epsilon)
                x_adv.data = original_image + diff
                
                # Constraint 2: Pixel Range [0, 1]
                x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
                
            x_adv.requires_grad_(True)

        return x_adv.detach()