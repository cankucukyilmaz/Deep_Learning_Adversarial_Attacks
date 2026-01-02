import torch
import os
import matplotlib.pyplot as plt
from src.data import get_celeba_loaders
from src.model import VictimResNet
from src.trainer import AdversarialTrainer
from src.attack import IterativeTargetClassAttacker

# --- PATHS ---
RAW_DATA_PATH = "data/raw/img_align_celeba"
PROCESSED_CSV_PATH = "data/processed/list_attr_celeba.csv"
CHECKPOINT_DIR = "checkpoints"
PLOT_DIR = "plots"

# --- CONFIG ---
BATCH_SIZE = 16
NUM_EPOCHS = 1  # Increase this for real training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"--- Running on {DEVICE} ---")
    
    # 1. Load Data
    train_loader, val_loader, test_loader = get_celeba_loaders(
        RAW_DATA_PATH, PROCESSED_CSV_PATH, batch_size=BATCH_SIZE
    )
    
    # 2. Setup Model
    model = VictimResNet(num_classes=40).to(DEVICE)
    
    # 3. Train Defense
    trainer = AdversarialTrainer(model, train_loader, val_loader)
    
    print("\n[Phase 1] Adversarial Training...")
    for epoch in range(NUM_EPOCHS):
        loss = trainer.train_epoch(attack_epsilon=0.03, attack_steps=5)
        # val_loss = trainer.evaluate() # Optional: Uncomment if you want clean val loss
        print(f"Epoch {epoch+1} | Train Loss: {loss:.4f}")
        
    # Save Model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, "celeba_defended.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[Info] Model saved to {save_path}")

    # 4. Generate Robustness Curve (Target: Black Hair)
    print("\n[Phase 2] Testing Robustness against 'Black Hair' Attack...")
    
    # Assumption: Index 8 is Black_Hair. Check your CSV columns to confirm!
    TARGET_VAR_IDX = 8 
    
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]
    success_rates = []
    
    attacker = IterativeTargetClassAttacker(model, alpha=0.01, num_steps=10)
    model.eval()
    
    for eps in epsilons:
        success_count = 0
        total_count = 0
        print(f"Testing Epsilon {eps}...", end="\r")
        
        # We limit the validation to 100 images for speed during testing
        for i, (images, labels) in enumerate(val_loader):
            if total_count >= 100: break
            
            images = images.to(DEVICE)
            
            # Attack: Try to force Black Hair = True
            adv_images = attacker.attack(images, TARGET_VAR_IDX, epsilon=eps)
            
            with torch.no_grad():
                outputs = model(adv_images)
                # Sigmoid > 0.5 means the model predicts "True"
                preds = torch.sigmoid(outputs[:, TARGET_VAR_IDX])
                
                # Success = Model predicts True (probability > 0.5)
                success_count += (preds > 0.5).sum().item()
                total_count += images.size(0)
            
        rate = 100 * success_count / total_count
        success_rates.append(rate)
        print(f"Epsilon {eps}: Attack Success Rate {rate:.2f}%     ")

    # --- PHASE 3: VISUALIZATION ---
    print("\n[Phase 3] Saving Plot...")
    
    # Create plots folder
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, success_rates, marker='o', color='red', label='Defended ResNet50')
    plt.title("Attack Success Rate on Variable 'Black Hair'")
    plt.xlabel("Epsilon (Perturbation Magnitude)")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)
    plt.legend()
    
    # Save to the specific folder
    plot_path = os.path.join(PLOT_DIR, "celeba_robustness.png")
    plt.savefig(plot_path)
    print(f"[Done] Plot saved to {plot_path}")

if __name__ == "__main__":
    main()