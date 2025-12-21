import torch
import torch.nn as nn
import torch.optim as optim
from src.model import get_celeba_resnet50
from src.dataset import get_celeba_loaders

# --- Configuration ---
ROOT_DIR = "./data" # Path to your data folder
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Load Data
    train_loader, test_loader = get_celeba_loaders(ROOT_DIR, BATCH_SIZE)

    # 2. Load Pretrained ResNet-50
    model = get_celeba_resnet50(pretrained=True).to(DEVICE)

    # 3. Loss and Optimizer
    # We use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Save checkpoint after each epoch
        torch.save(model.state_dict(), f"models/resnet50_celeba_epoch{epoch+1}.pth")

if __name__ == "__main__":
    train()