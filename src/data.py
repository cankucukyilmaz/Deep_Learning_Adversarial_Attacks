import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class CelebADataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.attr_df = pd.read_csv(csv_file)
        # Assumption: First column is filename (e.g., "000001.jpg"), rest are attributes
        self.img_names = self.attr_df.iloc[:, 0].values
        self.labels = self.attr_df.iloc[:, 1:].values.astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        
        # Open image
        try:
            image = Image.open(img_name).convert('RGB')
        except (FileNotFoundError, OSError):
            # Fallback for missing images (prevents crashing on bad data)
            image = Image.new('RGB', (178, 218))

        if self.transform:
            image = self.transform(image)
            
        # Return attributes as FloatTensor (needed for BCE Loss)
        # Shape: (40,)
        attributes = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, attributes

def get_celeba_loaders(img_dir, csv_file, batch_size=32):
    print("--- Loading CelebA Data ---")
    
    # Resize to 224 for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = CelebADataset(img_dir, csv_file, transform=transform)
    total_len = len(dataset)
    
    # 70% Train, 15% Val, 15% Test
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )
    
    print(f"Total Images: {total_len}")
    print(f"Split: Train={train_len}, Val={val_len}, Test={test_len}")
    
    # Num_workers=2 speeds up loading; set to 0 if debugging on Windows
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader