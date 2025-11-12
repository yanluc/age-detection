import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm
from model import effnetv2_s, effnetv2_m, effnetv2_l, effnetv2_xs

class AgeDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
        self.label_map = {
            0: 5.5,    # 'age 01-10'
            1: 15.5,   # 'age 11-20'
            2: 25.5,   # 'age 21-30'
            3: 35.5,   # 'age 31-40'
            4: 48.0,   # 'age 41-55'
            5: 60.5,   # 'age 56-65'
            6: 73.0,   # 'age 66-80'
            7: 85.0    # 'age 80 +' 
        }
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert('RGB')
        class_label = sample["label"]
        age = self.label_map[class_label] 

        if self.transform:
            image = self.transform(image)
        
        return image, age

def get_transforms(input_size=200, augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, ages in pbar:
        images = images.to(device)
        ages = ages.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_mae += torch.abs(outputs - ages).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    
    return epoch_loss, epoch_mae

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    
    with torch.no_grad():
        for images, ages in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            ages = ages.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, ages)
            
            running_loss += loss.item() * images.size(0)
            running_mae += torch.abs(outputs - ages).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    
    return epoch_loss, epoch_mae

def main():

    model_size = 'xs'
    epochs = 50
    batch_size = 90 
    lr = 0.001
    input_size = 200
    val_split = 0.2
    checkpoint_dir = 'data/checkpoints'
    num_workers = 4
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    dataset = load_dataset("prithivMLmods/Face-Age-10K")
    split_dataset = dataset['train'].train_test_split(test_size=val_split, shuffle=True, seed=42)
    
    train_data = split_dataset['train']
    val_data = split_dataset['test']
    
    train_transform, val_transform = get_transforms(input_size, augment=True)
    
    train_dataset = AgeDataset(train_data, transform=train_transform)
    val_dataset = AgeDataset(val_data, transform=val_transform)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    
    if model_size == 'xs':
        model = effnetv2_xs(num_classes=1)
    elif model_size == 's':
        model = effnetv2_s(num_classes=1)
    elif model_size == 'm':
        model = effnetv2_m(num_classes=1)
    else:
        model = effnetv2_l(num_classes=1)
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_mae = float('inf')
    
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f} years')
        print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f} years')
        
        # Save checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_effnetv2_{model_size}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
            }, checkpoint_path)
            print(f'Saved best model with MAE: {val_mae:.2f} years')
    
    print(f'\nTraining completed. Best validation MAE: {best_val_mae:.2f} years')

if __name__ == '__main__':
    main()