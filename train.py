import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Assuming imports from your project:
from models.segmentation_model import SegmentationModel
from datasets.image_dataset import NYUImageDepthDataset
from utils.transforms import get_transforms
from utils.utils import load_config # Assuming you have a load_config function

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch_idx, data in enumerate(tqdm(dataloader, desc="Training")):
        images = data['image'].to(device)
        target_masks = data['mask'].to(device) # Target is [N, 1, H, W]

        # FIX: Ensure target mask shape is [N, 1, H, W]
        # This is a redundant check if the dataset is correct, but safer.
        if target_masks.dim() == 3:
            target_masks = target_masks.unsqueeze(1) 

        optimizer.zero_grad()
        
        outputs = model(images)
        
        # NOTE: Loss calculation now works because: 
        # outputs: [N, 1, H, W] and target_masks: [N, 1, H, W]
        loss = criterion(outputs, target_masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Add validation_epoch function here (similar structure to train_epoch but with model.eval() and torch.no_grad())

def main():
    # 1. Configuration
    # Assuming config path is fixed
    cfg = load_config('configs/config.yaml')
    print("Configuration loaded.")

    # 2. Setup Device
    device = torch.device(cfg['SYSTEM']['DEVICE'] if torch.cuda.is_available() else 'cpu')
    print(f"Model initialized on {device}.")

    # 3. Data Loading
    root_dir = cfg['DATA']['ROOT_DIR']
    train_dir = cfg['DATA']['TRAIN_DIR']
    val_dir = cfg['DATA']['VAL_DIR']
    img_size = cfg['DATA']['IMG_SIZE']

    train_transforms = get_transforms(cfg, 'train')
    val_transforms = get_transforms(cfg, 'val')
    
    train_dataset = NYUImageDepthDataset(os.path.join(root_dir, train_dir), transforms=train_transforms)
    val_dataset = NYUImageDepthDataset(os.path.join(root_dir, val_dir), transforms=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=False, num_workers=6, pin_memory=True)

    # 4. Model Initialization
    num_classes = cfg['DATA']['NUM_CLASSES'] # Should be 1
    model = SegmentationModel(num_classes=num_classes).to(device)

    # 5. Define Loss Function and Optimizer
    # FIX: Use MSELoss for Depth Estimation (Regression)
    criterion = nn.MSELoss().to(device) 
    
    # FIX: Ensure numerical values from config are explicitly cast as floats
    wd = float(cfg['TRAIN']['WEIGHT_DECAY']) 
    lr = float(cfg['TRAIN']['LEARNING_RATE'])
    
    if cfg['TRAIN']['OPTIMIZER'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # ... add SGD or other optimizers

    # 6. Training Loop
    best_val_loss = float('inf')
    num_epochs = cfg['TRAIN']['EPOCHS']

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        # val_loss = validation_epoch(model, val_loader, criterion, device) # Uncomment once validation_epoch is ready
        val_loss = 0 # Placeholder if validation_epoch is not ready
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Assuming you have a save_checkpoint function
            # save_checkpoint(model, optimizer, epoch, best_val_loss, 'checkpoints/best_model.pth') 
            print("Checkpoint saved.")

if __name__ == '__main__':
    # Ensure 'checkpoints' directory exists
    os.makedirs('checkpoints', exist_ok=True)
    main()