import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import local modules
from utils.utils import load_config
from utils.transforms import get_transforms
from utils.metrics import compute_segmentation_metrics
from datasets.image_dataset import NYUImageDepthDataset
from models.segmentation_model import SegmentationModel # Or resnet_model

def evaluate_model(cfg):
    
    # 1. Setup
    device = torch.device(cfg['DEVICE'] if torch.cuda.is_available() else 'cpu')
    
    # 2. Data Loading (Test Set)
    test_transforms = get_transforms(cfg, split='test')
    test_dataset = NYUImageDepthDataset(
        root_dir=os.path.join(cfg['DATA']['ROOT_DIR'], cfg['DATA']['TEST_DIR']),
        transforms=test_transforms
    )
    test_loader = DataLoader(test_dataset, 
                             batch_size=cfg['TRAIN']['BATCH_SIZE'], 
                             shuffle=False, # Must be False for consistent evaluation
                             num_workers=4)
    print(f"Test Set: {len(test_dataset)} samples loaded.")

    # 3. Model Initialization and Loading Weights
    model = SegmentationModel(
        num_classes=cfg['DATA']['NUM_CLASSES'],
        pretrained=False, # We are loading the trained weights, not ImageNet weights
        freeze_layers=False
    ).to(device)
    
    # Load the best weights saved during training
    weights_path = os.path.join(cfg['LOGS']['SAVE_DIR'], 'best_model.pth')
    if not os.path.exists(weights_path):
        print(f"Error: Could not find model weights at {weights_path}. Did training complete?")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Set to evaluation mode
    print(f"Successfully loaded model weights from {weights_path}.")

    # 4. Inference and Collection
    all_pred_masks = []
    all_true_masks = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            images = data['image'].to(device)
            true_masks = data['mask'].squeeze(1).long() # True mask (N, H, W)

            outputs = model(images)
            
            # Get the predicted class index by finding the max logit along the channel (class) dimension
            # Outputs shape (N, C, H, W) -> Pred shape (N, H, W)
            predicted_masks = torch.argmax(outputs, dim=1) 
            
            all_true_masks.append(true_masks.cpu().numpy())
            all_pred_masks.append(predicted_masks.cpu().numpy())

    # Concatenate all batches into large numpy arrays
    final_true_masks = np.concatenate(all_true_masks, axis=0)
    final_pred_masks = np.concatenate(all_pred_masks, axis=0)
    
    # 5. Metric Calculation
    print("\n--- Calculating Final Metrics ---")
    metrics = compute_segmentation_metrics(final_pred_masks, final_true_masks, cfg['DATA']['NUM_CLASSES'])
    
    # 6. Reporting and Saving Results
    print("\n✅ FINAL EVALUATION RESULTS:")
    print(f"  Pixel Accuracy: {metrics['Pixel Accuracy'] * 100:.2f}%")
    print(f"  Mean IoU (mIoU): {metrics['Mean IoU (mIoU)'] * 100:.2f}%")
    print(f"  Frequency Weighted IoU: {metrics['FWIoU'] * 100:.2f}%")
    
    # You can save the metrics dict to a JSON file in the 'results/' folder here

if __name__ == '__main__':
    # You need to ensure config.yaml is in the correct location
    evaluate_model(load_config('configs/config.yaml'))