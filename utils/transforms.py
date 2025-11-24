import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms(cfg, split='train'):
    """
    Generates the appropriate transformation pipeline for the given split (train/val/test).

    Args:
        cfg (dict): Configuration dictionary containing AUGMENTATION and DATA settings.
        split (str): 'train' or 'val'/'test'.
    
    Returns:
        A.Compose: An Albumentations composition object.
    """
    
    # 1. Define standard normalization values
    # These are often derived from the dataset itself, but ImageNet values are a good default start.
    MEAN = [0.485, 0.456, 0.406]  # ImageNet mean for RGB channels
    STD = [0.229, 0.224, 0.225]   # ImageNet standard deviation

    # Get parameters from the configuration
    IMG_SIZE = cfg['DATA']['IMG_SIZE']
    PROB = cfg['AUGMENTATION']['PROBABILITY']
    ROTATION_LIMIT = cfg['AUGMENTATION']['RANDOM_ROTATION']
    
    # --- 2. Training Pipeline (with Augmentation) ---
    if split == 'train':
        return A.Compose([
            # A. Geometric Transforms (Applied to both image and mask)
            # Resizing is always needed to ensure uniform input size
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
            
            # Synchronized Random Augmentations
            A.HorizontalFlip(p=PROB if cfg['AUGMENTATION']['FLIP'] else 0),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=ROTATION_LIMIT, p=PROB, border_mode=0),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0),
            
            # A. Random Crop (Applied to both image and mask)
            A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, always_apply=True), 

            # B. Non-Geometric (Color/Intensity) Transforms (Applied only to the image)
            # Note: We must specify 'image' in the targets for intensity transforms
            A.GaussNoise(p=PROB * 0.5), # Apply noise
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=PROB),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=PROB),

            # C. Final Normalization and Tensor Conversion
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, always_apply=True),
            # ToTensorV2 is an Albumentations utility that converts HWC (Numpy) to CHW (PyTorch)
            # and handles the final data type conversion.
            ToTensorV2(transpose_mask=False) # Keep mask as HW or 1xHW, let the Dataset class handle the final unsqueeze if needed
            
        ], 
        # Crucial for paired data: specify that the transformation includes a 'mask' target
        p=1.0, 
        additional_targets={'mask': 'mask'}) 

    # --- 3. Validation/Test Pipeline (Preprocessing Only) ---
    else:
        # No random geometric augmentations here!
        return A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, always_apply=True),
            ToTensorV2(transpose_mask=False)
        ], 
        p=1.0, 
        additional_targets={'mask': 'mask'})