import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class NYUImageDepthDataset(Dataset):
    """
    Custom PyTorch Dataset for paired RGB image and Depth Map data (NYU-style).
    Handles path collection, 16-bit depth loading, and synchronized transformations.
    """
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = []
        self.mask_paths = []

        print(f"Collecting file paths from: {self.root_dir}...")
        
        # 1. Get all subdirectories (scene folders) that end with '_out'
        scene_dirs = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d)) and d.endswith('_out')]
        
        for scene_dir in scene_dirs:
            scene_path = os.path.join(root_dir, scene_dir)
            
            # 2. Find ALL .jpg files inside that specific scene folder
            rgb_files = sorted(glob.glob(os.path.join(scene_path, '*.jpg')))
            
            for rgb_path in rgb_files:
                # 3. Assume the mask/depth file has the same name but a .png extension
                mask_path = rgb_path.replace('.jpg', '.png')
                
                if os.path.exists(mask_path):
                    self.image_paths.append(rgb_path)
                    self.mask_paths.append(mask_path)

        print(f"Found {len(self.image_paths)} image-mask pairs.")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)


    def __getitem__(self, idx):
        """Loads and returns one sample of data."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Load the Image and Depth Map
        rgb_image = Image.open(self.image_paths[idx]).convert('RGB')
        # Load PNG without conversion to preserve 16-bit depth values
        mask = Image.open(self.mask_paths[idx]) 

        # 2. Convert to numpy arrays (required by Albumentations)
        image_np = np.array(rgb_image, dtype=np.uint8)
        mask_np = np.array(mask) # Loaded as native dtype (e.g., uint16)

        # 3. Apply Synchronized Transformations
        if self.transforms:
            transformed = self.transforms(image=image_np, mask=mask_np)
            image_output = transformed['image']
            mask_output = transformed['mask']
        else:
            image_output = image_np
            mask_output = mask_np

        # 4. Final Conversion and Formatting for PyTorch
        
        # A. Image Tensor: Ensure it's C x H x W and float
        if not isinstance(image_output, torch.Tensor):
            # Manual ToTensor (H x W x C -> C x H x W)
            image_tensor = torch.from_numpy(image_output.transpose(2, 0, 1)).float() / 255.0
        else:
            # Already a tensor from ToTensorV2, ensure float type
            image_tensor = image_output.float()
        
        # B. Mask/Depth Tensor: Must be torch.float() for depth prediction
        if not isinstance(mask_output, torch.Tensor):
            # Convert NumPy array to Float Tensor
            mask_tensor = torch.from_numpy(mask_output).float() 
        else:
            # Already a tensor, cast to Float
            mask_tensor = mask_output.float() 
        
        # Ensure mask has 1 channel (1, H, W)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0) 

        sample = {
            'image': image_tensor,
            'mask': mask_tensor
        }

        return sample