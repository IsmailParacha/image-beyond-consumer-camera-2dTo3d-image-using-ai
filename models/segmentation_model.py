import torch
import torch.nn as nn
from torchvision import models

class SegmentationModel(nn.Module):
    """
    Encoder-Decoder Model based on ResNet for Depth Estimation/Segmentation.
    Uses size-based upsampling to ensure correct skip connection dimensions.
    """
    def __init__(self, num_classes, pretrained=True, freeze_layers=False):
        super().__init__()
        
        # 1. ENCODER (ResNet Backbone)
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1 # -> 256 channels (e2)
        self.layer2 = self.backbone.layer2 # -> 512 channels (e3)
        self.layer3 = self.backbone.layer3 # -> 1024 channels (e4)
        self.layer4 = self.backbone.layer4 # -> 2048 channels (e5)

        # 2. DECODER
        self.upconv1 = self._make_up_block(2048 + 1024, 1024) 
        self.upconv2 = self._make_up_block(1024 + 512, 512)
        self.upconv3 = self._make_up_block(512 + 256, 256)
        self.upconv4 = self._make_up_block(256 + 64, 128) 

        # FINAL OUTPUT: num_classes=1 for depth estimation
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def _make_up_block(self, in_channels, out_channels):
        """Helper for creating a combined upsampling and convolution block."""
        return nn.Sequential(
            # Using ConvTranspose for initial upsampling
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _crop_tensor(self, target_tensor, size):
        """Center crops the target_tensor to match the given size (H, W)."""
        target_H, target_W = target_tensor.shape[2:]
        size_H, size_W = size
        
        if target_H == size_H and target_W == size_W:
            return target_tensor
        
        start_H = (target_H - size_H) // 2
        start_W = (target_W - size_W) // 2
        
        return target_tensor[:, :, start_H : start_H + size_H, start_W : start_W + size_W]


    def forward(self, x):
        # 1. ENCODER PASS 
        x1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.maxpool(x1) # Skip P1 (64 channels)

        e2 = self.layer1(p1)  # Skip E2 (256 channels)
        e3 = self.layer2(e2)  # Skip E3 (512 channels)
        e4 = self.layer3(e3)  # Skip E4 (1024 channels)
        e5 = self.layer4(e4)  # Bottleneck (2048 channels)

        # 2. DECODER PASS (Fixed Upsampling)
        
        # UP 1: Concat with E4
        up_e5 = nn.Upsample(size=e4.shape[2:], mode='bilinear', align_corners=False)(e5) 
        e4_cropped = self._crop_tensor(e4, up_e5.shape[2:]) 
        d1 = torch.cat([e4_cropped, up_e5], dim=1)
        d1 = self.upconv1(d1) 

        # UP 2: Concat with E3
        up_d1 = nn.Upsample(size=e3.shape[2:], mode='bilinear', align_corners=False)(d1)
        e3_cropped = self._crop_tensor(e3, up_d1.shape[2:]) 
        d2 = torch.cat([e3_cropped, up_d1], dim=1)
        d2 = self.upconv2(d2)

        # UP 3: Concat with E2
        up_d2 = nn.Upsample(size=e2.shape[2:], mode='bilinear', align_corners=False)(d2)
        e2_cropped = self._crop_tensor(e2, up_d2.shape[2:]) 
        d3 = torch.cat([e2_cropped, up_d2], dim=1)
        d3 = self.upconv3(d3)

        # UP 4: Concat with P1
        up_d3 = nn.Upsample(size=p1.shape[2:], mode='bilinear', align_corners=False)(d3)
        p1_cropped = self._crop_tensor(p1, up_d3.shape[2:]) 
        d4 = torch.cat([p1_cropped, up_d3], dim=1)
        d4 = self.upconv4(d4)

        # 3. FINAL OUTPUT
        output = self.final_conv(d4)
        
        # FIX: Final upsampling to match the INPUT size (which should match the target size)
        final_H, final_W = x.shape[2:] 
        
        if output.shape[2] != final_H or output.shape[3] != final_W:
             output = nn.Upsample(size=(final_H, final_W), mode='bilinear', align_corners=False)(output)
        
        return output