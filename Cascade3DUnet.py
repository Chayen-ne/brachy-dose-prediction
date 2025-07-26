import torch
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(SEResidualBlock, self).__init__()
        self.module = nn.Module()  # เพิ่ม module เพื่อให้ตรงกับโมเดลที่โหลด
        self.module.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.module.bn1 = nn.BatchNorm3d(out_channels)
        self.module.relu = nn.ReLU(inplace=True)
        self.module.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.module.bn2 = nn.BatchNorm3d(out_channels)
        self.module.se = SEBlock(out_channels)
        self.module.downsample = downsample

    def forward(self, x):
        identity = x
        if self.module.downsample is not None:
            identity = self.module.downsample(x)

        out = self.module.conv1(x)
        out = self.module.bn1(out)
        out = self.module.relu(out)
        out = self.module.conv2(out)
        out = self.module.bn2(out)
        out = self.module.se(out)
        out += identity
        out = self.module.relu(out)
        return out

def crop_to_match(tensor, target_tensor):
    """
    Crops tensor to match the size of target_tensor along spatial dimensions (D, H, W).
    """
    if tensor.size()[2:] == target_tensor.size()[2:]:
        return tensor
        
    diff_depth = tensor.size(2) - target_tensor.size(2)
    diff_height = tensor.size(3) - target_tensor.size(3)
    diff_width = tensor.size(4) - target_tensor.size(4)

    # Crop along each dimension if needed
    if diff_depth > 0 and diff_height > 0 and diff_width > 0:
        tensor = tensor[:, :, 
                    diff_depth // 2:tensor.size(2) - (diff_depth - diff_depth // 2),
                    diff_height // 2:tensor.size(3) - (diff_height - diff_height // 2),
                    diff_width // 2:tensor.size(4) - (diff_width - diff_width // 2)]
    return tensor

class Cascade3DUNet(nn.Module):
    def __init__(self, in_channels=7, num_classes=1):
        super(Cascade3DUNet, self).__init__()
        
        # ปรับโครงสร้างตามชื่อตัวแปรที่พบในโมเดลที่โหลด
        self.input_projection = nn.Conv3d(in_channels, 32, kernel_size=1)
        
        # Encoder blocks
        self.encoder1 = SEResidualBlock(32, 32)
        self.encoder2 = SEResidualBlock(32, 64, downsample=nn.Conv3d(32, 64, kernel_size=1))
        self.encoder3 = SEResidualBlock(64, 128, downsample=nn.Conv3d(64, 128, kernel_size=1))
        self.encoder4 = SEResidualBlock(128, 256, downsample=nn.Conv3d(128, 256, kernel_size=1))
        self.encoder5 = SEResidualBlock(256, 512, downsample=nn.Conv3d(256, 512, kernel_size=1))
        
        # Decoder blocks - ปรับตามชื่อตัวแปรในโมเดล
        self.decoder4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder4_conv = nn.Module()
        self.decoder4_conv.module = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        
        self.decoder3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3_conv = nn.Module()
        self.decoder3_conv.module = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        
        self.decoder2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2_conv = nn.Module()
        self.decoder2_conv.module = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        
        self.decoder1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1_conv = nn.Module()
        self.decoder1_conv.module = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        
        # Projection layers
        self.proj4 = nn.Conv3d(256, 256, kernel_size=1)
        self.proj3 = nn.Conv3d(128, 128, kernel_size=1)
        self.proj2 = nn.Conv3d(64, 64, kernel_size=1)
        self.proj1 = nn.Conv3d(32, 32, kernel_size=1)
        
        # Final layer
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Encoder
        enc1 = self.encoder1(x)  # [B, 32, D, H, W]
        enc2 = self.encoder2(F.max_pool3d(enc1, 2))  # [B, 64, D/2, H/2, W/2]
        enc3 = self.encoder3(F.max_pool3d(enc2, 2))  # [B, 128, D/4, H/4, W/4]
        enc4 = self.encoder4(F.max_pool3d(enc3, 2))  # [B, 256, D/8, H/8, W/8]
        enc5 = self.encoder5(F.max_pool3d(enc4, 2))  # [B, 512, D/16, H/16, W/16]
        
        # Decoder
        x = self.decoder4(enc5)  # [B, 256, D/8, H/8, W/8]
        x = self.decoder4_conv.module(x)  # ปรับการเรียกใช้
        x = crop_to_match(x, enc4) + self.proj4(enc4)
        
        x = self.decoder3(x)  # [B, 128, D/4, H/4, W/4]
        x = self.decoder3_conv.module(x)  # ปรับการเรียกใช้
        x = crop_to_match(x, enc3) + self.proj3(enc3)
        
        x = self.decoder2(x)  # [B, 64, D/2, H/2, W/2]
        x = self.decoder2_conv.module(x)  # ปรับการเรียกใช้
        x = crop_to_match(x, enc2) + self.proj2(enc2)
        
        x = self.decoder1(x)  # [B, 32, D, H, W]
        x = self.decoder1_conv.module(x)  # ปรับการเรียกใช้
        x = crop_to_match(x, enc1) + self.proj1(enc1)
        
        x = self.final_conv(x)  # Final layer
        
        # เพิ่ม sigmoid เพื่อให้ค่าอยู่ระหว่าง 0-1
        x = torch.sigmoid(x)
        
        return x

