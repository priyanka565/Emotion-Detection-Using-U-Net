import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Encoder with more features
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
        
        # Attention blocks
        self.att1 = AttentionBlock(1024)
        self.att2 = AttentionBlock(512)
        self.att3 = AttentionBlock(256)
        self.att4 = AttentionBlock(128)
        
        # Decoder with skip connections
        self.dec1 = self.conv_block(1024 + 512, 512)
        self.dec2 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec4 = self.conv_block(128 + 64, 64)
        
        # Dropout layers
        self.dropout = nn.Dropout(0.5)
        
        # Final classification
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e5 = self.enc5(F.max_pool2d(e4, 2))
        
        # Apply attention
        e5 = self.att1(e5)
        e4 = self.att2(e4)
        e3 = self.att3(e3)
        e2 = self.att4(e2)
        
        # Decoder with dropout
        d1 = self.dec1(torch.cat([F.interpolate(e5, scale_factor=2), e4], dim=1))
        d1 = self.dropout(d1)
        
        d2 = self.dec2(torch.cat([F.interpolate(d1, scale_factor=2), e3], dim=1))
        d2 = self.dropout(d2)
        
        d3 = self.dec3(torch.cat([F.interpolate(d2, scale_factor=2), e2], dim=1))
        d3 = self.dropout(d3)
        
        d4 = self.dec4(torch.cat([F.interpolate(d3, scale_factor=2), e1], dim=1))
        d4 = self.dropout(d4)
        
        return self.final(d4)
