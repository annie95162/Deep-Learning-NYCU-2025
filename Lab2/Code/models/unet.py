import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # 兩次 conv + ReLU
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 保持尺寸
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=64):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_c)
        self.enc2 = DoubleConv(base_c, base_c * 2)
        self.enc3 = DoubleConv(base_c * 2, base_c * 4)
        self.enc4 = DoubleConv(base_c * 4, base_c * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_c * 8, base_c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_c * 16, base_c * 8)

        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_c * 8, base_c * 4)

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_c * 4, base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.dec1 = DoubleConv(base_c * 2, base_c)

        # Output layer
        self.out_conv = nn.Conv2d(base_c, out_channels, 1)  # 輸出

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Bottleneck
        x5 = self.bottleneck(self.pool(x4))

        # Decoder with skip connections
        d4 = self.up4(x5)
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        out = self.out_conv(d1)
        return out
