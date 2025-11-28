import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # ResNet
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def make_layer(in_channels, out_channels, num_blocks, stride=1):
    layers = []

    # 第一個 block 可能需要 downsample
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    layers.append(BasicBlock(in_channels, out_channels, stride, downsample))

    # 其他 block 保持 stride = 1
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels))

    return nn.Sequential(*layers)


class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = make_layer(64, 64, num_blocks=3)
        self.layer2 = make_layer(64, 128, num_blocks=4, stride=2)
        self.layer3 = make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4 = make_layer(256, 512, num_blocks=3, stride=2)

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class Up(nn.Module):
    """上採樣 + conv"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        
        # 使用 F.interpolate，與 skip 大小匹配
        x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = ResNet34Encoder()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up1 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up3 = Up(128, 64, 64)
        self.up4 = Up(64, 64, 64)

        # 輸出層
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)  # skip connections
        x = self.bottleneck(x4)
        
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x1)

        # 確保最後尺寸匹配
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        
        out = self.out_conv(x)
        return out
