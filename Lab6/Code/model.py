import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, t):
        t = t.float().unsqueeze(-1) / 1000 # default: 1000/1000，不爆梯度
        return self.lin(t) # 擴充維度變成128層，embedding讓模型學複雜特徵

class LabelEmbedding(nn.Module): # 將one-hot label轉成跟時間一樣維度(128)
    def __init__(self, num_classes, dim):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(num_classes, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, y):
        return self.lin(y)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential( # 兩層convolution + batchnorm 一層ReLU
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        # 如果輸入和輸出維度不同，用1x1卷積做shortcut，否則輸入輸出相加
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x): # 讓差異加上源資訊
        return nn.ReLU(inplace=True)(self.block(x) + self.shortcut(x)) # 再經過一個ReLU激活

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, cond_dim=128, num_classes=24):
        super().__init__()
        self.time_emb = TimeEmbedding(cond_dim)
        self.label_emb = LabelEmbedding(num_classes, cond_dim)
        # Encoder
        # 解析度下降，通道數上升
        self.enc1 = ResidualBlock(in_channels, 64)    # 64x64
        self.enc2 = ResidualBlock(64, 128)            # 32x32
        self.enc3 = ResidualBlock(128, 256)           # 16x16
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.middle = ResidualBlock(256, 256)         # 8x8
        # Decoder
        # 先上採樣，skip connection結合後再卷積
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2) # 16x16
        self.dec1 = ResidualBlock(128 + 256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)  # 32x32
        self.dec2 = ResidualBlock(64 + 128, 64)
        self.up3 = nn.ConvTranspose2d(64, 64, 2, 2)   # 64x64
        self.dec3 = ResidualBlock(64 + 64, 64)
        self.out_conv = nn.Conv2d(64, in_channels, 1) # RGB = 3
        self.cond_proj = nn.Linear(cond_dim * 2, 256) # embedding合併後轉256維
        
    def forward(self, x, t, y):
        te = self.time_emb(t)
        le = self.label_emb(y)
        # 合併兩個embedding，unsqueeze讓shape變成[B, 256, 1, 1]方便家道feature map
        cond = self.cond_proj(torch.cat([te, le], dim=-1)).unsqueeze(-1).unsqueeze(-1) #一維轉四維
        e1 = self.enc1(x)                # [B, 64, 64, 64]
        e2 = self.enc2(self.pool(e1))    # [B, 128, 32, 32]
        e3 = self.enc3(self.pool(e2))    # [B, 256, 16, 16]
        # bottleneck
        m = self.middle(self.pool(e3)) + cond  # [B, 256, 8, 8]
        d1 = self.up1(m)                 # [B, 128, 16, 16]
        d1 = self.dec1(torch.cat([d1, e3], dim=1)) # skip connection
        d2 = self.up2(d1)                # [B, 64, 32, 32]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d3 = self.up3(d2)                # [B, 64, 64, 64]
        d3 = self.dec3(torch.cat([d3, e1], dim=1))
        out = self.out_conv(d3)          # [B, 3, 64, 64]
        return out
