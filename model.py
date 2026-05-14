"""
Standard 2-D UNet for binary segmentation.
Input:  (B, 4, 256, 256)  — 4 BraTS modalities
Output: (B, 1, 256, 256)  — logit map (apply sigmoid for probability)

Memory: base_ch=64 → ~31M params.
        Use gradient_checkpointing=True to trade compute for ~40% less activation memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x, use_ckpt=False):
        x = self.pool(x)
        if use_ckpt:
            return checkpoint(self.conv, x, use_reentrant=False)
        return self.conv(x)


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x, skip, use_ckpt=False):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        cat = torch.cat([skip, x], dim=1)
        if use_ckpt:
            return checkpoint(self.conv, cat, use_reentrant=False)
        return self.conv(cat)


class UNet(nn.Module):
    """
    4-level UNet. base_ch=64 → ~31M params.
    Set gradient_checkpointing=True to save ~40% activation memory at ~20% extra compute.
    """

    def __init__(self, in_channels: int = 4, base_ch: int = 64,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.use_ckpt = gradient_checkpointing
        c = base_ch
        self.inc   = _DoubleConv(in_channels, c)
        self.down1 = _Down(c,      c * 2)
        self.down2 = _Down(c * 2,  c * 4)
        self.down3 = _Down(c * 4,  c * 8)
        self.down4 = _Down(c * 8,  c * 16)
        self.up1   = _Up(c * 16,   c * 8)
        self.up2   = _Up(c * 8,    c * 4)
        self.up3   = _Up(c * 4,    c * 2)
        self.up4   = _Up(c * 2,    c)
        self.head  = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        ck = self.use_ckpt and self.training
        x1 = self.inc(x)
        x2 = self.down1(x1, ck)
        x3 = self.down2(x2, ck)
        x4 = self.down3(x3, ck)
        x5 = self.down4(x4, ck)
        x  = self.up1(x5, x4, ck)
        x  = self.up2(x,  x3, ck)
        x  = self.up3(x,  x2, ck)
        x  = self.up4(x,  x1, ck)
        return self.head(x)
