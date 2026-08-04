"""IC-U-Net architecture copied from AIEEG/model/cumbersome_model2.py.

Only comments/formatting and unused skip arguments were simplified. Layer names,
activations, channel sizes, kernels, and forward order are unchanged so the
original checkpoint state_dict loads directly.
"""

from __future__ import annotations

import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool1d(2), DoubleConv(in_channels, out_channels, kernel_size))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1, _x2):
        return self.conv(self.up(x1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)


class UNet1(nn.Module):
    def __init__(self, n_channels: int = 30, n_classes: int = 30, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64, kernel_size=7)
        self.down1 = Down(64, 128, kernel_size=7)
        self.down2 = Down(128, 256, kernel_size=5)
        self.down3 = Down(256, 512, kernel_size=3)
        self.up1 = Up(512, 256, kernel_size=3)
        self.up2 = Up(256, 128, kernel_size=3)
        self.up3 = Up(128, 64, kernel_size=3)
        self.outc = OutConv(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)
