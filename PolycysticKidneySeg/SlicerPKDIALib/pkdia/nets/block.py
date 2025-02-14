#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional


class SingleConv(nn.Module):
    """{Conv2d, BN, ReLU}"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.single_conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)

    def forward(self, x):
        return self.single_conv(x)


class SimpleConv(nn.Module):
    """{Conv2d, BN, ReLU}"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.simple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1), nn.BatchNorm2d(out_chan), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.simple_conv(x)


class DoubleConv(nn.Module):
    """{Conv2d, BN, ReLU}x2"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """{Conv2d, BN, ReLU}x3"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.triple_conv(x)


class QuadripleConv(nn.Module):
    """{Conv2d, BN, ReLU}x4"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.quadriple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.quadriple_conv(x)


class SimpleDown(nn.Module):
    """maxPool2d + {Conv2d, BN, ReLU}x2"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.simple_down = nn.Sequential(nn.MaxPool2d(2, 2), SimpleConv(in_chan, out_chan))

    def forward(self, x):
        return self.simple_down(x)


class DoubleDown(nn.Module):
    """maxPool2d + {Conv2d, BN, ReLU}x2"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.double_down = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(in_chan, out_chan))

    def forward(self, x):
        return self.double_down(x)


class SimpleUpAE(nn.Module):
    """ConvTranspose2d + {Conv2d, BN, ReLU}x2"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, in_chan, kernel_size=2, stride=2)
        self.conv = SimpleConv(in_chan, out_chan)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DoubleUpAE(nn.Module):
    """ConvTranspose2d + {Conv2d, BN, ReLU}x2"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, in_chan, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_chan, out_chan)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DoubleUp(nn.Module):
    """ConvTranspose2d + {Conv2d, BN, ReLU}x2"""

    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan is None:
            mid_chan = in_chan
        self.conv = DoubleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class TripleUp(nn.Module):
    """ConvTranspose2d + {Conv2d, BN, ReLU}x3"""

    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan is None:
            mid_chan = in_chan
        self.conv = TripleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class QuadripleUp(nn.Module):
    """ConvTranspose2d + {Conv2d, BN, ReLU}x4"""

    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan is None:
            mid_chan = in_chan
        self.conv = QuadripleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Conv2d"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def up_sample2d(x, t, mode="bilinear"):
    """2D up-sampling"""

    return functional.interpolate(x, t.size()[2:], mode=mode, align_corners=False)


class MixBlock(nn.Module):
    """for attention purposes"""

    def __init__(self, in_chan, out_chan):
        super(MixBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(in_chan, out_chan // 4, 5, padding=2)
        self.conv5 = nn.Conv2d(in_chan, out_chan // 4, 7, padding=3)
        self.conv7 = nn.Conv2d(in_chan, out_chan // 4, 9, padding=4)
        self.bn1 = nn.BatchNorm2d(out_chan // 4)
        self.bn3 = nn.BatchNorm2d(out_chan // 4)
        self.bn5 = nn.BatchNorm2d(out_chan // 4)
        self.bn7 = nn.BatchNorm2d(out_chan // 4)
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, x):
        k1 = self.bn1(self.conv1(x))
        k3 = self.bn3(self.conv3(x))
        k5 = self.bn5(self.conv5(x))
        k7 = self.bn7(self.conv7(x))
        return self.nonlinear(torch.cat((k1, k3, k5, k7), dim=1))


class Attention(nn.Module):
    """attention modules"""

    def __init__(self, in_chan, out_chan):
        super(Attention, self).__init__()
        self.mix1 = MixBlock(in_chan, out_chan)
        self.conv1 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.mix2 = MixBlock(out_chan, out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_chan)
        self.norm2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        mix1 = self.conv1(self.mix1(x))
        mix2 = self.mix2(mix1)
        att_map = torch.sigmoid(self.conv2(mix2))
        out = self.norm1(x * att_map) + self.norm2(shortcut)
        return self.relu(out), att_map
