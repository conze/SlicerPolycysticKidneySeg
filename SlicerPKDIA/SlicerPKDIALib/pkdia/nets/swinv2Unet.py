import timm
import torch
import torch.nn as nn

from . import block


class Doublewith2Up(nn.Module):
    """ConvTranspose2d + {Conv2d, BN, ReLU}x2"""

    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan is None:
            mid_chan = in_chan
        self.conv = block.DoubleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.up(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class SwinV2OneDecoder(nn.Module):
    def __init__(
        self,
        model_name="swinv2_cr_tiny_224",
        pretrained=False,
        in_chans=1,
        img_size=(256, 256),
        patch_size=4,
        n_classes=1,
    ):
        super(SwinV2OneDecoder, self).__init__()
        encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
            img_size=img_size,
            patch_size=patch_size,
        )
        base_layers = list(encoder.children())

        self.n_channels = in_chans
        self.n_classes = n_classes

        self.patch_embed = base_layers[0]
        self.skip0 = base_layers[1][0]
        self.skip1 = base_layers[1][1]
        self.skip2 = base_layers[1][2]
        self.center = base_layers[1][3]

        self.deblock1 = block.QuadripleUp(768, 384)
        self.deblock2 = block.QuadripleUp(384, 192)
        self.deblock3 = block.DoubleUp(192, 96)
        self.deblock4 = Doublewith2Up(96, 48)
        self.conv = nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock5 = block.DoubleUp(48, 48, 96)
        self.outc = block.OutConv(48, n_classes)

    def forward(self, x):
        x1 = self.patch_embed(x)
        x2 = self.skip0(x1)
        x3 = self.skip1(x2)
        x4 = self.skip2(x3)
        x5 = self.center(x4)
        out = self.deblock1(x5, x4)
        out = self.deblock2(out, x3)
        out = self.deblock3(out, x2)
        out = self.deblock4(out, x1)
        out = self.deblock5(out, self.conv(x))
        logits = self.outc(out)
        return logits


class SwinV2TwoDecoder(nn.Module):
    def __init__(
        self,
        model_name="swinv2_cr_tiny_224",
        pretrained=False,
        img_size=(256, 256),
        in_chans=1,
        n_classes_1dec=1,
        n_classes_2dec=1,
    ):
        super(SwinV2TwoDecoder, self).__init__()
        encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="", img_size=img_size, in_chans=in_chans
        )
        base_layers = list(encoder.children())

        self.n_channels = in_chans
        self.n_classes = 1  # n_classes_1dec+n_classes_2dec

        self.patch_embed = base_layers[0]
        self.skip0 = base_layers[1][0]
        self.skip1 = base_layers[1][1]
        self.skip2 = base_layers[1][2]
        self.center = base_layers[1][3]

        # first decoder
        self.deblock1_1 = block.QuadripleUp(768, 384)
        self.deblock1_2 = block.QuadripleUp(384, 192)
        self.deblock1_3 = block.DoubleUp(192, 96)
        self.deblock1_4 = Doublewith2Up(96, 48)
        self.conv1 = nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock1_5 = block.DoubleUp(48, 48, 96)
        self.outc1 = block.OutConv(48, n_classes_1dec)

        # second decoder
        self.deblock2_1 = block.QuadripleUp(768, 384)
        self.deblock2_2 = block.QuadripleUp(384, 192)
        self.deblock2_3 = block.DoubleUp(192, 96)
        self.deblock2_4 = Doublewith2Up(96, 48)
        self.conv2 = nn.Conv2d(in_chans, 48, kernel_size=3, padding=1)
        self.deblock2_5 = block.DoubleUp(48, 48, 96)
        self.outc2 = block.OutConv(48, n_classes_2dec)

    def forward(self, x):
        x1 = self.patch_embed(x)
        x2 = self.skip0(x1)
        x3 = self.skip1(x2)
        x4 = self.skip2(x3)
        x5 = self.center(x4)

        # first decoder
        out1 = self.deblock1_1(x5, x4)
        out1 = self.deblock1_2(out1, x3)
        out1 = self.deblock1_3(out1, x2)
        out1 = self.deblock1_4(out1, x1)
        out1 = self.deblock1_5(out1, self.conv1(x))
        logits1 = self.outc1(out1)

        # second decoder
        out2 = self.deblock2_1(x5, x4)
        out2 = self.deblock2_2(out2, x3)
        out2 = self.deblock2_3(out2, x2)
        out2 = self.deblock2_4(out2, x1)
        out2 = self.deblock2_5(out2, self.conv2(x))
        logits2 = self.outc2(out2)

        return logits1, logits2
