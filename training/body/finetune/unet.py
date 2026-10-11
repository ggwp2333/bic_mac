import torch
import torch.nn as nn


# -----------------------------
# Residual Conv Block
# -----------------------------

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)

        self.skip = None

        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.relu(out)

        return out


# -----------------------------
# Encoder Block
# -----------------------------

class EncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.block = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):

        x = self.block(x)
        p = self.pool(x)

        return x, p


# -----------------------------
# Decoder Block
# -----------------------------

class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)

        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([x, skip], dim=1)

        x = self.block(x)

        return x


# -----------------------------
# UNet
# -----------------------------

class UNet3D(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, base_channels=64):

        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(c4, c5),
            nn.Dropout3d(0.2)
        )

        # Decoder
        self.dec4 = DecoderBlock(c5, c4)
        self.dec3 = DecoderBlock(c4, c3)
        self.dec2 = DecoderBlock(c3, c2)
        self.dec1 = DecoderBlock(c2, c1)

        # Output
        self.out_conv = nn.Conv3d(c1, out_channels, 1)

    def forward(self, x):

        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.out_conv(d1)

        return out


# -----------------------------
# Builder
# -----------------------------

# Width for the pretraining run. baseline_v7 ships width 64 (91.7 M
# parameters) fitted to 75 subjects; the pretraining corpus is 171 scans from
# 123 subjects, so the width is raised to 96 (206.3 M). Depth is deliberately
# unchanged: the encoder still pools four times, so patch sizes stay
# multiples of 16 and the effective receptive field -- measured at r90 = 49
# voxels and flat in patch size (plan.md §5.3) -- is the same function it was.
# Changing depth instead would move the receptive field and break weight
# transfer to any width-matched downstream model.
DEFAULT_BASE_CHANNELS = 96


def build_model(in_channels=4, base_channels=DEFAULT_BASE_CHANNELS):

    return UNet3D(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_channels,
    )


def describe_model(model) -> str:

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return f"{total / 1e6:.1f}M parameters ({trainable / 1e6:.1f}M trainable)"
