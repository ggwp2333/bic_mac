"""The head cVAE (CondUNetVAE), verbatim from training/head/*/train_vae_ddp.py.

Inference only: `ct=None` decodes from the prior mean, so the prediction is
deterministic.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        idt = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.skip is not None:
            idt = self.skip(idt)
        return self.relu(out + idt)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.block(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))


class CondUNetVAE(nn.Module):
    def __init__(self, in_channels=4, base=32, post_base=16, z_ch=8, dropout=0.2):
        super().__init__()
        self.z_ch = z_ch
        self.enc1 = EncoderBlock(in_channels, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.enc3 = EncoderBlock(base * 2, base * 4)
        self.enc4 = EncoderBlock(base * 4, base * 8)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base * 8, base * 16), nn.Dropout3d(dropout)
        )
        self.prior = nn.Sequential(
            ResidualBlock(base * 16, base * 4), nn.Conv3d(base * 4, 2 * z_ch, 1)
        )
        pb = post_base
        self.post = nn.Sequential(
            ResidualBlock(in_channels + 1, pb), nn.MaxPool3d(2),
            ResidualBlock(pb, pb * 2), nn.MaxPool3d(2),
            ResidualBlock(pb * 2, pb * 4), nn.MaxPool3d(2),
            ResidualBlock(pb * 4, pb * 8), nn.MaxPool3d(2),
            ResidualBlock(pb * 8, pb * 8),
        )
        self.post_head = nn.Conv3d(pb * 8, 2 * z_ch, 1)
        self.z_merge = ResidualBlock(base * 16 + z_ch, base * 16)
        self.dec4 = DecoderBlock(base * 16, base * 8)
        self.dec3 = DecoderBlock(base * 8, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2)
        self.dec1 = DecoderBlock(base * 2, base)
        self.out_conv = nn.Conv3d(base, 1, 1)

    def encode_cond(self, cond):
        s1, p1 = self.enc1(cond)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        return self.bottleneck(p4), (s1, s2, s3, s4)

    def decode(self, b, skips, z):
        s1, s2, s3, s4 = skips
        x = self.z_merge(torch.cat([b, z], dim=1))
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.out_conv(x)

    @staticmethod
    def _split(h):
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(self, cond, ct=None, sample=False):
        """ct given -> posterior z (training). ct None -> prior mean (inference)."""
        b, skips = self.encode_cond(cond)
        mu_p, logvar_p = self._split(self.prior(b))
        if ct is not None:
            mu_q, logvar_q = self._split(self.post_head(self.post(torch.cat([cond, ct], dim=1))))
            z = mu_q + torch.randn_like(mu_q) * (0.5 * logvar_q).exp() if self.training else mu_q
        else:
            mu_q, logvar_q = mu_p, logvar_p
            z = mu_p + torch.randn_like(mu_p) * (0.5 * logvar_p).exp() if sample else mu_p
        return self.decode(b, skips, z), (mu_q, logvar_q, mu_p, logvar_p)


def build_head_vae(cfg, in_channels=4):
    """Build from the cfg dict stored inside the training checkpoint."""
    return CondUNetVAE(
        in_channels=in_channels,
        base=int(cfg["base_ch"]),
        post_base=int(cfg["post_base_ch"]),
        z_ch=int(cfg["z_ch"]),
    )
