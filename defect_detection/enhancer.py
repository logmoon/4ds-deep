"""
defect_detection/enhancer.py
============================
Convolutional Denoising Autoencoder for image quality enhancement.

Architecture (from Ch5 slides):
  Encoder: compresses degraded input → latent representation (bottleneck)
  Decoder: reconstructs clean image from latent representation

Trained to map: blurry/dark/noisy image → clean image
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder with skip connections.
    Input/Output: (B, 3, 256, 256) — RGB image normalized to [0, 1]

    Encoder: 3 → 32 → 64 → 128 channels  (compress)
    Bottleneck: 128 channels latent space
    Decoder: 128 → 64 → 32 → 3 channels  (reconstruct)
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)   # 256 → 128

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)   # 128 → 64

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)   # 64 → 32  ← bottleneck

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # skip from enc3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),    # skip from enc2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),    # skip from enc1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),          # → RGB output
            nn.Sigmoid(),                                          # [0, 1] range
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck = self.pool3(e3)

        # Decode with skip connections (like U-Net)
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return d1