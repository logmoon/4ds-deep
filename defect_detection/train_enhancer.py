"""
defect_detection/train_enhancer.py
====================================
Run this ONCE in Google Colab (GPU) to train the autoencoder.
Saves: defect_detection/enhancer.pth

Usage:
    python defect_detection/train_enhancer.py

What it does (Denoising Autoencoder — slide 12):
  1. Takes clean images from your static/images folder (or downloads a dataset)
  2. Artificially degrades them: blur + noise + darkening  ← "add noise to input"
  3. Trains encoder-decoder to recover the clean original  ← "train to recover"
  4. At inference: feed bad image → get enhanced image out
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image, ImageFilter
import numpy as np
import random
from pathlib import Path

# ── Import the model ──────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent))
from defect_detection.enhancer import ConvAutoencoder

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE    = 256
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-3
SAVE_PATH   = 'defect_detection/enhancer.pth'

# If you have more data, point this to a folder of clean images
# We'll use DIV2K — a standard image enhancement dataset (free)
DATA_DIR    = 'enhancement_data'

print(f'Device: {DEVICE}')


# ── Dataset ───────────────────────────────────────────────────────────────────
class DegradedImageDataset(Dataset):
    """
    Denoising Autoencoder dataset.
    Input  (x): artificially degraded image
    Target (y): original clean image

    Degradations applied randomly (matching real defects from model.py):
    - Gaussian blur    → fixes blurry images
    - Gaussian noise   → fixes noisy images
    - Brightness drop  → fixes dark images
    - Combined         → fixes multiple issues at once
    """

    def __init__(self, image_paths, augment=True):
        self.paths   = image_paths
        self.augment = augment
        self.to_tensor = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),           # → [0, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def degrade(self, img_pil):
        """Apply random degradation to a clean PIL image."""
        degradation = random.choice(['blur', 'noise', 'dark', 'combined'])

        if degradation == 'blur' or degradation == 'combined':
            radius = random.uniform(1.5, 4.0)
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

        if degradation == 'noise' or degradation == 'combined':
            arr   = np.array(img_pil).astype(np.float32)
            noise = np.random.normal(0, random.uniform(15, 40), arr.shape)
            arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(arr)

        if degradation == 'dark' or degradation == 'combined':
            factor  = random.uniform(0.3, 0.65)
            img_pil = TF.adjust_brightness(img_pil, factor)

        return img_pil

    def __getitem__(self, idx):
        img_clean = Image.open(self.paths[idx]).convert('RGB')

        # Random crop for augmentation
        if self.augment:
            i, j, h, w = T.RandomCrop.get_params(img_clean, output_size=(IMG_SIZE, IMG_SIZE))
            img_clean = TF.crop(img_clean, i, j, h, w)
            if random.random() > 0.5:
                img_clean = TF.hflip(img_clean)
        else:
            img_clean = img_clean.resize((IMG_SIZE, IMG_SIZE))

        img_degraded = self.degrade(img_clean.copy())

        x = T.ToTensor()(img_degraded)   # degraded → model input
        y = T.ToTensor()(img_clean)      # clean    → target

        return x, y


def get_image_paths(folder, exts={'.jpg', '.jpeg', '.png', '.webp'}):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(root, f))
    return paths


# ── Download dataset if needed ────────────────────────────────────────────────
def prepare_data():
    """
    Uses DIV2K validation set (100 high-quality images, ~400MB).
    You can replace this with any folder of clean images.
    """
    if not os.path.exists(DATA_DIR):
        print('Downloading DIV2K validation set...')
        os.makedirs(DATA_DIR, exist_ok=True)
        os.system('wget -q https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip')
        os.system(f'unzip -q DIV2K_valid_HR.zip -d {DATA_DIR}')
        os.system('rm DIV2K_valid_HR.zip')
        print('Dataset ready.')

    paths = get_image_paths(DATA_DIR)

    # Fallback: use whatever images are in static/images
    if len(paths) < 10:
        print('Using static/images as training data (limited — add more for better results)')
        paths = get_image_paths('static/images')

    print(f'Found {len(paths)} training images')
    return paths


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    paths = prepare_data()
    random.shuffle(paths)

    split     = int(0.9 * len(paths))
    train_ds  = DegradedImageDataset(paths[:split], augment=True)
    val_ds    = DegradedImageDataset(paths[split:], augment=False)
    train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = ConvAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Combined loss: MSE (pixel accuracy) + perceptual smoothness
    mse_loss  = nn.MSELoss()

    best_val  = float('inf')
    os.makedirs('defect_detection', exist_ok=True)

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += mse_loss(pred, y).item()

        train_loss /= len(train_dl)
        val_loss   /= len(val_dl)
        scheduler.step()

        print(f'Epoch {epoch+1:2d}/{EPOCHS} | Train: {train_loss:.5f} | Val: {val_loss:.5f}')

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  ✅ Saved best model (val={best_val:.5f})')

        # Save sample comparison every 5 epochs
        if (epoch + 1) % 5 == 0 and len(val_ds) > 0:
            x_sample, y_sample = val_ds[0]
            with torch.no_grad():
                enhanced = model(x_sample.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
            save_image(
                torch.stack([x_sample, enhanced, y_sample]),
                f'sample_epoch_{epoch+1}.png',
                nrow=3
            )
            print(f'  📸 Saved sample: [degraded | enhanced | original]')

    print(f'\n✅ Training complete. Best val loss: {best_val:.5f}')
    print(f'Model saved to: {SAVE_PATH}')
    print('Download enhancer.pth and put it in defect_detection/')


if __name__ == '__main__':
    train()