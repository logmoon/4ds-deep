"""
Categorization model loader — EfficientNet-B0 fine-tuned on 9 product classes.
Loads once on first call; subsequent calls reuse the cached model.
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "product_categorizer_v3.pth")

_model = None
_class_names = None
_transform = None


def _load():
    global _model, _class_names, _transform

    if _model is not None:
        return

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {_MODEL_PATH}. "
            "Place product_categorizer_v2.pth in the project root."
        )

    checkpoint = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    img_size = checkpoint["img_size"]
    mean = checkpoint["imagenet_mean"]
    std = checkpoint["imagenet_std"]

    # Rebuild the same architecture used during training.
    # The checkpoint keys use a "backbone." prefix, so we wrap EfficientNet-B0
    # in a container class to match the original training structure.
    class ProductCategorizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.efficientnet_b0(weights=None)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(1280, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.backbone(x)

    net = ProductCategorizer()
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    _model = net
    _class_names = class_names
    _transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def predict(image: Image.Image) -> dict:
    """
    Run inference on a PIL image.
    Returns category, confidence, and top-3 predictions.
    """
    _load()

    tensor = _transform(image.convert("RGB")).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top3 = probs.topk(3)
    top3_results = [
        {"category": _class_names[idx.item()], "confidence": round(score.item(), 4)}
        for score, idx in zip(top3.values, top3.indices)
    ]

    return {
        "category": top3_results[0]["category"],
        "confidence": top3_results[0]["confidence"],
        "top3": top3_results,
    }
