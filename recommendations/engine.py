"""
recommendations/engine.py
Objective 3 — Multimodal Product Recommendation

Pipeline:
Image -> ResNet-50 embedding
Text metadata -> TF-IDF embedding
Fusion -> Cosine similarity -> Top-K products
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from database import get_all_products, get_product


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet-50 once
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
image_encoder = nn.Sequential(*list(resnet50.children())[:-1])
image_encoder = image_encoder.to(device)
image_encoder.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_image_embedding(image_path: str) -> np.ndarray:
    """Extract a 2048-dimensional visual embedding using ResNet-50."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = image_encoder(image)

    embedding = embedding.view(embedding.size(0), -1)
    return embedding.cpu().numpy()[0]


def build_text(product: dict) -> str:
    """Build text metadata representation for each product."""
    name = product.get("name", "")
    category = product.get("category", "")
    alt_text = product.get("alt_text", "")
    sentiment_label = product.get("sentiment_label", "")

    return f"{name} {category} {alt_text} {sentiment_label}"


def get_valid_products():
    """Return products that have a valid image_path."""
    products = get_all_products()
    valid_products = []

    for product in products:
        image_path = product.get("image_path")
        if image_path and os.path.exists(image_path):
            valid_products.append(product)

    return valid_products


def build_multimodal_embeddings(products):
    """
    Build fused embeddings:
    visual embedding + text embedding
    """

    # 1. Image embeddings
    image_embeddings = []

    for product in products:
        emb = extract_image_embedding(product["image_path"])
        image_embeddings.append(emb)

    image_embeddings = np.array(image_embeddings)
    image_embeddings = normalize(image_embeddings)

    # 2. Text embeddings
    texts = [build_text(product) for product in products]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=100
    )

    text_embeddings = vectorizer.fit_transform(texts).toarray()
    text_embeddings = normalize(text_embeddings)

    # 3. Fusion
    # alpha controls the importance of image vs text
    alpha = 0.75

    fused_embeddings = np.concatenate([
        alpha * image_embeddings,
        (1 - alpha) * text_embeddings
    ], axis=1)

    fused_embeddings = normalize(fused_embeddings)

    return fused_embeddings


def get_similar_products(product_id: str, top_k: int = 5):
    """
    Return top-K visually and semantically similar products.
    """

    current_product = get_product(product_id)

    if not current_product:
        return {
            "error": "Product not found",
            "status": 404
        }

    products = get_valid_products()

    if len(products) == 0:
        return {
            "product_id": product_id,
            "similar": [],
            "total_found": 0
        }

    product_ids = [p["id"] for p in products]

    if product_id not in product_ids:
        return {
            "error": "Current product image not found",
            "product_id": product_id,
            "status": 404
        }

    embeddings = build_multimodal_embeddings(products)

    query_index = product_ids.index(product_id)
    query_embedding = embeddings[query_index].reshape(1, -1)

    scores = cosine_similarity(query_embedding, embeddings)[0]

    results = []

    for i, product in enumerate(products):
        if product["id"] == product_id:
            continue

        results.append({
            "id": product["id"],
            "name": product["name"],
            "price": product["price"],
            "category": product.get("category", "Unknown"),
            "image_path": product.get("image_path"),
            "score": round(float(scores[i]), 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "product_id": product_id,
        "similar": results[:top_k],
        "total_found": len(results)
    }