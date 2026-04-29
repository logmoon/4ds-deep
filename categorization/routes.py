"""
Objectif 4 — Automatic product categorization  [STUB]
=======================================================
Replace stub with CLIP zero-shot classification.
Endpoint contract: POST /api/categorize
"""

from flask import Blueprint, request, jsonify

bp = Blueprint("categorization", __name__)

CATEGORIES = ["electronics", "furniture", "clothing", "food", "sports", "books", "toys", "beauty"]


@bp.route("/categorize", methods=["POST"])
def categorize():
    """
    Expected body: { "image_b64": "base64_data" }
                or { "image_path": "static/images/prod_001.jpg" }
    Expected response: { 
        "category": "electronics", 
        "confidence": 0.94,
        "top_categories": [
            {"label": "electronics", "score": 0.94},
            {"label": "accessories", "score": 0.78},
            {"label": "audio", "score": 0.65}
        ]
    }
    """
    data = request.get_json()
    image_path = data.get("image_path")
    image_b64 = data.get("image_b64")

    if not image_path and not image_b64:
        return jsonify({"error": "image_path or image_b64 is required"}), 400

    # ── TODO: load CLIP, encode image + category labels, return argmax ──────
    # import clip, torch
    # model, preprocess = clip.load("ViT-B/32")
    # ...
    
    # Stub: return realistic category distribution
    import random
    if image_path:
        seed_val = hash(image_path)
    else:
        seed_val = hash(image_b64[:100] if image_b64 else "default")
    
    random.seed(seed_val)
    
    # Pick a random primary category
    primary_idx = random.randint(0, len(CATEGORIES) - 1)
    primary_category = CATEGORIES[primary_idx]
    primary_confidence = round(random.uniform(0.85, 0.98), 2)
    
    # Generate top 3 categories
    top_categories = [{"label": primary_category, "score": primary_confidence}]
    
    remaining = [c for i, c in enumerate(CATEGORIES) if i != primary_idx]
    random.shuffle(remaining)
    
    for cat in remaining[:2]:
        score = round(random.uniform(0.45, 0.75), 2)
        top_categories.append({"label": cat, "score": score})
    
    top_categories.sort(key=lambda x: x["score"], reverse=True)
    # ────────────────────────────────────────────────────────────────────────

    return jsonify({
        "category": top_categories[0]["label"],
        "confidence": top_categories[0]["score"],
        "top_categories": top_categories,
        "all_categories": CATEGORIES
    })
