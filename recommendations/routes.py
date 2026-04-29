"""
recommendations/routes.py
==========================
Flask blueprint for Objective 3 — Visual product recommendations.

Endpoints:
  GET /api/similar/<product_id> → find visually similar products
"""

from flask import Blueprint, jsonify
from database import get_all_products, get_product

bp = Blueprint("recommendations", __name__)


@bp.route("/similar/<product_id>", methods=["GET"])
def get_similar(product_id: str):
    """
    Expected response:
    {
      "product_id": "prod_001",
      "similar": [
        { "id": "prod_002", "name": "...", "score": 0.91, "price": 249.00 },
        ...
      ]
    }
    """
    # ── TODO: extract embedding for product_id, compute cosine similarity ──
    # For now, return other products with simulated similarity scores
    current_product = get_product(product_id)
    if not current_product:
        return jsonify({"error": "Product not found"}), 404
    
    all_products = get_all_products()
    similar = []
    
    for product in all_products:
        if product["id"] != product_id:
            # Stub: simulate similarity score (in real impl, use ResNet-50 embeddings)
            import random
            random.seed(hash(product_id + product["id"]))  # Deterministic for demo
            score = round(random.uniform(0.65, 0.95), 2)
            
            similar.append({
                "id": product["id"],
                "name": product["name"],
                "price": product["price"],
                "score": score,
                "category": product.get("category", "Unknown")
            })
    
    # Sort by similarity score descending
    similar.sort(key=lambda x: x["score"], reverse=True)
    # ───────────────────────────────────────────────────────────────────────

    return jsonify({
        "product_id": product_id,
        "similar": similar[:5],  # Return top 5
        "total_found": len(similar)
    })
