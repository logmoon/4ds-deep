"""
recommendations/routes.py
==========================
Flask blueprint for Objective 3 — Multimodal product recommendations.

Endpoint:
  GET /api/similar/<product_id> → find similar products
"""

from flask import Blueprint, jsonify
from recommendations.engine import get_similar_products

bp = Blueprint("recommendations", __name__)


@bp.route("/similar/<product_id>", methods=["GET"])
def get_similar(product_id: str):
    result = get_similar_products(product_id, top_k=5)

    if "error" in result:
        return jsonify(result), result.get("status", 500)

    return jsonify(result)