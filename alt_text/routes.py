"""
Objectif 1 — Alt-text generator  [STUB]
========================================
Replace the stub response with your ResNet-50 + Transformer model.
Endpoint contract: POST /api/alt-text
"""

from flask import Blueprint, request, jsonify

bp = Blueprint("alt_text", __name__)


@bp.route("/alt-text", methods=["POST"])
def generate_alt_text():
    """
    Expected body: { "image_b64": "base64_encoded_image_data" }
                or { "image_path": "static/images/prod_001.jpg" }
    Expected response: { 
        "alt_text": "A pair of wireless headphones in black",
        "confidence": 0.92
    }
    """
    data = request.get_json()
    image_path = data.get("image_path")
    image_b64 = data.get("image_b64")

    if not image_path and not image_b64:
        return jsonify({"error": "image_path or image_b64 is required"}), 400

    # ── TODO: replace stub with your ResNet-50 + Transformer model ──────────
    # For now, generate realistic stub based on common product types
    if image_path:
        alt_text = f"Product image showing item from {image_path}"
    else:
        alt_text = "High-quality product photograph with clear details and professional lighting"
    
    confidence = 0.89
    # ────────────────────────────────────────────────────────────────────────

    return jsonify({
        "alt_text": alt_text,
        "confidence": confidence,
        "model": "ResNet-50 + Transformer"
    })
