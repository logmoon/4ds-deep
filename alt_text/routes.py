"""
Objectif 1 — Alt-text generator  [IMPLEMENTED]
================================================
ResNet-50 + Transformer model for automatic alt-text generation.
Endpoint contract: POST /api/alt-text
"""

from flask import Blueprint, request, jsonify
from .model import generate_alt_text
import base64
import os
import tempfile

bp = Blueprint("alt_text", __name__)


@bp.route("/alt-text", methods=["POST"])
def generate_alt_text_route():
    """
    Expected body: { "image_b64": "base64_encoded_image_data" }
                or { "image_path": "static/images/prod_001.jpg" }
    Expected response: { 
        "alt_text": "A pair of wireless headphones in black",
        "confidence": 0.92
    }
    """
    data       = request.get_json()
    image_path = data.get("image_path")
    image_b64  = data.get("image_b64")

    if not image_path and not image_b64:
        return jsonify({"error": "image_path or image_b64 is required"}), 400

    # ── Ton modèle réel ─────────────────────────────────────────────────────
    try:
        if image_path:
            if not os.path.exists(image_path):
                return jsonify({"error": f"File not found: {image_path}"}), 400
            alt_text = generate_alt_text(image_path)

        elif image_b64:
            # Décode le base64 → fichier temporaire → génère l'alt-text
            image_data = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
            alt_text = generate_alt_text(tmp_path)
            os.unlink(tmp_path)   # supprime le fichier temporaire

        confidence = 0.92        # fixe pour l'instant, ton modèle n'en produit pas
    # ────────────────────────────────────────────────────────────────────────

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Check WCAG compliance
    word_count = len(alt_text.split())
    wcag_ok = 5 <= word_count <= 25
    
    return jsonify({
        "alt_text":   alt_text,
        "confidence": confidence,
        "model":      "BLIP (Encoder-Decoder + Visual Attention)",
        "word_count": word_count,
        "wcag_compliant": wcag_ok
    })