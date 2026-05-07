"""
Objectif 4 — Automatic product categorization
==============================================
Endpoint: POST /api/categorize
Body:     { "image_b64": "<base64>" }  OR  { "image_path": "static/images/x.jpg" }
Response: { "category": "...", "confidence": 0.95, "top3": [...] }
"""

import base64
import io
import os

from flask import Blueprint, request, jsonify
from PIL import Image

from categorization.model import predict

bp = Blueprint("categorization", __name__)


def _image_from_request(data: dict) -> Image.Image | None:
    if data.get("image_b64"):
        raw = base64.b64decode(data["image_b64"])
        return Image.open(io.BytesIO(raw))
    if data.get("image_path"):
        path = data["image_path"]
        if not os.path.exists(path):
            return None
        return Image.open(path)
    return None


@bp.route("/categorize", methods=["POST"])
def categorize():
    data = request.get_json(silent=True) or {}

    if not data.get("image_b64") and not data.get("image_path"):
        return jsonify({"error": "image_path or image_b64 is required"}), 400

    try:
        image = _image_from_request(data)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400

    if image is None:
        return jsonify({"error": "image_path not found or image_b64 invalid"}), 400

    try:
        result = predict(image)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    return jsonify(result)
