"""
Objectif 6 — Defective image detection
================================================
Real OpenCV heuristics replacing the stub.
Endpoint contract: POST /api/check-image
"""

from flask import Blueprint, request, jsonify
from .model import analyze_image

bp = Blueprint("defect_detection", __name__)


@bp.route("/check-image", methods=["POST"])
def check_image():
    """
    Expected body: { "image_b64": "base64_data" }
                or { "image_path": "static/images/prod_001.jpg" }
    Expected response:
    {
      "ok": true,
      "quality_score": 0.87,
      "details": {
        "sharpness": 0.92,
        "brightness": 0.85,
        "contrast": 0.84
      },
      "issues": []
    }
    — or —
    {
      "ok": false,
      "quality_score": 0.34,
      "details": {
        "sharpness": 0.23,
        "brightness": 0.45,
        "contrast": 0.34
      },
      "issues": ["Image is too blurry", "Low contrast detected"],
      "reason": "Image quality does not meet publication standards"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    image_path = data.get("image_path")
    image_b64  = data.get("image_b64")

    if not image_path and not image_b64:
        return jsonify({"error": "image_path or image_b64 is required"}), 400

    result = analyze_image(image_path=image_path, image_b64=image_b64)
    return jsonify(result)