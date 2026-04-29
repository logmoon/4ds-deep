"""
Objectif 6 — Defective image detection  [STUB]
================================================
Replace stub with OpenCV heuristics or a trained binary classifier.
Endpoint contract: POST /api/check-image
"""

from flask import Blueprint, request, jsonify

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
    image_path = data.get("image_path")
    image_b64 = data.get("image_b64")

    if not image_path and not image_b64:
        return jsonify({"error": "image_path or image_b64 is required"}), 400

    # ── TODO: compute Laplacian variance for blur, check brightness, etc. ───
    # import cv2, numpy as np
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    # if blur_score < 100: return {"ok": False, "reason": f"Too blurry ({blur_score:.1f})"}
    
    # Stub: simulate quality checks with random but realistic values
    import random
    if image_path:
        seed_val = hash(image_path)
    else:
        seed_val = hash(image_b64[:100] if image_b64 else "default")
    
    random.seed(seed_val)
    
    # Generate quality metrics
    sharpness = round(random.uniform(0.70, 0.98), 2)
    brightness = round(random.uniform(0.65, 0.95), 2)
    contrast = round(random.uniform(0.68, 0.96), 2)
    
    quality_score = round((sharpness + brightness + contrast) / 3, 2)
    
    issues = []
    if sharpness < 0.75:
        issues.append("Image appears blurry or out of focus")
    if brightness < 0.70:
        issues.append("Image is too dark")
    elif brightness > 0.92:
        issues.append("Image is overexposed")
    if contrast < 0.72:
        issues.append("Low contrast detected")
    
    ok = quality_score >= 0.75 and len(issues) == 0
    
    response = {
        "ok": ok,
        "quality_score": quality_score,
        "details": {
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast
        }
    }
    
    if not ok:
        response["issues"] = issues
        response["reason"] = "Image quality does not meet publication standards"
    else:
        response["issues"] = []
    
    # ────────────────────────────────────────────────────────────────────────

    return jsonify(response)
