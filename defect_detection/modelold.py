# defect_detection/model.py
import cv2
import numpy as np
import base64


# ── Thresholds ────────────────────────────────────────────────────────────────
BLUR_THRESHOLD       = 100.0   # Laplacian variance
DARK_THRESHOLD       = 50.0    # Mean pixel brightness (0-255)
BRIGHT_THRESHOLD     = 220.0   # Mean pixel brightness (0-255)
CONTRAST_THRESHOLD   = 40.0    # Std deviation of pixel values
RESOLUTION_MIN       = 100     # Minimum width or height in pixels


def _load_image(image_path=None, image_b64=None):
    """Load image from path or base64 string. Returns BGR numpy array or None."""
    if image_path:
        img = cv2.imread(image_path)
        return img
    if image_b64:
        try:
            decoded = base64.b64decode(image_b64)
            arr     = np.frombuffer(decoded, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None
    return None


def _normalize(value, low, high):
    """Map a raw score to a 0-1 float, clamped."""
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def analyze_image(image_path=None, image_b64=None) -> dict:
    """
    Returns a dict matching the route contract exactly:
    {
        "ok": bool,
        "quality_score": float,
        "details": { "sharpness": float, "brightness": float, "contrast": float },
        "issues": [...],
        "reason": "..."   # only present when ok=False
    }
    """
    img = _load_image(image_path, image_b64)

    if img is None:
        return {
            "ok": False,
            "quality_score": 0.0,
            "details": {"sharpness": 0.0, "brightness": 0.0, "contrast": 0.0},
            "issues": ["Could not read image"],
            "reason": "Image quality does not meet publication standards"
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    issues = []

    # 1 — Resolution
    if w < RESOLUTION_MIN or h < RESOLUTION_MIN:
        issues.append("Image resolution is too low")

    # 2 — Sharpness (Laplacian variance)
    blur_raw      = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness     = round(_normalize(blur_raw, 0, 500), 2)
    if blur_raw < BLUR_THRESHOLD:
        issues.append("Image appears blurry or out of focus")

    # 3 — Brightness (mean pixel value)
    brightness_raw = gray.mean()
    brightness     = round(_normalize(brightness_raw, 0, 255), 2)
    if brightness_raw < DARK_THRESHOLD:
        issues.append("Image is too dark")
    elif brightness_raw > BRIGHT_THRESHOLD:
        issues.append("Image is overexposed")

    # 4 — Contrast (std deviation of pixel values)
    contrast_raw = gray.std()
    contrast     = round(_normalize(contrast_raw, 0, 128), 2)
    if contrast_raw < CONTRAST_THRESHOLD:
        issues.append("Low contrast detected")

    # Overall quality score — weighted average
    quality_score = round(0.5 * sharpness + 0.3 * brightness + 0.2 * contrast, 2)
    ok            = len(issues) == 0 and quality_score >= 0.75

    response = {
        "ok":            ok,
        "quality_score": quality_score,
        "details": {
            "sharpness":  sharpness,
            "brightness": brightness,
            "contrast":   contrast
        },
        "issues": issues
    }

    if not ok:
        response["reason"] = "Image quality does not meet publication standards"

    return response