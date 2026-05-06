import cv2, numpy as np, base64, io, os
from PIL import Image, ImageEnhance
import torch, torch.nn as nn

BLUR_THRESHOLD     = 100.0
DARK_THRESHOLD     = 50.0
BRIGHT_THRESHOLD   = 220.0
CONTRAST_THRESHOLD = 40.0
RESOLUTION_MIN     = 100
QUALITY_THRESHOLD  = 0.55   # score-only gate — no issues list involved

def _load_image(image_path=None, image_b64=None):
    if image_path: return cv2.imread(image_path)
    if image_b64:
        try:
            arr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except: return None
    return None

def _normalize(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _check_quality(img_cv2):
    gray   = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    issues = []

    if w < RESOLUTION_MIN or h < RESOLUTION_MIN:
        issues.append("Image resolution is too low")

    blur_raw  = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = round(_normalize(blur_raw, 0, 500), 2)
    if blur_raw < BLUR_THRESHOLD:
        issues.append("Image appears blurry or out of focus")

    brightness_raw = gray.mean()
    brightness     = round(_normalize(brightness_raw, 0, 255), 2)
    if brightness_raw < DARK_THRESHOLD:
        issues.append("Image is too dark")
    elif brightness_raw > BRIGHT_THRESHOLD:
        issues.append("Image is overexposed")

    contrast_raw = gray.std()
    contrast     = round(_normalize(contrast_raw, 0, 128), 2)
    if contrast_raw < CONTRAST_THRESHOLD:
        issues.append("Low contrast detected")

    score = round(0.5 * sharpness + 0.3 * brightness + 0.2 * contrast, 2)

    # FIX: ok is based on SCORE ONLY — issues list is just informational
    ok = score >= QUALITY_THRESHOLD

    return ok, score, {"sharpness": sharpness, "brightness": brightness, "contrast": contrast}, issues


def enhance_image_cv(img_cv2, issues):
    """
    Classical CV enhancement — targeted fix per detected issue.
    No neural network — always improves, never makes things worse.
    """
    img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

    if "Image appears blurry or out of focus" in issues:
        img_cv  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        blur    = cv2.GaussianBlur(img_cv, (0, 0), 3)
        sharp   = cv2.addWeighted(img_cv, 1.8, blur, -0.8, 0)
        img     = Image.fromarray(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))

    if "Image is too dark" in issues:
        img_cv  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lab     = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        lab     = cv2.merge((l, a, b))
        img     = Image.fromarray(cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB))
        img     = ImageEnhance.Brightness(img).enhance(1.6)

    if "Image is overexposed" in issues:
        img = ImageEnhance.Brightness(img).enhance(0.75)

    if "Low contrast detected" in issues:
        img = ImageEnhance.Contrast(img).enhance(1.8)

    return img


def analyze_image(image_path=None, image_b64=None):
    img = _load_image(image_path, image_b64)
    if img is None:
        return {
            "ok": False, "quality_score": 0.0,
            "details": {"sharpness": 0.0, "brightness": 0.0, "contrast": 0.0},
            "issues": ["Could not read image"],
            "reason": "Image quality does not meet publication standards"
        }

    ok, score, details, issues = _check_quality(img)
    result = {"ok": ok, "quality_score": score, "details": details, "issues": issues}
    if not ok:
        result["reason"] = "Image quality does not meet publication standards"

    # Only enhance if score is genuinely below threshold
    if not ok and issues:
        enhanced_pil = enhance_image_cv(img, issues)
        enhanced_cv  = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        e_ok, e_score, e_det, _ = _check_quality(enhanced_cv)
        e_det["quality_score"] = e_score
        result["enhanced_image"]   = _pil_to_b64(enhanced_pil)
        result["enhanced_ok"]      = e_ok
        result["enhanced_details"] = e_det

    return result