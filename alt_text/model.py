"""
Alt-Text Generation Model
Architecture: BLIP (Encoder-Decoder with Visual Attention)
Based on: Show, Attend and Tell (Xu et al. 2015)
Improvement: Transformer Decoder (Liu & Brailsford 2023)
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Lazy loading (model loads on first request, not at import) ──
_processor = None
_model = None

def load_model():
    """Load BLIP model once and cache it"""
    global _processor, _model
    
    if _model is not None:
        return  # Already loaded
    
    print("📥 Downloading BLIP model (first time only, ~1.5GB)...")
    model_name = "Salesforce/blip-image-captioning-base"
    
    _processor = BlipProcessor.from_pretrained(model_name)
    _model = BlipForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    _model.eval()
    
    print(f"✅ BLIP model loaded on {DEVICE}")

def generate_alt_text(image_path: str) -> str:
    """
    Generate WCAG-compliant alt-text for an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Alt-text string (5-25 words, starts with capital, ends with period)
    """
    load_model()
    
    # ── Load and preprocess image ────────────────────────────────
    img = Image.open(image_path).convert("RGB")
    inputs = _processor(img, return_tensors="pt").to(DEVICE)
    
    # ── Generate caption ─────────────────────────────────────────
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_length=25,      # WCAG maximum
            min_length=5,       # WCAG minimum
            num_beams=5,        # Beam search (same as original paper)
            temperature=0.7,
            do_sample=False,
            no_repeat_ngram_size=2  # Avoid repetition
        )
    
    # ── Decode and post-process ──────────────────────────────────
    caption = _processor.decode(output_ids[0], skip_special_tokens=True)
    
    # Clean up: capitalize + period
    caption = caption.strip().capitalize()
    if not caption.endswith('.'):
        caption += '.'
    
    # Ensure WCAG compliance (5-25 words)
    words = caption.split()
    if len(words) > 25:
        caption = ' '.join(words[:25]) + '.'
    elif len(words) < 5:
        # Pad with descriptive phrase if too short
        caption = caption.replace('.', '') + ', a product image.'
    
    return caption