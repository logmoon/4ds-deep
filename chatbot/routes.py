"""
Objectif 5 — Shopping assistant chatbot  [STUB]
================================================
Replace stub with a real LLM call (Gemini / HuggingFace).
Endpoint contract: POST /api/chat
"""

from flask import Blueprint, request, jsonify

bp = Blueprint("chatbot", __name__)


@bp.route("/chat", methods=["POST"])
def chat():
    """
    Expected body:
    {
      "message": "I need headphones under 150€",
      "history": [
        { "role": "user",      "content": "Hello" },
        { "role": "assistant", "content": "Hi! How can I help?" }
      ],
      "product": { "id": "prod_001", "name": "...", ... }  // optional context
    }
    Expected response: { 
        "reply": "Based on our catalogue, I recommend...",
        "suggested_products": ["prod_001", "prod_002"]  // optional
    }
    """
    data    = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])
    product = data.get("product")

    if not message:
        return jsonify({"error": "message is required"}), 400

    # ── TODO: build prompt with catalogue context, call LLM API ─────────────
    # from database import get_all_products
    # catalogue = get_all_products()
    # system_prompt = f"You are a shopping assistant. Catalogue: {catalogue}"
    # reply = call_gemini(system_prompt, history, message)
    
    # Stub: Generate contextual responses based on keywords
    message_lower = message.lower()
    
    if product:
        product_name = product.get("name", "this product")
        product_price = product.get("price", "N/A")
        
        if any(word in message_lower for word in ["price", "cost", "expensive", "cheap"]):
            reply = f"The {product_name} is priced at €{product_price}. This is competitive for products in this category. Would you like to know about similar alternatives?"
        elif any(word in message_lower for word in ["feature", "spec", "detail", "about"]):
            reply = f"The {product_name} is one of our popular items. It offers excellent quality and value. Would you like me to find similar products or answer specific questions about its features?"
        elif any(word in message_lower for word in ["shipping", "delivery", "ship"]):
            reply = f"We offer free shipping on orders over €50. The {product_name} qualifies for standard delivery (3-5 business days) or express shipping (1-2 days) for an additional fee."
        elif any(word in message_lower for word in ["return", "refund", "warranty"]):
            reply = f"All our products, including the {product_name}, come with a 30-day return policy and manufacturer's warranty. You can return it for a full refund if you're not satisfied."
        else:
            reply = f"I'd be happy to help you with the {product_name}! Could you tell me more about what you'd like to know? I can provide information about features, pricing, shipping, or suggest similar products."
    else:
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            reply = "Hello! Welcome to SmartShop AI. I'm here to help you find the perfect product. What are you looking for today?"
        elif any(word in message_lower for word in ["recommend", "suggest", "looking for"]):
            reply = "I'd love to help you find the right product! Could you tell me more about what you're looking for? For example, what category interests you (electronics, furniture, clothing, etc.) and what's your budget?"
        elif any(word in message_lower for word in ["headphone", "audio", "music"]):
            reply = "We have excellent audio products! I'd recommend checking out our Wireless Noise-Cancelling Headphones - they offer amazing sound quality and battery life. Would you like to see similar products?"
        else:
            reply = f"I understand you're asking about: '{message}'. I'm here to help you find the perfect product from our catalogue. Could you provide more details about what you're looking for?"
    
    # ────────────────────────────────────────────────────────────────────────

    return jsonify({
        "reply": reply,
        "timestamp": "2024-01-01T12:00:00Z"
    })
