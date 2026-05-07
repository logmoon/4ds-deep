"""
app.py — SmartShop AI entry point
====================================
The integration lead owns this file.
Add a new blueprint here when a module is ready.
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from database import get_all_products, add_product, get_product, update_product
import uuid
import base64
import os
from dotenv import load_dotenv

load_dotenv()

# ── Import blueprints (swap stubs for real ones as modules are completed) ──
from alt_text.routes       import bp as alt_text_bp
from sentiment.routes      import bp as sentiment_bp
from recommendations.routes import bp as recommendations_bp
from categorization.routes  import bp as categorization_bp
from chatbot.routes         import bp as chatbot_bp
from defect_detection.routes import bp as defect_bp

app = Flask(__name__, static_folder="static")
CORS(app)  # allow the frontend to call the API

# ── Register blueprints ────────────────────────────────────────────────────
app.register_blueprint(alt_text_bp,        url_prefix="/api")
app.register_blueprint(sentiment_bp,       url_prefix="/api")
app.register_blueprint(recommendations_bp, url_prefix="/api")
app.register_blueprint(categorization_bp,  url_prefix="/api")
app.register_blueprint(chatbot_bp,         url_prefix="/api")
app.register_blueprint(defect_bp,          url_prefix="/api")

# ── Healthcheck ────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "platform": "SmartShop AI"})

# ── Product catalog endpoints ──────────────────────────────────────────────
@app.route("/api/products")
def get_products():
    """Return all products from the catalog"""
    products = get_all_products()
    return jsonify({"products": products, "count": len(products)})

@app.route("/api/products", methods=["POST"])
def create_product():
    """
    Create a new product with AI-generated metadata
    Expected body:
    {
      "name": "Product Name",
      "price": 99.99,
      "image_b64": "base64_encoded_image",
      "alt_text": "...",      // from AI
      "category": "...",       // from AI
      "image_ok": true         // from AI
    }
    """
    data = request.get_json()
    
    # Validate required fields
    if not data.get("name") or not data.get("price"):
        return jsonify({"error": "name and price are required"}), 400
    
    # Generate unique product ID
    product_id = f"prod_{uuid.uuid4().hex[:8]}"
    
    # Save image if provided
    image_path = None
    if data.get("image_b64"):
        image_path = f"static/images/{product_id}.jpg"
        os.makedirs("static/images", exist_ok=True)
        image_data = base64.b64decode(data["image_b64"])
        with open(image_path, "wb") as f:
            f.write(image_data)
    
    # Create product object
    product = {
        "id": product_id,
        "name": data["name"],
        "price": float(data["price"]),
        "description": data.get("description", ""),
        "seller": data.get("seller", "Anonymous"),
        "seller_id": data.get("seller_id", ""),
        "image_path": image_path or "static/images/placeholder.jpg",
        "category": data.get("category"),
        "alt_text": data.get("alt_text"),
        "image_ok": data.get("image_ok"),
        "sentiment_score": None,
        "sentiment_label": None,
        "reviews": []
    }
    
    # Add to database
    add_product(product)
    
    return jsonify({
        "success": True,
        "product": product,
        "message": "Product created successfully"
    }), 201

@app.route("/api/products/<product_id>/reviews", methods=["POST"])
def add_review(product_id):
    """
    Add a review to a product
    Expected body: { "review": "Great product!" }
    """
    data = request.get_json()
    review_text = data.get("review", "").strip()
    
    if not review_text:
        return jsonify({"error": "review text is required"}), 400
    
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "Product not found"}), 404
    
    # Add review to product
    if "reviews" not in product:
        product["reviews"] = []
    
    product["reviews"].append(review_text)
    
    # Update product in database
    update_product(product_id, {"reviews": product["reviews"]})
    
    return jsonify({
        "success": True,
        "review": review_text,
        "total_reviews": len(product["reviews"])
    }), 201

# ── Serve the frontend ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)

