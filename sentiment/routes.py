from flask import Blueprint, request, jsonify
from database import get_product, update_product
from sentiment.inference import analyze_reviews

bp = Blueprint("sentiment", __name__)

@bp.route("/sentiment", methods=["POST"])
def sentiment():
    data       = request.get_json()
    reviews    = data.get("reviews", [])
    product_id = data.get("product_id")

    if not reviews:
        return jsonify({"error": "reviews array is required"}), 400

    result = analyze_reviews(reviews)

    if product_id:
        update_product(product_id, {
            "sentiment_score": result["overall_score"],
            "sentiment_label": result["overall_label"],
        })

    return jsonify(result)
