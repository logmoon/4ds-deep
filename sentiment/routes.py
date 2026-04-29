"""
sentiment/routes.py
====================
Flask blueprint for Objective 2 — Sentiment Analysis.

Endpoints:
  POST /api/sentiment/<id> → analyze reviews of a product + persist result
"""

from flask import Blueprint, request, jsonify
from database import get_product, update_product

bp = Blueprint("sentiment", __name__)

@bp.route("/sentiment", methods=["POST"])
def sentiment():
    """
    Analyze reviews (can be from a product or standalone).
    
    Expected body:
    {
      "reviews": ["Great product!", "Not bad", "Terrible quality"],
      "product_id": "prod_001"  // optional, for persistence
    }

    Response:
    {
      "score": 0.72,
      "label": "positive",
      "breakdown": [
        {"review": "Great product!", "label": "positive", "score": 0.95},
        {"review": "Not bad", "label": "neutral", "score": 0.62},
        {"review": "Terrible quality", "label": "negative", "score": 0.15}
      ],
      "summary": {
        "positive": 1,
        "neutral": 1,
        "negative": 1,
        "total": 3
      }
    }
    """
    data = request.get_json()
    reviews = data.get("reviews", [])
    product_id = data.get("product_id")
    
    if not reviews:
        return jsonify({"error": "reviews array is required"}), 400
    
    # TODO: Replace with BERT/LSTM sentiment model
    # For now, simple keyword-based stub
    breakdown = []
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    
    positive_words = ["great", "amazing", "love", "excellent", "perfect", "best", "incredible", "worth"]
    negative_words = ["terrible", "bad", "worst", "awful", "hate", "poor", "disappointing", "nightmare"]
    
    for review in reviews:
        review_lower = review.lower()
        pos_score = sum(1 for word in positive_words if word in review_lower)
        neg_score = sum(1 for word in negative_words if word in review_lower)
        
        if pos_score > neg_score:
            label = "positive"
            score = min(0.95, 0.65 + (pos_score * 0.1))
            positive_count += 1
        elif neg_score > pos_score:
            label = "negative"
            score = max(0.05, 0.35 - (neg_score * 0.1))
            negative_count += 1
        else:
            label = "neutral"
            score = 0.5
            neutral_count += 1
        
        breakdown.append({
            "review": review,
            "label": label,
            "score": round(score, 2)
        })
    
    # Calculate overall score (weighted average)
    overall_score = sum(item["score"] for item in breakdown) / len(breakdown)
    
    # Determine overall label
    if overall_score >= 0.6:
        overall_label = "positive"
    elif overall_score <= 0.4:
        overall_label = "negative"
    else:
        overall_label = "neutral"
    
    # Persist if product_id provided
    if product_id:
        update_product(product_id, {
            "sentiment_score": round(overall_score, 2),
            "sentiment_label": overall_label,
        })
    
    return jsonify({
        "score": round(overall_score, 2),
        "label": overall_label,
        "breakdown": breakdown,
        "summary": {
            "positive": positive_count,
            "neutral": neutral_count,
            "negative": negative_count,
            "total": len(reviews)
        }
    })
