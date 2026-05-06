"""
Objectif 5 — Shopping assistant chatbot  [DEEP LEARNING IMPLEMENTATION]
========================================================================
Architecture:
  - RAG (Retrieval-Augmented Generation) : Sentence-BERT embeddings
  - GPT-2 base : Text generation (no fine-tuning needed)
  
Fallback modes (if DL model not available):
  - Groq API — llama-3.1-8b-instant

Endpoint contract: POST /api/chat
"""

import os
import re
import requests
from flask import Blueprint, request, jsonify
from datetime import datetime, timezone

from database import get_all_products

bp = Blueprint("chatbot", __name__)

# ── Config ───────────────────────────────────────────────────────────────────
USE_DL_MODEL = os.environ.get("USE_DL_MODEL", "true").lower() == "true"
USE_OLLAMA  = os.environ.get("USE_OLLAMA", "true").lower() == "true"

# Ollama (local)
OLLAMA_URL  = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"

# Groq (hosted)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"


# ─── System prompt ───────────────────────────────────────────────────────────

def build_system_prompt(catalogue: list | None, product: dict | None) -> str:
    lines = [
        "You are SmartShop AI, a friendly and knowledgeable shopping assistant.",
        "You help customers find products, answer questions, and make purchase decisions.",
        "Always be concise, helpful, and specific. Reply in the same language the user writes in.",
        "If you recommend products, reference them by their exact name from the catalogue.",
        "Never invent products that are not in the catalogue.",
        "",
    ]

    if product:
        lines += [
            "## Product currently viewed by the customer",
            f"Name     : {product.get('name', 'N/A')}",
            f"Price    : €{product.get('price', 'N/A')}",
            f"Category : {product.get('category') or 'Not yet categorised'}",
            f"Alt-text : {product.get('alt_text') or 'N/A'}",
            f"Sentiment: {product.get('sentiment_label') or 'N/A'} "
            f"(score {product.get('sentiment_score') or 'N/A'})",
            "",
        ]

    if catalogue:
        lines.append("## Full product catalogue")
        for p in catalogue:
            lines.append(
                f"- [{p['id']}] {p['name']} — €{p.get('price', '?')} "
                f"| category: {p.get('category') or 'unknown'}"
            )
        lines.append("")

    lines += [
        "When suggesting products, include their IDs in parentheses like (prod_001).",
        "Keep replies under 120 words unless the customer explicitly asks for more detail.",
    ]

    return "\n".join(lines)


# ─── Ollama (local) ──────────────────────────────────────────────────────────

def call_ollama(system_prompt: str, history: list[dict], message: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        role, content = turn.get("role", ""), turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                  "options": {"temperature": 0.7, "num_predict": 300}},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama n'est pas lancé. Tape : ollama serve")

    if not resp.ok:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")

    try:
        return resp.json()["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Ollama response: {resp.json()}") from exc


# ─── Groq (hosted) ───────────────────────────────────────────────────────────

def call_groq(system_prompt: str, history: list[dict], message: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY non définie. "
            "Ajoute GROQ_API_KEY=gsk_... dans ton .env"
        )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        role, content = turn.get("role", ""), turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":       GROQ_MODEL,
            "messages":    messages,
            "max_tokens":  300,
            "temperature": 0.7,
        },
        timeout=30,
    )

    if not resp.ok:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")

    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Groq response: {resp.json()}") from exc


# ─── Deep Learning Model ─────────────────────────────────────────────────────

_dl_chatbot = None

def get_dl_chatbot():
    """Lazy load du modèle DL (chargé une seule fois)"""
    global _dl_chatbot
    if _dl_chatbot is None:
        try:
            from chatbot.model import get_chatbot_model
            _dl_chatbot = get_chatbot_model()  # Pas besoin de path pour GPT-2 base
            print("[chatbot] ✓ Deep Learning model loaded (RAG + GPT-2)")
        except Exception as e:
            print(f"[chatbot] ⚠ Failed to load DL model: {e}")
            print("[chatbot] → Falling back to API mode")
            return None
    return _dl_chatbot


def call_dl_model(catalogue: list, history: list[dict], message: str) -> str:
    """
    Utilise le modèle Deep Learning (RAG + Fine-tuned Flan-T5).
    
    Architecture:
      1. RAG : Sentence-BERT trouve les produits pertinents
      2. Flan-T5 : Génère la réponse avec contexte
    """
    chatbot = get_dl_chatbot()
    if chatbot is None:
        raise RuntimeError("DL model not available")
    
    # Step 1: RAG - Retrieve relevant products
    relevant_products = chatbot.retrieve_products(
        query=message,
        catalogue=catalogue,
        top_k=3
    )
    
    print(f"[chatbot] RAG retrieved: {[p['name'] for p in relevant_products]}")
    
    # Step 2: Generate response with GPT-2
    response = chatbot.generate_response(
        message=message,
        history=history,
        context_products=relevant_products,
        max_length=60,
        temperature=0.7
    )
    
    return response


# ─── Simple Rule-Based Chatbot (Fallback offline) ───────────────────────────

def call_simple_chatbot(catalogue: list, message: str) -> str:
    """
    Chatbot simple basé sur des règles (fonctionne offline).
    Utilisé quand aucun modèle DL ou API n'est disponible.
    """
    message_lower = message.lower()
    
    # Recherche de mots-clés
    if any(word in message_lower for word in ["headphone", "audio", "sound", "music", "listen"]):
        products = [p for p in catalogue if "headphone" in p["name"].lower()]
        if products:
            p = products[0]
            return f"I recommend the {p['name']} at €{p['price']}. It's perfect for audio needs!"
    
    elif any(word in message_lower for word in ["chair", "office", "sit", "desk", "back"]):
        products = [p for p in catalogue if "chair" in p["name"].lower()]
        if products:
            p = products[0]
            return f"Check out the {p['name']} at €{p['price']}. Great for office comfort!"
    
    elif any(word in message_lower for word in ["clothing", "jacket", "wear", "fashion"]):
        products = [p for p in catalogue if "jacket" in p["name"].lower() or p.get("category") == "clothing"]
        if products:
            p = products[0]
            return f"We have the {p['name']} at €{p['price']}. Stylish and affordable!"
    
    elif any(word in message_lower for word in ["cheap", "affordable", "budget", "inexpensive"]):
        if catalogue:
            cheapest = min(catalogue, key=lambda x: x.get("price", float('inf')))
            return f"Our most affordable item is the {cheapest['name']} at €{cheapest['price']}!"
    
    elif any(word in message_lower for word in ["expensive", "premium", "best", "top"]):
        if catalogue:
            most_expensive = max(catalogue, key=lambda x: x.get("price", 0))
            return f"Our premium item is the {most_expensive['name']} at €{most_expensive['price']}."
    
    # Réponse par défaut
    if catalogue:
        sample_products = catalogue[:3]
        products_list = ", ".join([f"{p['name']} (€{p['price']})" for p in sample_products])
        return f"We have great products available: {products_list}. What are you looking for?"
    
    return "Hello! I'm here to help you find products. What are you looking for today?"


# ─── Router ──────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, history: list[dict], message: str, catalogue: list = None) -> str:
    """
    Route vers le modèle approprié:
      1. Deep Learning (RAG + Flan-T5) si USE_DL_MODEL=true
      2. Ollama (local) si USE_OLLAMA=true
      3. Groq (hosted) si disponible
      4. Simple chatbot (fallback offline)
    """
    if USE_DL_MODEL:
        try:
            print("[chatbot] mode DEEP LEARNING — RAG + GPT-2")
            return call_dl_model(catalogue or [], history, message)
        except Exception as e:
            print(f"[chatbot] DL model failed: {e}")
            print("[chatbot] → Falling back to API mode")
    
    if USE_OLLAMA:
        try:
            print("[chatbot] mode LOCAL — Ollama llama3.1")
            return call_ollama(system_prompt, history, message)
        except Exception as e:
            print(f"[chatbot] Ollama failed: {e}")
            print("[chatbot] → Falling back to simple chatbot")
    
    # Try Groq if available
    if GROQ_API_KEY:
        try:
            print("[chatbot] mode PROD — Groq llama-3.1-8b-instant")
            return call_groq(system_prompt, history, message)
        except Exception as e:
            print(f"[chatbot] Groq failed (no internet?): {e}")
            print("[chatbot] → Falling back to simple chatbot")
    
    # Fallback: Simple rule-based chatbot (works offline)
    print("[chatbot] mode SIMPLE — Rule-based chatbot (offline)")
    return call_simple_chatbot(catalogue or [], message)


def extract_suggested_products(reply: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\(prod_\w+\)", reply)))


# ─── Flask route ─────────────────────────────────────────────────────────────

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
      "product": { "id": "prod_001", "name": "...", ... }   // optional context
    }

    Expected response:
    {
        "reply": "Based on our catalogue, I recommend...",
        "suggested_products": ["prod_001", "prod_002"]       // may be empty
    }
    """
    data    = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []
    product = data.get("product")

    if not message:
        return jsonify({"error": "message is required"}), 400

    catalogue     = get_all_products()
    system_prompt = build_system_prompt(catalogue, product)

    try:
        reply = call_llm(system_prompt, history, message, catalogue)
    except RuntimeError as exc:
        print("ERREUR LLM:", str(exc))
        return jsonify({"error": str(exc)}), 503

    suggested = extract_suggested_products(reply)

    return jsonify({
        "reply":              reply,
        "suggested_products": suggested,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    })