"""
Objectif 5 — Shopping assistant chatbot  [IMPLEMENTED]
=======================================================
Hybrid mode:
  - Local (dev)   : Ollama — llama3.1  (USE_OLLAMA=true)
  - Hosted (prod) : Groq API — llama-3.1-8b-instant (USE_OLLAMA=false)

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


# ─── Router ──────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, history: list[dict], message: str) -> str:
    """Route vers Ollama (local) ou Groq (hébergé) selon USE_OLLAMA."""
    if USE_OLLAMA:
        print("[chatbot] mode LOCAL — Ollama llama3.1")
        return call_ollama(system_prompt, history, message)
    else:
        print("[chatbot] mode PROD — Groq llama-3.1-8b-instant")
        return call_groq(system_prompt, history, message)


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
        reply = call_llm(system_prompt, history, message)
    except RuntimeError as exc:
        print("ERREUR LLM:", str(exc))
        return jsonify({"error": str(exc)}), 503

    suggested = extract_suggested_products(reply)

    return jsonify({
        "reply":              reply,
        "suggested_products": suggested,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    })