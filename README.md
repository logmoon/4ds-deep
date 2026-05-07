# SmartShop AI 🛍️

**E-Commerce Intelligent & Inclusif** — 6 AI modules, 1 shared platform.

---

## Project Structure

```
smartshop/
│
├── main.py               ← Integration lead owns this
├── database.py           ← Shared product catalog (READ THIS FIRST)
├── products.json         ← The data store
├── requirements.txt
├── static/
│   └── index.html        ← Frontend shell
│
├── alt_text/             ← Obj 1 (ResNet-50 + Transformer)
├── sentiment/            ← Obj 2 (DistilBERT)
├── recommendations/      ← Obj 3 (ResNet-50 cosine similarity)
├── categorization/       ← Obj 4 (CLIP zero-shot)
├── chatbot/              ← Obj 5 (LLM API)
└── defect_detection/     ← Obj 6 (OpenCV heuristics)
```

---

## Quickstart

```bash
# 1. Clone the repo and navigate to the project
git clone <your-repo-url>
cd smartshop

# 2. Create a shared virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install shared dependencies
pip install -r requirements.txt

# 4. Run the platform
python main.py
# → Open http://localhost:5000
```

---

## The Shared Product Schema

Every module reads/writes using this shape (via `database.py`):

```json
{
  "id": "prod_001",
  "name": "Wireless Headphones",
  "image_path": "static/images/prod_001.jpg",
  "price": 129.99,
  "category": null,        ← Obj 4 writes here
  "alt_text": null,        ← Obj 1 writes here
  "sentiment_score": null, ← Obj 2 writes here
  "sentiment_label": null, ← Obj 2 writes here
  "image_ok": null,        ← Obj 6 writes here
  "reviews": ["..."]
}
```

---

## For Each Module Owner

### Your job is simple:
1. Open your folder (`sentiment/`, `chatbot/`, etc.)
2. Replace the stub in `routes.py` with a real call to your model
3. Put your model logic in `model.py`
4. Test your endpoint with curl or Postman
5. Push to your feature branch (`feature/sentiment`, `feature/chatbot`, etc.)

### Never:
- Touch another module's folder
- Invent a different response format than what's in your `routes.py` docstring
- Add a new product field without telling the integration lead

### Always:
- Import from `database` to read/write products
- Return JSON with exactly the shape documented in your route docstring
- Add your model's pip packages to `requirements.txt`

---

## Testing Your Endpoint

```bash
# Test sentiment (Obj 2)
curl -X POST http://localhost:5000/api/sentiment \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great product!", "Terrible quality.", "It is okay."]}'

# Test sentiment on a catalogue product
curl -X POST http://localhost:5000/api/sentiment/prod_001

# Test alt-text stub
curl -X POST http://localhost:5000/api/alt-text \
  -H "Content-Type: application/json" \
  -d '{"image_path": "static/images/prod_001.jpg"}'

# Health check
curl http://localhost:5000/api/health
```

---

## Git Workflow

```bash
# Each person works on their own branch
git checkout -b feature/sentiment

# Commit your work
git add sentiment/
git commit -m "feat(sentiment): implement DistilBERT pipeline"

# When ready, open a PR to main
# Integration lead reviews and merges
```
