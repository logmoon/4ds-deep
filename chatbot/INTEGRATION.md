# 🔌 Intégration — Ce qui change dans routes.py

## 📋 Résumé rapide

**AVANT (API) :**
```python
call_llm() → call_ollama() ou call_groq() → API externe
```

**APRÈS (Deep Learning) :**
```python
call_llm() → call_dl_model() → RAG + GPT-2 local
```

---

## 🔄 Flux d'exécution détaillé

### Quand une requête arrive sur `/api/chat`

```python
@bp.route("/chat", methods=["POST"])
def chat():
    # 1. Récupérer les données
    message = "I need headphones under 150€"
    history = []
    product = None
    
    # 2. Charger le catalogue
    catalogue = get_all_products()  # Tous les produits
    
    # 3. Construire le system prompt (pour fallback API)
    system_prompt = build_system_prompt(catalogue, product)
    
    # 4. APPEL DU MODÈLE DL ⭐
    reply = call_llm(system_prompt, history, message, catalogue)
    
    # 5. Extraire les produits suggérés
    suggested = extract_suggested_products(reply)
    
    # 6. Retourner la réponse
    return jsonify({
        "reply": reply,
        "suggested_products": suggested
    })
```

---

## ⭐ La fonction clé : `call_llm()`

### Nouvelle implémentation

```python
def call_llm(system_prompt: str, history: list[dict], message: str, catalogue: list = None) -> str:
    """
    Route vers le modèle approprié selon la config.
    """
    # PRIORITÉ 1 : Deep Learning Model
    if USE_DL_MODEL:
        try:
            print("[chatbot] mode DEEP LEARNING — RAG + Fine-tuned GPT-2")
            return call_dl_model(catalogue or [], history, message)
        except Exception as e:
            print(f"[chatbot] DL model failed: {e}")
            print("[chatbot] → Falling back to API mode")
    
    # PRIORITÉ 2 : Ollama (local)
    if USE_OLLAMA:
        print("[chatbot] mode LOCAL — Ollama llama3.1")
        return call_ollama(system_prompt, history, message)
    
    # PRIORITÉ 3 : Groq (hosted)
    else:
        print("[chatbot] mode PROD — Groq llama-3.1-8b-instant")
        return call_groq(system_prompt, history, message)
```

### Configuration

Dans `.env` :
```bash
USE_DL_MODEL=true   # Active le modèle DL
USE_OLLAMA=false    # Fallback si DL échoue
GROQ_API_KEY=...    # Fallback si Ollama échoue
```

---

## 🧠 La fonction Deep Learning : `call_dl_model()`

### Code complet

```python
def call_dl_model(catalogue: list, history: list[dict], message: str) -> str:
    """
    Utilise le modèle Deep Learning (RAG + Fine-tuned GPT-2).
    
    Architecture:
      1. RAG : Sentence-BERT trouve les produits pertinents
      2. GPT-2 : Génère la réponse avec contexte
    """
    # Charger le modèle (une seule fois, singleton)
    chatbot = get_dl_chatbot()
    if chatbot is None:
        raise RuntimeError("DL model not available")
    
    # ── STEP 1: RAG (Retrieval) ──────────────────────────────────────
    relevant_products = chatbot.retrieve_products(
        query=message,
        catalogue=catalogue,
        top_k=3
    )
    
    print(f"[chatbot] RAG retrieved: {[p['name'] for p in relevant_products]}")
    
    # ── STEP 2: Generation ───────────────────────────────────────────
    response = chatbot.generate_response(
        message=message,
        history=history,
        context_products=relevant_products,
        max_length=150,
        temperature=0.7
    )
    
    return response
```

### Ce qui se passe en détail

#### STEP 1 : RAG (Retrieval)

```python
relevant_products = chatbot.retrieve_products(
    query="I need headphones under 150€",
    catalogue=[
        {"id": "prod_001", "name": "Wireless Headphones", "price": 129.99, ...},
        {"id": "prod_002", "name": "Office Chair", "price": 249.00, ...},
        {"id": "prod_003", "name": "Denim Jacket", "price": 59.99, ...}
    ],
    top_k=3
)
```

**Dans `model.py` → `retrieve_products()` :**

1. **Encoder la requête**
   ```python
   query_embedding = self.embedder.encode("I need headphones under 150€")
   # → [0.234, -0.456, 0.678, ..., 0.123]  (384 dimensions)
   ```

2. **Encoder les produits**
   ```python
   product_texts = [
       "Wireless Headphones electronics",
       "Office Chair furniture",
       "Denim Jacket clothing"
   ]
   product_embeddings = self.embedder.encode(product_texts)
   # → [[0.221, ...], [0.012, ...], [0.445, ...]]
   ```

3. **Calculer la similarité**
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   similarities = cosine_similarity([query_embedding], product_embeddings)[0]
   # → [0.92, 0.34, 0.21]
   ```

4. **Sélectionner Top-K**
   ```python
   top_indices = np.argsort(similarities)[-3:][::-1]
   # → [0, 1, 2]  (indices triés par score)
   
   relevant_products = [catalogue[i] for i in top_indices]
   # → [
   #     {"id": "prod_001", "name": "Wireless Headphones", "similarity_score": 0.92},
   #     {"id": "prod_002", "name": "Office Chair", "similarity_score": 0.34},
   #     {"id": "prod_003", "name": "Denim Jacket", "similarity_score": 0.21}
   # ]
   ```

**Résultat :** Les 3 produits les plus pertinents

---

#### STEP 2 : Generation

```python
response = chatbot.generate_response(
    message="I need headphones under 150€",
    history=[],
    context_products=[
        {"id": "prod_001", "name": "Wireless Headphones", "price": 129.99},
        {"id": "prod_002", "name": "Office Chair", "price": 249.00},
        {"id": "prod_003", "name": "Denim Jacket", "price": 59.99}
    ],
    max_length=150,
    temperature=0.7
)
```

**Dans `model.py` → `generate_response()` :**

1. **Construire le prompt**
   ```python
   prompt = self._build_prompt(message, history, context_products)
   # Résultat :
   # """
   # You are SmartShop AI, a helpful shopping assistant.
   # 
   # Available products:
   # - Wireless Headphones (€129.99) - electronics
   # - Office Chair (€249.00) - furniture
   # - Denim Jacket (€59.99) - clothing
   # 
   # Customer: I need headphones under 150€
   # Assistant:
   # """
   ```

2. **Tokenizer**
   ```python
   inputs = self.tokenizer(prompt, return_tensors="pt")
   # → {"input_ids": tensor([[16, 1990, 3186, 25, ...]])}
   ```

3. **Générer avec GPT-2**
   ```python
   outputs = self.model.generate(
       inputs.input_ids,
       max_length=inputs.input_ids.shape[1] + 150,
       temperature=0.7,
       do_sample=True,
       top_p=0.9,
       pad_token_id=self.tokenizer.eos_token_id
   )
   # → tensor([[16, 1990, 3186, 25, ..., 314, 4313, 262, ...]])
   ```

4. **Décoder**
   ```python
   full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
   # → "You are SmartShop AI... Customer: I need headphones... Assistant: I recommend the Wireless Headphones at €129.99..."
   
   response = full_response[len(prompt):].strip()
   # → "I recommend the Wireless Headphones at €129.99. They offer excellent sound quality and battery life."
   ```

5. **Nettoyer**
   ```python
   response = self._clean_response(response)
   # → Enlever les artefacts, tronquer si trop long
   ```

**Résultat :** La réponse générée

---

## 🔍 Exemple complet d'exécution

### Requête

```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need headphones under 150€",
    "history": []
  }'
```

### Logs du serveur

```
[chatbot] mode DEEP LEARNING — RAG + Fine-tuned GPT-2
[ChatbotModel] Loading models...
  ✓ Embeddings model loaded: sentence-transformers/all-MiniLM-L6-v2
  ✓ Loading fine-tuned GPT-2 from: chatbot/models/gpt2-finetuned
[ChatbotModel] ✓ All models loaded successfully!

[chatbot] ✓ Deep Learning model loaded (RAG + Fine-tuned GPT-2)
[chatbot] RAG retrieved: ['Wireless Noise-Cancelling Headphones', 'Ergonomic Office Chair', 'Slim Fit Denim Jacket']
```

### Réponse

```json
{
  "reply": "I recommend the Wireless Noise-Cancelling Headphones at €129.99. They offer excellent sound quality and battery life.",
  "suggested_products": ["(prod_001)"],
  "timestamp": "2024-05-05T10:30:00.000Z"
}
```

---

## 📊 Comparaison des modes

### Mode 1 : Deep Learning (USE_DL_MODEL=true)

```
User Query
    ↓
call_llm()
    ↓
call_dl_model()
    ↓
get_dl_chatbot() → Charge le modèle (singleton)
    ↓
chatbot.retrieve_products() → RAG avec Sentence-BERT
    ↓
chatbot.generate_response() → Génération avec GPT-2
    ↓
Response
```

**Avantages :**
- ✅ Deep Learning réel
- ✅ Contrôle total
- ✅ Offline
- ✅ Gratuit

### Mode 2 : Ollama (USE_DL_MODEL=false, USE_OLLAMA=true)

```
User Query
    ↓
call_llm()
    ↓
call_ollama()
    ↓
HTTP POST → http://localhost:11434/api/chat
    ↓
Response
```

**Avantages :**
- ✅ Qualité élevée
- ✅ Local
- ⚠️ Pas d'entraînement

### Mode 3 : Groq (USE_DL_MODEL=false, USE_OLLAMA=false)

```
User Query
    ↓
call_llm()
    ↓
call_groq()
    ↓
HTTP POST → https://api.groq.com/openai/v1/chat/completions
    ↓
Response
```

**Avantages :**
- ✅ Qualité élevée
- ✅ Rapide
- ⚠️ Payant
- ⚠️ Pas d'entraînement

---

## 🎯 Ce qui est appelé après Option 1 + 2

### Résumé final

Après avoir fait **Option 1 (Fine-tuning GPT-2)** + **Option 2 (RAG)**, voici ce qui est appelé dans `routes.py` :

```python
# 1. Requête arrive
@bp.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    history = request.json.get("history", [])
    catalogue = get_all_products()
    
    # 2. Appel du modèle DL
    reply = call_llm(system_prompt, history, message, catalogue)
    #         ↓
    #    call_dl_model(catalogue, history, message)
    #         ↓
    #    get_dl_chatbot()  → Charge le modèle (une fois)
    #         ↓
    #    chatbot.retrieve_products(message, catalogue, top_k=3)
    #         ↓ RAG : Sentence-BERT + Cosine Similarity
    #    relevant_products = [prod_001, prod_002, prod_003]
    #         ↓
    #    chatbot.generate_response(message, history, relevant_products)
    #         ↓ GPT-2 : Transformer + Sampling
    #    response = "I recommend the Wireless Headphones..."
    
    # 3. Retour de la réponse
    return jsonify({"reply": reply, ...})
```

### En une phrase

**`routes.py` appelle `call_dl_model()` qui utilise `model.py` pour faire du RAG (Sentence-BERT) puis de la génération (GPT-2 fine-tuné).**

---

## ✅ Checklist d'intégration

- [x] `model.py` créé avec `ChatbotModel`
- [x] `routes.py` modifié avec `call_dl_model()`
- [x] Configuration `USE_DL_MODEL` ajoutée
- [x] Singleton `get_dl_chatbot()` implémenté
- [x] Fallback vers API si DL échoue
- [x] Logs informatifs ajoutés
- [x] Documentation complète

**Tout est prêt ! Il ne reste plus qu'à entraîner le modèle avec le notebook.** 🚀
