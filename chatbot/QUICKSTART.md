# 🚀 Quickstart — Chatbot Deep Learning

## En 3 étapes simples

### 1️⃣ Entraîner le modèle

Ouvre le notebook Jupyter :

```bash
# Installer Jupyter si nécessaire
pip install jupyter

# Lancer le notebook
jupyter notebook chatbot/train_chatbot.ipynb
```

**Dans le notebook :**
- Exécute toutes les cellules (Cell → Run All)
- Attends la fin de l'entraînement (~5-10 minutes)
- Le modèle sera sauvegardé dans `gpt2-finetuned-smartshop-final/`

### 2️⃣ Copier le modèle dans le projet

```bash
# Créer le dossier models
mkdir -p chatbot/models

# Copier le modèle entraîné
cp -r gpt2-finetuned-smartshop-final chatbot/models/gpt2-finetuned
```

**Ou sur Windows :**
```powershell
New-Item -ItemType Directory -Force -Path chatbot\models
Copy-Item -Recurse gpt2-finetuned-smartshop-final chatbot\models\gpt2-finetuned
```

### 3️⃣ Tester le chatbot

```bash
# Installer les dépendances DL
pip install transformers torch sentence-transformers scikit-learn

# Lancer l'application
python app.py
```

Le chatbot utilisera automatiquement ton modèle Deep Learning ! 🎉

---

## 🧪 Test rapide

```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need headphones under 150€",
    "history": []
  }'
```

Tu devrais voir dans les logs :
```
[chatbot] mode DEEP LEARNING — RAG + Fine-tuned GPT-2
[chatbot] RAG retrieved: ['Wireless Noise-Cancelling Headphones', ...]
```

---

## ⚙️ Configuration

Dans ton fichier `.env` :

```bash
# Utiliser le modèle DL (recommandé)
USE_DL_MODEL=true

# Fallback si le modèle DL n'est pas disponible
USE_OLLAMA=false
GROQ_API_KEY=your_key_here
```

---

## 🐛 Troubleshooting

### Le modèle ne se charge pas

**Problème :** `DL model not available`

**Solution :**
1. Vérifie que le dossier `chatbot/models/gpt2-finetuned/` existe
2. Vérifie qu'il contient les fichiers : `config.json`, `pytorch_model.bin`, etc.
3. Réexécute le notebook d'entraînement

### Erreur de mémoire

**Problème :** `RuntimeError: CUDA out of memory`

**Solution :**
- Le modèle tourne sur CPU par défaut (pas besoin de GPU)
- Si problème persiste, réduis `per_device_train_batch_size` dans le notebook

### Dépendances manquantes

**Problème :** `ModuleNotFoundError: No module named 'transformers'`

**Solution :**
```bash
pip install transformers torch sentence-transformers scikit-learn
```

---

## 📊 Vérifier que ça marche

### Dans les logs du serveur

Tu dois voir :
```
[ChatbotModel] Loading models...
  ✓ Embeddings model loaded: sentence-transformers/all-MiniLM-L6-v2
  ✓ Loading fine-tuned GPT-2 from: chatbot/models/gpt2-finetuned
[ChatbotModel] ✓ All models loaded successfully!

[chatbot] ✓ Deep Learning model loaded (RAG + Fine-tuned GPT-2)
```

### Dans la réponse API

Tu dois recevoir :
```json
{
  "reply": "I recommend the Wireless Noise-Cancelling Headphones...",
  "suggested_products": ["(prod_001)"],
  "timestamp": "2024-..."
}
```

---

## 🎯 Pour présenter au prof

1. **Montre le notebook** (`train_chatbot.ipynb`) :
   - Dataset créé
   - Entraînement de GPT-2
   - Tests du modèle

2. **Montre le code** (`model.py`) :
   - Architecture RAG
   - Classe ChatbotModel
   - Méthodes retrieve_products() et generate_response()

3. **Démo live** :
   - Lance l'app
   - Teste plusieurs requêtes
   - Montre les logs (RAG + GPT-2)

4. **Explique l'architecture** :
   ```
   User Query → Sentence-BERT → Top-K Products → GPT-2 → Response
   ```

---

## ✅ Checklist finale

- [ ] Notebook exécuté avec succès
- [ ] Modèle copié dans `chatbot/models/gpt2-finetuned/`
- [ ] Dépendances installées (`transformers`, `torch`, etc.)
- [ ] `.env` configuré avec `USE_DL_MODEL=true`
- [ ] Serveur démarre sans erreur
- [ ] Test API fonctionne
- [ ] Logs montrent "DEEP LEARNING mode"

**Si tous les points sont cochés, tu es prêt ! 🚀**
