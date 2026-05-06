# 📚 Index — Documentation Chatbot Deep Learning

## 🎯 Par où commencer ?

### Si tu veux **utiliser rapidement** :
👉 Lis **[QUICKSTART.md](QUICKSTART.md)** (5 minutes)

### Si tu veux **comprendre l'architecture** :
👉 Lis **[ARCHITECTURE.md](ARCHITECTURE.md)** (15 minutes)

### Si tu veux **comprendre l'intégration** :
👉 Lis **[INTEGRATION.md](INTEGRATION.md)** (10 minutes)

### Si tu veux **tout savoir** :
👉 Lis **[README.md](README.md)** (20 minutes)

### Si tu veux **un résumé pour le prof** :
👉 Lis **[SUMMARY.md](SUMMARY.md)** (5 minutes)

---

## 📁 Structure des fichiers

```
chatbot/
│
├── 📓 train_chatbot.ipynb      ← COMMENCE ICI (entraînement)
│   └─ Notebook Jupyter complet pour fine-tuner GPT-2
│
├── 🧠 model.py                 ← Code du modèle DL
│   └─ Classe ChatbotModel (RAG + GPT-2)
│
├── 🔌 routes.py                ← API Flask (modifié)
│   └─ Endpoint /api/chat qui utilise le modèle DL
│
├── 📊 dataset.json             ← Dataset d'entraînement
│   └─ 15 conversations e-commerce
│
├── 📖 Documentation/
│   ├── README.md               ← Guide complet du module
│   ├── QUICKSTART.md           ← Démarrage rapide (3 étapes)
│   ├── ARCHITECTURE.md         ← Diagrammes techniques
│   ├── INTEGRATION.md          ← Explications du code
│   ├── SUMMARY.md              ← Résumé pour le prof
│   └── INDEX.md                ← Ce fichier
│
└── models/                     ← Modèles entraînés (à créer)
    └── gpt2-finetuned/         ← Ton GPT-2 fine-tuné
```

---

## 🚀 Workflow en 3 étapes

### 1️⃣ Entraînement (30 min)

```bash
# Ouvrir le notebook
jupyter notebook train_chatbot.ipynb

# Exécuter toutes les cellules
# → Le modèle sera sauvegardé dans gpt2-finetuned-smartshop-final/
```

📖 **Détails :** [train_chatbot.ipynb](train_chatbot.ipynb)

---

### 2️⃣ Installation (5 min)

```bash
# Copier le modèle
cp -r gpt2-finetuned-smartshop-final chatbot/models/gpt2-finetuned

# Installer les dépendances
pip install transformers torch sentence-transformers scikit-learn

# Configurer
echo "USE_DL_MODEL=true" >> .env
```

📖 **Détails :** [QUICKSTART.md](QUICKSTART.md)

---

### 3️⃣ Utilisation (2 min)

```bash
# Lancer l'app
python app.py

# Tester
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need headphones", "history": []}'
```

📖 **Détails :** [README.md](README.md)

---

## 📚 Guide de lecture par objectif

### 🎓 Pour présenter au prof

1. **[SUMMARY.md](SUMMARY.md)** — Résumé exécutif
   - Ce qui a été fait
   - Preuves de Deep Learning
   - Comparaison DL vs API

2. **[train_chatbot.ipynb](train_chatbot.ipynb)** — Preuve d'entraînement
   - Dataset créé
   - Fine-tuning GPT-2
   - Tests et évaluation

3. **[ARCHITECTURE.md](ARCHITECTURE.md)** — Explications techniques
   - Diagrammes
   - Composants DL
   - Références académiques

---

### 💻 Pour développer/débugger

1. **[model.py](model.py)** — Code du modèle
   - Classe `ChatbotModel`
   - Méthodes RAG et génération
   - Singleton pattern

2. **[routes.py](routes.py)** — API Flask
   - Endpoint `/api/chat`
   - Fonction `call_dl_model()`
   - Configuration

3. **[INTEGRATION.md](INTEGRATION.md)** — Explications du code
   - Flux d'exécution
   - Ce qui est appelé
   - Exemples détaillés

---

### 🚀 Pour démarrer rapidement

1. **[QUICKSTART.md](QUICKSTART.md)** — 3 étapes simples
   - Entraîner
   - Installer
   - Tester

2. **[dataset.json](dataset.json)** — Dataset prêt à l'emploi
   - 15 conversations
   - Format correct

---

### 🧠 Pour comprendre l'architecture

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — Diagrammes complets
   - Vue d'ensemble
   - Composants DL
   - Flux de données

2. **[README.md](README.md)** — Documentation complète
   - Architecture
   - Utilisation
   - Configuration

---

## 🎯 Checklist de compréhension

### Niveau 1 : Utilisation basique
- [ ] J'ai exécuté le notebook
- [ ] J'ai copié le modèle
- [ ] L'API fonctionne
- [ ] Je vois les logs "DEEP LEARNING mode"

### Niveau 2 : Compréhension technique
- [ ] Je comprends ce qu'est le RAG
- [ ] Je comprends le fine-tuning
- [ ] Je sais ce que fait Sentence-BERT
- [ ] Je sais ce que fait GPT-2

### Niveau 3 : Maîtrise complète
- [ ] Je peux expliquer l'architecture au prof
- [ ] Je peux modifier le dataset
- [ ] Je peux ajuster les hyperparamètres
- [ ] Je peux débugger les erreurs

---

## 🔗 Liens rapides

| Fichier | Description | Temps de lecture |
|---------|-------------|------------------|
| [QUICKSTART.md](QUICKSTART.md) | Démarrage rapide | 5 min |
| [SUMMARY.md](SUMMARY.md) | Résumé pour le prof | 5 min |
| [INTEGRATION.md](INTEGRATION.md) | Explications du code | 10 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Diagrammes techniques | 15 min |
| [README.md](README.md) | Documentation complète | 20 min |
| [train_chatbot.ipynb](train_chatbot.ipynb) | Notebook d'entraînement | 30 min |
| [model.py](model.py) | Code du modèle | 10 min |
| [routes.py](routes.py) | API Flask | 5 min |
| [dataset.json](dataset.json) | Dataset | 2 min |

---

## 💡 FAQ Rapide

### Q: Par où commencer ?
**R:** Ouvre [QUICKSTART.md](QUICKSTART.md) et suis les 3 étapes.

### Q: Comment ça marche ?
**R:** Lis [ARCHITECTURE.md](ARCHITECTURE.md) pour les diagrammes.

### Q: Qu'est-ce qui est appelé dans routes.py ?
**R:** Lis [INTEGRATION.md](INTEGRATION.md) pour les détails.

### Q: Comment présenter au prof ?
**R:** Lis [SUMMARY.md](SUMMARY.md) pour le résumé.

### Q: Où est le code du modèle ?
**R:** Dans [model.py](model.py) (classe `ChatbotModel`).

### Q: Où est le notebook d'entraînement ?
**R:** [train_chatbot.ipynb](train_chatbot.ipynb)

### Q: C'est vraiment du Deep Learning ?
**R:** Oui ! Lis [SUMMARY.md](SUMMARY.md) section "Preuves de Deep Learning".

---

## 🎓 Pour le prof

### Documents à montrer

1. **Preuve d'entraînement** : [train_chatbot.ipynb](train_chatbot.ipynb)
2. **Architecture technique** : [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Code du modèle** : [model.py](model.py)
4. **Résumé exécutif** : [SUMMARY.md](SUMMARY.md)

### Points clés à expliquer

1. **RAG** : Recherche sémantique avec Sentence-BERT
2. **Fine-tuning** : Entraînement de GPT-2 sur dataset custom
3. **Architecture** : Transformer (12 layers, 124M params)
4. **Pas d'API** : Tout tourne localement

---

## ✅ Checklist finale

### Fichiers créés
- [x] `train_chatbot.ipynb` — Notebook d'entraînement
- [x] `model.py` — Code du modèle DL
- [x] `routes.py` — API Flask (modifié)
- [x] `dataset.json` — Dataset de 15 conversations
- [x] `README.md` — Documentation complète
- [x] `QUICKSTART.md` — Guide rapide
- [x] `ARCHITECTURE.md` — Diagrammes techniques
- [x] `INTEGRATION.md` — Explications du code
- [x] `SUMMARY.md` — Résumé pour le prof
- [x] `INDEX.md` — Ce fichier

### Prêt pour
- [x] Entraînement du modèle
- [x] Intégration dans Flask
- [x] Tests et démo
- [x] Présentation au prof

---

## 🚀 Prochaines étapes

1. **Maintenant** : Ouvre [QUICKSTART.md](QUICKSTART.md)
2. **Ensuite** : Exécute [train_chatbot.ipynb](train_chatbot.ipynb)
3. **Puis** : Teste l'API
4. **Enfin** : Lis [SUMMARY.md](SUMMARY.md) pour la présentation

**Bonne chance ! 🎉**
