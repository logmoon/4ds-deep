"""
chatbot/model.py
================
Deep Learning chatbot model: RAG + GPT-2 base
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ChatbotModel:
    """RAG + GPT-2 chatbot for e-commerce."""
    
    def __init__(self, gpt2_path=None, embeddings_model="sentence-transformers/all-MiniLM-L6-v2"):
        print("[ChatbotModel] Loading models...")
        
        # Load embeddings for RAG
        self.embedder = SentenceTransformer(embeddings_model)
        print(f"  ✓ Embeddings: {embeddings_model}")
        
        # Load base GPT-2
        print("  ✓ Loading GPT-2 base")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        
        print("[ChatbotModel] ✓ Ready!\n")
    
    def retrieve_products(self, query: str, catalogue: list, top_k: int = 3) -> list:
        """RAG: Retrieve relevant products."""
        if not catalogue:
            return []
        
        query_emb = self.embedder.encode(query)
        texts = [f"{p['name']} {p.get('category', '')}" for p in catalogue]
        product_embs = self.embedder.encode(texts)
        
        sims = cosine_similarity([query_emb], product_embs)[0]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        
        return [catalogue[i] for i in top_idx]
    
    def generate_response(self, message: str, history: list, context_products: list, max_length: int = 60, temperature: float = 0.7) -> str:
        """Generate response with GPT-2."""
        prompt = f"Products: "
        for p in context_products[:2]:
            prompt += f"{p['name']} €{p['price']}, "
        prompt += f"\nQ: {message}\nA:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=300, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )
        
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full[len(prompt):].strip()
        
        # Clean
        if "\n" in response:
            response = response.split("\n")[0]
        if len(response) > 120:
            response = response[:120].rsplit('.', 1)[0] + '.'
        
        return response.strip()


_chatbot_instance = None

def get_chatbot_model(gpt2_path=None):
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = ChatbotModel()
    return _chatbot_instance
