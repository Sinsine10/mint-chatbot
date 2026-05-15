import os, pickle, requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TOP_K        = 4

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODEL   = "llama-3.3-70b-versatile" 

SYSTEM_PROMPT = (
    "You are a helpful assistant for Ethiopia's Ministry of Innovation and Technology (MINT).\n"
    "Answer questions accurately using only the provided context.\n\nContext:\n{context}"
)


print("Booting up...")
try:
    index = faiss.read_index("mint_index.faiss")
    with open("mint_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL)
    print("✓ RAG engine ready.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

def retrieve(query: str) -> list:
    q = embed_model.encode([query], show_progress_bar=False)
    q = np.array(q, dtype="float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, TOP_K)
    return [chunks[i] for i in ids[0] if i >= 0]

def call_groq(system: str, user: str) -> str:
    if not GROQ_API_KEY:
        return "ERROR: Missing API Key in Space Settings."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.1,
    }
    
    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
        if r.status_code != 200:
            return f"Groq Error ({r.status_code}): {r.text}"
        
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"System Error: {str(e)}"

def chat(message, history):
    
    hits = retrieve(message)
    
  
    context = "\n\n".join([h['text'] for h in hits])
    system = SYSTEM_PROMPT.format(context=context)
    
   
    answer = call_groq(system, message)
    
 
    return answer


demo = gr.ChatInterface(
    fn=chat,
    title="MINT Ethiopia Assistant",
    description="Ask anything about the Ministry of Innovation and Technology."
)

if __name__ == "__main__":
    demo.launch()
