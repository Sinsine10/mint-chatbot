"""
app.py  — Hugging Face Spaces entry point
RAG chatbot for MINT Ethiopia. Uses:
  - FAISS + sentence-transformers for retrieval (CPU, free)
  - HF Inference API (Mistral-7B-Instruct) for generation (free tier)
  - Gradio for the UI
"""

import os, pickle
import numpy as np
import faiss
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL  = "all-MiniLM-L6-v2"
HF_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")   # set as a Space secret
TOP_K        = 4
MAX_NEW_TOKENS = 512

SYSTEM = (
    "You are a helpful assistant for Ethiopia's Ministry of Innovation and "
    "Technology (MINT). Answer questions using only the provided context. "
    "If the answer is not in the context, say so honestly. Be concise.\n\n"
    "Context:\n{context}"
)

# ── Load index once at startup ────────────────────────────────────────────────

print("Loading FAISS index and embedding model...")
index = faiss.read_index("mint_index.faiss")
with open("mint_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
embed_model = SentenceTransformer(EMBED_MODEL)
print(f"✓ {len(chunks)} chunks loaded, index dim={index.d}")


# ── RAG helpers ───────────────────────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    q = embed_model.encode([query], show_progress_bar=False)
    q = np.array(q, dtype="float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, TOP_K)
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0]) if i >= 0
    ]


def call_hf_api(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.3,
            "do_sample": True,
            "return_full_text": False,
        },
    }
    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        return str(data)
    except requests.exceptions.Timeout:
        return "The model is loading on the server — please wait 20 seconds and try again."
    except Exception as e:
        return f"Error calling inference API: {e}"


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['url']}]\n{c['text']}" for c in context_chunks
    )
    system = SYSTEM.format(context=context)
    # Mistral instruct format
    return f"<s>[INST] {system}\n\nQuestion: {query} [/INST]"


def answer(query: str) -> tuple[str, list[dict]]:
    if not query.strip():
        return "Please enter a question.", []
    hits = retrieve(query)
    if not hits:
        return "No relevant information found in the MINT knowledge base.", []
    prompt = build_prompt(query, hits)
    reply  = call_hf_api(prompt)
    return reply, hits


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    "What is MINT Ethiopia?",
    "What digital services does MINT provide?",
    "How can I contact the Ministry?",
    "What is the National Digital Payments strategy?",
    "What programs does MINT have for startups?",
]

CSS = """
.source-box { font-size: 12px; color: #555; margin-top: 4px; }
footer { display: none !important; }
"""

with gr.Blocks(title="MINT Ethiopia Chatbot", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:16px 0 8px">
      <h1 style="font-size:1.6rem;margin:0">🇪🇹 MINT Ethiopia Assistant</h1>
      <p style="color:#666;margin:6px 0 0;font-size:.95rem">
        Ask anything about the Ministry of Innovation &amp; Technology of Ethiopia.<br>
        Powered by local RAG — answers come directly from
        <a href="https://www.mint.gov.et/" target="_blank">mint.gov.et</a>.
      </p>
    </div>
    """)

    chatbot  = gr.Chatbot(height=440, bubble_full_width=False, show_label=False)
    sources  = gr.JSON(label="Sources used", visible=False)
    with gr.Row():
        msg   = gr.Textbox(placeholder="Ask about MINT Ethiopia...", scale=5, show_label=False, lines=1)
        send  = gr.Button("Send", variant="primary", scale=1)
    show_src = gr.Checkbox(label="Show source chunks", value=False)
    clear    = gr.Button("Clear", variant="secondary", size="sm")

    gr.Examples(EXAMPLES, inputs=msg, label="Example questions")

    def respond(message, history, show_sources):
        reply, hits = answer(message)
        src_data = [{"url": h["url"], "score": round(h["score"], 3)} for h in hits]
        history.append((message, reply))
        return "", history, src_data if show_sources else None

    show_src.change(lambda v: gr.update(visible=v), show_src, sources)
    msg.submit(respond, [msg, chatbot, show_src], [msg, chatbot, sources])
    send.click(respond, [msg, chatbot, show_src], [msg, chatbot, sources])
    clear.click(lambda: ([], None), outputs=[chatbot, sources])

demo.launch()
