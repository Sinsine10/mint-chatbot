"""
chatbot.py
MINT Ethiopia RAG Chatbot — fully local, no API key needed.
Requirements: scrape_and_index.py must have been run first.
Usage: python chatbot.py
"""

import pickle, textwrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama
import gradio as gr

INDEX_FILE  = "mint_index.faiss"
CHUNKS_FILE = "mint_chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "mistral"          # change to "llama3" or "phi3" if you prefer
TOP_K       = 4                  # how many chunks to retrieve

SYSTEM_PROMPT = """You are a helpful assistant for Ethiopia's Ministry of Innovation and Technology (MINT).
Answer questions accurately using only the provided context from the MINT website.
If the context does not contain enough information, say so honestly.
Be concise, clear, and professional.
Context:
{context}"""


# ── Load index + model (once at startup) ─────────────────────────────────────

print("Loading index and embedding model...")
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)
embed_model = SentenceTransformer(EMBED_MODEL)
print(f"✓ Loaded {len(chunks)} chunks, index dim={index.d}")


# ── RAG core ─────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed the query and return the top-k most similar chunks."""
    q_vec = embed_model.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append({**chunks[idx], "score": float(score)})
    return results


def generate_answer(query: str, relevant_chunks: list[dict]) -> str:
    """Build a prompt from retrieved chunks and call the local LLM."""
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(f"[Source {i}: {chunk['url']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    system = SYSTEM_PROMPT.format(context=context)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
    )
    return response["message"]["content"].strip()


def chat(query: str, history: list) -> str:
    if not query.strip():
        return "Please enter a question."
    chunks_found = retrieve(query)
    if not chunks_found:
        return "Sorry, I couldn't find relevant information in the MINT knowledge base."
    answer = generate_answer(query, chunks_found)

    # Append source URLs (deduplicated)
    seen, sources = set(), []
    for c in chunks_found:
        url = c["url"]
        if url not in seen:
            seen.add(url)
            sources.append(url)
    source_text = "\n\n---\n**Sources:**\n" + "\n".join(f"- {u}" for u in sources)
    return answer + source_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    "What is MINT Ethiopia?",
    "What digital services does MINT provide?",
    "How can I contact the Ministry?",
    "What is the National Digital Payments strategy?",
    "What programs does MINT have for youth and startups?",
]

with gr.Blocks(title="MINT Ethiopia Chatbot") as demo:
    gr.Markdown(
        """
        # 🇪🇹 MINT Ethiopia Chatbot
        ### Powered by local RAG — no internet or API key required
        Ask anything about the Ministry of Innovation and Technology of Ethiopia.
        """
    )

    chatbot = gr.Chatbot(height=420)
    msg     = gr.Textbox(placeholder="Ask about MINT Ethiopia...", label="Your question", lines=1)
    clear   = gr.Button("Clear conversation", variant="secondary", size="sm")

    gr.Examples(examples=EXAMPLES, inputs=msg, label="Try these")

    def respond(message, chat_history):
        answer = chat(message, chat_history)
        chat_history.append((message, answer))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    print("\nStarting chatbot at http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())