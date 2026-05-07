---
title: MINT Ethiopia Chatbot
emoji: 🇪🇹
colorFrom: green
colorTo: teal
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# MINT Ethiopia RAG Chatbot

A fully local Retrieval-Augmented Generation chatbot that answers questions about Ethiopia's **Ministry of Innovation and Technology (MINT)** using content scraped directly from [mint.gov.et](https://www.mint.gov.et/).

**No external API key required for basic use.**

## How it works

1. `scrape_and_index.py` (run locally) crawls mint.gov.et and builds a FAISS vector index
2. On each question, the app embeds your query and retrieves the most relevant chunks
3. Those chunks are sent as context to Mistral-7B via the HF Inference API
4. The answer is grounded in real MINT content with source links

## Deployment steps

### 1. Run indexing locally
```bash
pip install requests beautifulsoup4 sentence-transformers faiss-cpu
python scrape_and_index.py
# produces: mint_index.faiss, mint_chunks.pkl
```

### 2. Create a Hugging Face Space
- Go to https://huggingface.co/new-space
- SDK: **Gradio**, Hardware: **CPU basic (free)**

### 3. Push all files
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/mint-chatbot
cp app.py requirements.txt README.md mint_index.faiss mint_chunks.pkl mint-chatbot/
cd mint-chatbot
git add .
git commit -m "initial deploy"
git push
```

### 4. (Optional but recommended) Add HF_TOKEN secret
- Space Settings → Variables and Secrets → New Secret
- Name: `HF_TOKEN`, Value: your HF read token (free at huggingface.co/settings/tokens)
- This gives higher rate limits on the inference API

## Refreshing the knowledge base

When MINT updates their website, just re-run `scrape_and_index.py` and push the new `.faiss` and `.pkl` files.
