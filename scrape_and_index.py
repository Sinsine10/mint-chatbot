"""
scrape_and_index.py
Run this ONCE to scrape mint.gov.et and build a local vector index.
Usage: python scrape_and_index.py
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pickle, time, re

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

BASE_URL = "http://www.mint.gov.et/"
MAX_PAGES = 60
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
INDEX_FILE = "mint_index.faiss"
CHUNKS_FILE = "mint_chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── 1. Crawl ──────────────────────────────────────────────────────────────────

def get_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if "text/html" not in r.headers.get("content-type", ""):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text()).strip()
    except Exception as e:
        print(f"  x {url}: {e}")
        return ""


def crawl(base: str, max_pages: int) -> list[dict]:
    visited, queue, pages = set(), [base], []
    domain = urlparse(base).netloc

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"  Crawling ({len(pages)+1}/{max_pages}): {url}")
        text = get_text(url)
        if len(text) > 200:
            pages.append({"url": url, "text": text})

        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"]).split("#")[0].split("?")[0]
                if urlparse(link).netloc == domain and link not in visited:
                    queue.append(link)
        except:
            pass
        time.sleep(0.4)

    return pages


# ── 2. Chunk ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, url: str, size: int, overlap: int) -> list[dict]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if len(chunk) > 80:
            chunks.append({"text": chunk, "url": url})
        i += size - overlap
    return chunks


# ── 3. Embed + index ──────────────────────────────────────────────────────────

def build_index(chunks: list[dict], model_name: str):
    print(f"\nEmbedding {len(chunks)} chunks with '{model_name}'...")
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    # Guard: catch empty array before FAISS crashes on it
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"Embedding returned shape {embeddings.shape} — expected (N, dim).\n"
            "The text list was empty or the model returned nothing.\n"
            "Check Step 2 output to confirm chunks were created."
        )

    print(f"  Embedding shape: {embeddings.shape}")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 1: Crawling mint.gov.et ===")
    pages = crawl(BASE_URL, MAX_PAGES)
    print(f"\nCollected {len(pages)} pages")

    # ── Diagnose crawl failure ──
    if not pages:
        print("\n[!] No pages scraped. Running connectivity check...")
        try:
            r = requests.get(BASE_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            print(f"    HTTP {r.status_code} | content-type: {r.headers.get('content-type','?')}")
            print(f"    Response length: {len(r.text)} chars")
            if len(r.text) < 500:
                print(f"    Body: {r.text[:300]}")
            else:
                print("    Site is reachable but no pages passed the 200-char filter.")
                print("    The site likely uses JavaScript to render text.")
                print("    --> Use the Selenium fallback described below.")
        except Exception as e:
            print(f"    Connection failed: {e}")
        raise SystemExit(1)

    print("\n=== Step 2: Chunking ===")
    all_chunks = []
    for p in pages:
        all_chunks.extend(chunk_text(p["text"], p["url"], CHUNK_SIZE, CHUNK_OVERLAP))
    print(f"Total chunks: {len(all_chunks)}")

    # ── Diagnose empty chunks (JS-rendered site) ──
    if not all_chunks:
        print("\n[!] Pages fetched but no usable text extracted.")
        print("    The site renders content with JavaScript — requests cannot see it.")
        print("    Sample raw text from first page:")
        print("   ", pages[0]["text"][:400])
        print()
        print("=== Selenium fallback ===")
        print("Install:  pip install selenium webdriver-manager")
        print("Replace get_text() with this version:\n")
        print("""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

def get_text(url):
    opts = Options()
    opts.add_argument('--headless')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    try:
        driver.get(url)
        import time; time.sleep(3)
        from bs4 import BeautifulSoup
        import re
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for tag in soup(['script','style','nav','footer','header']):
            tag.decompose()
        return re.sub(r'\\s+', ' ', soup.get_text()).strip()
    except Exception as e:
        print(f'  x {url}: {e}')
        return ''
    finally:
        driver.quit()
        """)
        raise SystemExit(1)

    print(f"\nSample chunk preview:")
    print(f"  URL : {all_chunks[0]['url']}")
    print(f"  Text: {all_chunks[0]['text'][:200]}...")

    print("\n=== Step 3: Building FAISS index ===")
    index = build_index(all_chunks, EMBED_MODEL)

    print("\n=== Saving ===")
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nDone! Saved '{INDEX_FILE}' and '{CHUNKS_FILE}' ({len(all_chunks)} chunks)")
    print("Next: run  python chatbot.py")