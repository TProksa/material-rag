"""
Material RAG Server
FastAPI backend with BM25 search + GPT-4o answer generation.
Supports dynamic PDF upload and re-indexing.
"""
import os
import re
import math
import json
import logging
from pathlib import Path
from collections import Counter
from typing import Optional

import pdfplumber
import numpy as np
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# OpenAI — načti klíč z prostředí nebo ze souboru .env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Načti .env soubor pokud existuje
_env_file = Path(__file__).parent / ".env"
if _env_file.exists() and not OPENAI_API_KEY:
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("OPENAI_API_KEY="):
            OPENAI_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
        if line.startswith("OPENAI_MODEL="):
            OPENAI_MODEL = line.split("=", 1)[1].strip().strip('"').strip("'")
log = logging.getLogger("rag")

# ═══════════════════════════════════════════════════
# BM25 SEARCH ENGINE
# ═══════════════════════════════════════════════════

class BM25Engine:
    """BM25 full-text search with metadata boosting."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.documents: list[dict] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.dl: list[int] = []
        self.avgdl: float = 0
        self.N: int = 0

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r'[a-záčďéěíňóřšťúůýž0-9][a-záčďéěíňóřšťúůýž0-9.,/\-]*', text)
        expanded = []
        for t in tokens:
            expanded.append(t)
            parts = re.findall(r'[a-záčďéěíňóřšťúůýž]+|[0-9]+(?:[.,][0-9]+)?', t)
            if len(parts) > 1:
                expanded.extend(parts)
        return expanded

    def index(self, documents: list[dict]):
        self.documents = documents
        self.N = len(documents)
        self.tf, self.dl = [], []
        self.df = Counter()

        for doc in documents:
            text = f"{doc['title']} {doc['product']} {doc['category']} {doc['content']}"
            tokens = self.tokenize(text)
            self.dl.append(len(tokens))
            tf = Counter(tokens)
            self.tf.append(tf)
            for token in set(tokens):
                self.df[token] += 1

        self.avgdl = sum(self.dl) / self.N if self.N else 1
        log.info(f"Indexed {self.N} chunks, vocab size {len(self.df)}")

    def _score(self, query_tokens: list[str], idx: int) -> float:
        s = 0.0
        dl = self.dl[idx]
        tf = self.tf[idx]
        for token in query_tokens:
            n = self.df.get(token, 0)
            if n == 0:
                continue
            idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
            freq = tf.get(token, 0)
            s += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * (dl / self.avgdl)))
        return s

    def search(self, query: str, top_k: int = 5, category: str = None, product: str = None) -> list[dict]:
        tokens = self.tokenize(query)
        results = []
        for i, doc in enumerate(self.documents):
            if category and category.lower() not in doc["category"].lower():
                continue
            if product and product.lower() not in doc["product"].lower():
                continue
            score = self._score(tokens, i)
            # Metadata boosting
            for t in tokens:
                if t in doc["product"].lower():
                    score *= 1.4
                if t in doc["title"].lower():
                    score *= 1.2
            if score > 0:
                results.append({"idx": i, "score": round(score, 3), **doc})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ═══════════════════════════════════════════════════
# PDF EXTRACTION & CHUNKING
# ═══════════════════════════════════════════════════

def extract_text(path: Path) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def identify_product(text: str, filename: str) -> dict:
    info = {
        "filename": filename,
        "product_name": "",
        "manufacturer": "",
        "material_type": "",
        "iso_short": "",
        "doc_type": "",
    }

    if "taro plast" in text.lower() or "taromid" in filename.lower() or "nilflex" in filename.lower():
        info["manufacturer"] = "Taro Plast S.p.A."
    elif "nurel" in text.lower() or "promyde" in text.lower() or "PA 6100" in filename:
        info["manufacturer"] = "NUREL S.A. (PROMYDE)"

    fname = filename.replace(".pdf", "").strip()
    if " - " in fname:
        parts = fname.split(" - ")
        info["product_name"] = parts[0].strip()
        info["doc_type"] = parts[1].strip()
    elif "_STANDARD_" in fname:
        info["product_name"] = fname.split("_STANDARD_")[0].replace("_", " ").strip()
        info["doc_type"] = "TDS"
    else:
        info["product_name"] = fname
        info["doc_type"] = "TDS"

    if "CERTIFICATE OF COMPLIANCE" in text or "COMPLIANCE CERTIFICATE" in text:
        info["doc_type"] = "Certificate"
    elif "PRODUCT INFORMATION" in text or "PRELIMINARY DATASHEET" in text:
        info["doc_type"] = "TDS"

    m = re.search(r'ISO\s+(?:short\s+)?ISO\s+\d+[:\s]+([A-Z0-9\-\(\)]+(?:\s*[A-Z0-9\-\(\)]+)*)', text)
    if m:
        info["iso_short"] = m.group(1).strip()

    if "polyamide 66" in text.lower() or "PA66" in text:
        info["material_type"] = "PA66 (Polyamide 66)"
    elif "polyamide 6 " in text.lower() or "PA6-" in text:
        info["material_type"] = "PA6 (Polyamide 6)"
    elif "SEBS" in text or "TPE-S" in text:
        info["material_type"] = "TPE-S (SEBS Elastomer)"
    elif "PA 6100" in text:
        info["material_type"] = "PA (Polyamide)"

    return info


SECTION_MARKERS = ["ELECTRICAL", "PHYSICAL", "MECHANICAL", "THERMAL", "FLAMMABILITY", "INJECTION MOULDING"]
CATEGORY_MAP = {
    "Electrical": "Elektrické vlastnosti",
    "Physical": "Fyzikální vlastnosti",
    "Mechanical": "Mechanické vlastnosti",
    "Thermal": "Tepelné vlastnosti",
    "Flammability": "Hořlavost",
    "Injection Moulding": "Parametry vstřikování",
}


def chunk_document(text: str, info: dict) -> list[dict]:
    product = info["product_name"]
    manufacturer = info["manufacturer"]
    chunks = []

    if info["doc_type"] == "Certificate":
        chunks.append({
            "id": f"{product}__certificate__{info['filename']}",
            "product": product,
            "manufacturer": manufacturer,
            "material_type": info["material_type"],
            "category": "Certifikát",
            "title": f"{product} - Certifikát shody",
            "content": text,
            "iso_short": info.get("iso_short", ""),
        })
        return chunks

    # Parse sections
    sections = {}
    current = "General"
    lines_buf = []
    for line in text.split("\n"):
        upper = line.strip().upper()
        matched = False
        for marker in SECTION_MARKERS:
            if upper == marker or upper.startswith(marker + " "):
                if lines_buf:
                    sections[current] = "\n".join(lines_buf)
                current = marker.title()
                lines_buf = []
                matched = True
                break
        if not matched:
            lines_buf.append(line)
    if lines_buf:
        sections[current] = "\n".join(lines_buf)

    # Overview
    overview = []
    for line in text.split("\n")[:30]:
        if any(m in line.upper() for m in SECTION_MARKERS[:3]):
            break
        overview.append(line)
    if overview:
        chunks.append({
            "id": f"{product}__overview",
            "product": product,
            "manufacturer": manufacturer,
            "material_type": info["material_type"],
            "category": "Přehled produktu",
            "title": f"{product} - Přehled",
            "content": "\n".join(overview).strip(),
            "iso_short": info.get("iso_short", ""),
        })

    for sec_name, sec_text in sections.items():
        if sec_name == "General" or len(sec_text.strip()) < 20:
            continue
        cat = CATEGORY_MAP.get(sec_name, sec_name)
        enriched = f"Produkt: {product}\nVýrobce: {manufacturer}\nISO: {info.get('iso_short','')}\nKategorie: {cat}\n\n{sec_text}"
        chunks.append({
            "id": f"{product}__{sec_name.lower().replace(' ', '_')}",
            "product": product,
            "manufacturer": manufacturer,
            "material_type": info["material_type"],
            "category": cat,
            "title": f"{product} - {cat}",
            "content": enriched,
            "iso_short": info.get("iso_short", ""),
        })

    return chunks


def load_all_pdfs() -> list[dict]:
    all_chunks = []
    for pdf_file in sorted(DATA_DIR.glob("*.pdf")):
        log.info(f"Processing: {pdf_file.name}")
        text = extract_text(pdf_file)
        if not text:
            log.warning(f"  No text: {pdf_file.name}")
            continue
        info = identify_product(text, pdf_file.name)
        chunks = chunk_document(text, info)
        log.info(f"  → {info['product_name']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


# ═══════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════

engine = BM25Engine()
chunks = load_all_pdfs()
engine.index(chunks)

products_catalog = {}
for c in chunks:
    p = c["product"]
    if p not in products_catalog:
        products_catalog[p] = {
            "name": p,
            "manufacturer": c["manufacturer"],
            "material_type": c["material_type"],
            "iso_short": c.get("iso_short", ""),
            "categories": [],
        }
    if c["category"] not in products_catalog[p]["categories"]:
        products_catalog[p]["categories"].append(c["category"])

log.info(f"Ready: {len(chunks)} chunks, {len(products_catalog)} products")

# ═══════════════════════════════════════════════════
# OPENAI ANSWER GENERATION
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """Jsi expertní asistent pro technické materiály (plasty, elastomery, polyamidy).
Odpovídáš na dotazy na základě poskytnutých úryvků z technických datových listů (TDS).

Pravidla:
- Odpovídej vždy v češtině, pokud se tě uživatel neptá jinak.
- Vycházej POUZE z poskytnutého kontextu — nevymýšlej hodnoty.
- Pokud kontext neobsahuje odpověď, řekni to otevřeně.
- Buď konkrétní: cituj číselné hodnoty, normy, podmínky měření.
- Formátuj odpověď přehledně — použij odrážky pro výčty vlastností.
- Na konci uveď, ze kterého produktu/dokumentu informace pochází."""


def generate_openai_answer(query: str, sources: list[dict]) -> str | None:
    """Vygeneruj odpověď pomocí GPT-4o na základě retrieved chunků."""
    if not OPENAI_API_KEY:
        return None

    try:
        import urllib.request
        import json as _json

        context_parts = []
        for i, s in enumerate(sources[:4], 1):
            context_parts.append(
                f"[{i}] {s['title']} (skóre: {s['score']})\n"
                f"Výrobce: {s['manufacturer']} | Typ: {s.get('material_type','')}\n"
                f"{s['content'][:800]}"
            )
        context = "\n\n---\n\n".join(context_parts)

        payload = _json.dumps({
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Kontext z datových listů:\n\n{context}\n\n---\n\nOtázka: {query}"},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
            return result["choices"][0]["message"]["content"]

    except Exception as e:
        log.warning(f"OpenAI call failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════

app = FastAPI(title="Material RAG", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    top_k: int = 5
    category: Optional[str] = None
    product: Optional[str] = None


class ApiKeyRequest(BaseModel):
    api_key: str


@app.post("/api/set-key")
def set_api_key(req: ApiKeyRequest):
    """Nastav OpenAI API klíč za běhu (uloží do .env)."""
    global OPENAI_API_KEY
    OPENAI_API_KEY = req.api_key.strip()
    # Ulož do .env pro příští spuštění
    env_path = Path(__file__).parent / ".env"
    env_path.write_text(f'OPENAI_API_KEY="{OPENAI_API_KEY}"\nOPENAI_MODEL="{OPENAI_MODEL}"\n')
    log.info("OpenAI API klíč nastaven a uložen do .env")
    return {"status": "ok", "model": OPENAI_MODEL}


@app.get("/api/ai-status")
def ai_status():
    """Vrátí stav OpenAI integrace."""
    return {
        "openai_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "key_preview": f"{OPENAI_API_KEY[:8]}..." if OPENAI_API_KEY else None,
    }


@app.get("/api/stats")
def get_stats():
    categories = Counter(c["category"] for c in chunks)
    manufacturers = Counter(c["manufacturer"] for c in chunks)
    return {
        "total_chunks": len(chunks),
        "total_products": len(products_catalog),
        "categories": dict(categories),
        "manufacturers": dict(manufacturers),
    }


@app.get("/api/products")
def get_products():
    return {"products": list(products_catalog.values())}


@app.get("/api/product")
def get_product(name: str = Query(...)):
    matching = [c for c in chunks if name.lower() in c["product"].lower()]
    return {"product": name, "chunks": matching}


@app.post("/api/chat")
def chat(req: ChatRequest):
    results = engine.search(req.message, top_k=req.top_k, category=req.category, product=req.product)
    prods = list({r["product"] for r in results})

    if not results:
        return {
            "type": "answer",
            "answer": "Nenašel jsem žádné relevantní informace. Zkuste jiný dotaz nebo upřesnit název produktu.",
            "sources": [],
            "products_mentioned": [],
            "ai_powered": False,
        }

    sources_out = [
        {
            "title": r["title"],
            "product": r["product"],
            "category": r["category"],
            "manufacturer": r["manufacturer"],
            "material_type": r.get("material_type", ""),
            "iso_short": r.get("iso_short", ""),
            "content": r["content"],
            "score": r["score"],
        }
        for r in results
    ]

    # Pokus o GPT-4o odpověď
    ai_answer = generate_openai_answer(req.message, sources_out)

    if ai_answer:
        answer = ai_answer
        ai_powered = True
    else:
        # Fallback bez OpenAI
        top = results[0]
        if len(prods) == 1:
            answer = f"**{top['product']}** ({top['manufacturer']})\nMateriál: {top['material_type']}\n\nNalezeno {len(results)} relevantních sekcí:"
        else:
            answer = f"Nalezeno ve {len(prods)} produktech: {', '.join(prods)}"
        ai_powered = False
        log.info("OpenAI nedostupné — fallback na BM25 výsledky")

    return {
        "type": "answer",
        "answer": answer,
        "sources": sources_out,
        "products_mentioned": prods,
        "ai_powered": ai_powered,
    }


# Upload endpoint disabled in read-only variant


# Serve frontend
FRONTEND_FILE = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
def serve_index():
    return FRONTEND_FILE.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 52)
    print("  🔬 Material RAG Knowledge Base")
    print(f"  📦 {len(chunks)} chunks | {len(products_catalog)} products")
    print(f"  🌐 http://localhost:{port}")
    print("=" * 52 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
