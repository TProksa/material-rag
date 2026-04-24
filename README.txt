============================================================
  Material RAG — Znalostní báze materiálů
  Verze 1.0 | Taro Plast + NUREL datasheets
============================================================

RYCHLÝ START
────────────
Windows:  Dvakrát klikni na  run.bat
Mac/Linux: Spusť ./run.sh

Pak otevři prohlížeč na: http://localhost:8000


CO APLIKACE UMÍ
───────────────
• Prohledávání technických datových listů (BM25 algoritmus)
• Automatické zvýraznění klíčových vlastností (pevnost, hustota, teploty...)
• Nahrávání nových PDF přímo z aplikace (tlačítko v levém panelu)
• Automatické re-indexování po přidání souboru
• REST API pro integraci s dalšími systémy


STRUKTURA PROJEKTU
──────────────────
material-rag-server/
├── server.py          ← FastAPI backend (BM25 engine + REST API)
├── index.html         ← Frontend (chat UI)
├── requirements.txt   ← Python závislosti
├── run.bat            ← Spuštění na Windows
├── run.sh             ← Spuštění na Mac/Linux
├── README.txt         ← Tento soubor
└── data/              ← PDF soubory (sem přidávej nové)
    ├── NILFLEX SH A55 M131 U 01R 99 - TDS.pdf
    ├── TAROMID A 280 H G6 DX0 TR1_STANDARD_004_EN.pdf
    └── ... (celkem 10 PDF)


REST API ENDPOINTY
──────────────────
GET  /api/stats              → statistiky (počet chunků, produktů...)
GET  /api/products           → katalog všech produktů
GET  /api/product?name=...   → detail produktu
POST /api/chat               → RAG dotaz
     Body: { "message": "...", "top_k": 5 }
POST /api/upload             → nahrání nového PDF (multipart/form-data)


PŘIDÁNÍ NOVÝCH PDF
──────────────────
Způsob 1: Přes webové rozhraní
  → Klikni na "Nahrát nové PDF" v levém panelu
  → Soubor se automaticky zaindexuje

Způsob 2: Ručně
  → Zkopíruj PDF do složky data/
  → Restartuj server


POŽADAVKY
─────────
• Python 3.9+
• Internetové připojení pro instalaci závislostí (jen poprvé)
• Závislosti: fastapi, uvicorn, pdfplumber, python-multipart, numpy


ROZŠÍŘENÍ O OPENAI
──────────────────
Systém je připraven pro napojení na OpenAI API.
Stačí přidat do server.py:

  from openai import OpenAI
  client = OpenAI(api_key="sk-...")

  # V endpointu /api/chat:
  context = "\n\n".join([r["content"] for r in results[:3]])
  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": "Jsi expert na technické materiály. Odpovídej na základě poskytnutého kontextu."},
          {"role": "user", "content": f"Kontext:\n{context}\n\nOtázka: {req.message}"}
      ]
  )
  answer = response.choices[0].message.content

============================================================
