#!/bin/bash
echo ""
echo " =========================================="
echo "  Material RAG - Znalostní báze materiálů"
echo " =========================================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo " [CHYBA] Python 3 není nainstalován."
    exit 1
fi

# Install deps if needed
echo " Kontroluji závislosti..."
if ! python3 -c "import fastapi" &>/dev/null; then
    echo " Instaluji závislosti (první spuštění)..."
    pip3 install -r requirements.txt
fi

echo ""
echo " Server běží na: http://localhost:8000"
echo " Ukončení: Ctrl+C"
echo ""

# Open browser (works on macOS and Linux with xdg-open)
(sleep 2 && (open "http://localhost:8000" 2>/dev/null || xdg-open "http://localhost:8000" 2>/dev/null)) &

python3 server.py
