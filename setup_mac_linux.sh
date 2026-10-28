#!/bin/bash
# One-command setup for macOS / Linux:  bash setup_mac_linux.sh
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
python scripts/build_corpus.py && python scripts/ingest_corpus.py --embedder hashing
echo "Setup finished. Start the app with:  source venv/bin/activate && streamlit run app.py"
