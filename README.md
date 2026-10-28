# RAG-Based Document Q&A System (Project 72 - HCLTech Career Shaper Programme 2025-26)

Ask questions about your PDF / Word / HTML / Markdown / TXT documents. The app finds the most relevant chunks (ChromaDB vector search),
writes a short answer with an LLM (or offline extractive mode) and **always shows the source chunks**.
It refuses to answer when evidence is weak. 

> ⚠️ RAG systems may hallucinate. Always read the source chunk. Do not use for legal or medical advice without expert review.

---------------------------------------------------------------------

## PART 1 - Run on your laptop (step by step)

### Step 0 - What you need
* A laptop with **8 GB RAM**, Windows 10/11 (macOS/Linux also fine). No GPU needed.
* **Python 3.10, 3.11 or 3.12** from https://www.python.org/downloads/ . During install **tick "Add Python to PATH"**.
* Check in Command Prompt: `python --version`

### Step 1 - Unzip and open a terminal in the folder
1. Right-click the zip -> *Extract All*. You get a folder `rag-doc-qa`.
2. Open the folder, click the address bar, type `cmd`, press Enter (a terminal opens in that folder).
   (VS Code: *File -> Open Folder -> rag-doc-qa*, then *Terminal -> New Terminal*.)

### Step 2 - Fast way (one script)
* **Windows:** type `setup_windows.bat` and press Enter.
* **macOS/Linux:** `bash setup_mac_linux.sh`

This does steps 3-5 below automatically. If it works, jump to **Step 6**. If anything fails, do the manual steps.

### Step 3 - Create a virtual environment (manual way)
```
python -m venv venv
venv\Scripts\activate          (Windows)
source venv/bin/activate       (macOS / Linux)
```
You should see `(venv)` at the start of the line.

### Step 4 - Install libraries
```
pip install -r requirements.txt
```
(takes 3-10 minutes the first time; chromadb and streamlit are big).

### Step 5 - Build the sample corpus and index it
```
python scripts/build_corpus.py                       # makes 50 documents + 1,000 Q&A pairs in data/
python scripts/ingest_corpus.py --embedder hashing   # indexes them into ChromaDB (a few seconds)
```
The `data/corpus` folder already contains the 50 files, so `build_corpus.py` is only needed if you edit the texts.

### Step 6 - Start the app
```
streamlit run app.py
```
The browser opens at http://localhost:8501 . (If not, open that link yourself.)

### Step 7 - Use the app
1. **Sidebar:** keep *extractive (no LLM, offline)* and *hashing* for the first test (no key, no internet).
2. **Documents tab:** click *Index sample corpus* (or upload your own PDF/Word/HTML/MD/TXT and click *Index uploaded files*).
3. **Ask tab:** try "When did the Constitution of India come into force?" -> read the answer, scores and **source chunks**.
4. Try "What is the price of Bitcoin today?" -> the system should refuse (not in documents).
5. **Analytics tab:** shows your questions, scores, feedback. **Evaluation tab:** shows the saved test results.

### Step 8 - Use a real LLM (better, natural answers) - choose ONE
| Option | Cost | What to do |
|--------|------|------------|
| **Groq** | free tier | Get a key at https://console.groq.com -> in the sidebar choose `groq`, paste the key. |
| **Gemini** | free tier | Key from https://aistudio.google.com -> choose `gemini`, paste key. |
| **Ollama (offline)** | free | Install https://ollama.com, run `ollama pull llama3.2` (needs about 4-8 GB RAM), choose `ollama (local)`. |
| **OpenAI / Claude** | paid | Choose `openai` or `anthropic`, paste the key. |

Model names change over time. If you get a "model not found" error, type a current model name in the *Model name* box.
You can also put keys in a file: copy `.env.example` to `.env` and fill your key (never share `.env`).

### Step 9 - Better search with MiniLM (optional, recommended for the final demo)
```
pip install sentence-transformers
```
In the sidebar choose embedding = `minilm` (downloads about 90 MB once), then *Index sample corpus* again.
(MiniLM and hashing keep **separate** indexes, so nothing is overwritten.)

### Step 10 - Run the evaluation (numbers for your report, Part B/C)
```
python scripts/evaluate.py --embedder hashing --k 4 --n 1000 --answers
python scripts/ablation.py
python scripts/refusal_test.py
# with a real LLM and MiniLM (fill the [FILL IN] cells of report section 6.3):
python scripts/evaluate.py --embedder minilm --provider groq --api-key YOUR_KEY --n 100 --reindex
```
Results are saved in `data/` (`eval_summary_*.json`, `eval_results.csv`, `ablation_results.csv`, `refusal_test.csv`).

### Step 11 - EDA notebook
```
jupyter notebook notebooks/EDA.ipynb
```
Click *Run -> Run All Cells*. It has 8 visualisations and is already executed once (outputs are saved inside).
Figures go to `figures/`.

### Step 12 - Run the tests
```
python tests/test_rag.py       (or: python -m pytest tests -q)
```
Expected: `11 of 11 tests passed`.

---------------------------------------------------------------------

## PART 2 - Deliverables checklist (from the Project Brief)

| Deliverable | Where it is | What YOU still do |
|-------------|-------------|-------------------|
| Working prototype | `app.py`, `rag/` | Run it, take screenshots |
| Project report (min 8 pages) | `report/Project_Report_RAG_Document_QA.docx` (13 pages) | Fill name, roll no., college, mentor, `[FILL IN]` cells, screenshots, tick and **sign** the declaration (section 9) |
| EDA notebook, 3+ visualisations | `notebooks/EDA.ipynb` (8 figures), `figures/` | Re-run once on your laptop |
| `dataset.csv` cleaned | `data/dataset.csv` (5,000 rows), `data/dataset_features.csv`, `data/cleaning_report.json` | - |
| `data_dictionary.csv` | `data_dictionary.csv` (all 13 columns described) | - |
| Demo video 1-2 min | script: `docs/DEMO_SCRIPT.md` | Record it |
| Signed ethical declaration | report section 9 | Sign and date |
| Dataset-generation prompt: 50 documents + 1,000 Q&A | `data/corpus/` (50 files), `data/qa_pairs.csv` (1,000 rows) | Optional: `scripts/generate_qa_with_llm.py` for independent LLM-written questions |
| Dataset 500 -> 5,000 rows, same columns | `scripts/expand_dataset.py` -> `data/dataset.csv` | - |
| Ethical note (show source chunk, no legal/medical use) | App banner, source expanders, refusal rule | Mention it in your demo |

## PART 3 - Be honest about these points (mentors may ask)
1. **4,500 of the 5,000 rows in `dataset.csv` are synthetic** (generated from the original 500 rows' ranges). The original columns are uncorrelated, so EDA shows no relationship; real behaviour is measured on the 50-document corpus.
2. The 1,000 Q&A pairs come from 400 hand-written facts (plus wording variants). Scores are therefore optimistic; the report also gives a harder subset. Use `generate_qa_with_llm.py` for more independent questions.
3. The corpus is written in our own words from public facts. **Verify facts** (especially government schemes, which change) against official sources before using them anywhere important.
4. The offline `hashing` embedder matches words, not meaning. Use `minilm` for semantic search and re-measure.
5. Disclose AI help in Appendix A of the report and confirm the statement is true for you.

## Troubleshooting
| Problem | Fix |
|---------|-----|
| `python` not recognised | Reinstall Python and tick *Add Python to PATH*, or use `py` instead of `python`. |
| `ModuleNotFoundError: No module named 'rag'` | Run commands from inside the `rag-doc-qa` folder. |
| `pip install chromadb` fails on Windows (build tools error) | Update pip: `python -m pip install --upgrade pip`; use Python 3.11/3.12; or install "Microsoft C++ Build Tools". |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| "The index is empty" | Documents tab -> *Index sample corpus*. |
| `429` / rate-limit error | Wait a minute, lower top-k, or switch provider. |
| `model not found` | Type a current model name from the provider's website in *Model name*. |
| Answers look weak | Use `minilm` embeddings and a real LLM; increase chunk size to 800 and top-k to 4-5. |
| Want a clean start | Delete the `chroma_db/` and `qa_log.db` files, then index again. |

## Project structure
```
rag-doc-qa/
  app.py                  Streamlit app (Ask, Documents, Analytics, Evaluation)
  rag/                    loaders.py  embeddings.py  store.py  llm.py  pipeline.py  logdb.py
  scripts/                build_corpus, ingest_corpus, evaluate, ablation, refusal_test,
                          expand_dataset, clean_dataset, make_data_dictionary,
                          build_notebook, build_report, draw_diagrams, generate_qa_with_llm
  data/                   corpus/ (50 files)  qa_pairs.csv  dataset.csv  dataset_features.csv ...
  data_dictionary.csv     notebooks/EDA.ipynb     figures/     tests/     docs/     report/
```

## How it works (one paragraph)
Load -> split into ~500-character chunks (LangChain splitter) -> embed (MiniLM / OpenAI / hashing) -> store in ChromaDB (cosine) ->
embed the question -> retrieve top-k -> if the best score is below the threshold, refuse -> else the LLM answers **only** from the numbered
passages -> compute a grounding score -> show answer + sources + warnings -> log the question and your feedback.
