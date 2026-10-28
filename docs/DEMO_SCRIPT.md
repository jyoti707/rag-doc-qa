# Demo script (1-2 minutes) - record with any screen recorder (Windows: Win+G, or OBS Studio)

**Before recording:** `streamlit run app.py`, set provider = extractive (or your LLM), click *Index sample corpus* once.

| Time | What to show | What to say (simple) |
|------|--------------|----------------------|
| 0:00-0:10 | App home page with the red warning banner | "This is my RAG-Based Document Q&A System. It answers only from documents and always shows the source. It can hallucinate, so there is a warning." |
| 0:10-0:30 | Documents tab: upload one PDF/Word, click *Index uploaded files*; show chunk count table | "I upload PDF, Word, HTML, Markdown or TXT. The text is split into chunks, converted to embeddings and stored in ChromaDB." |
| 0:30-1:00 | Ask tab: "When did the Constitution of India come into force?" -> Get answer; open source chunk [1] | "The system retrieves the top chunks, shows the answer with citation, retrieval score and grounding score. I can verify the sentence in the source." |
| 1:00-1:20 | Ask: "What is the price of Bitcoin today?" | "This is not in my documents, so the system refuses instead of guessing." |
| 1:20-1:40 | Analytics tab + Evaluation tab | "Every question is logged. On 1,000 test questions, document hit@4 is about 99% with the offline embedder." |
| 1:40-2:00 | Slide/README: limitations | "Limits: no OCR, text only, answers need human review. Not for legal or medical advice." |
