"""RAG-Based Document Q&A System - Streamlit app.   Run:  streamlit run app.py"""
import os
import pandas as pd
import streamlit as st
from rag.pipeline import RAGPipeline
from rag.llm import PROVIDERS
from rag.embeddings import DEFAULT_THRESHOLD
from rag import logdb

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "data", "corpus")
st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Answer generator (LLM)", list(PROVIDERS), index=0,
                            help="'extractive' needs no internet or key. Others need an API key (Ollama needs a local server).")
    model = st.text_input("Model name", value=PROVIDERS[provider]["model"], disabled=provider.startswith("extractive"))
    api_key = st.text_input("API key", type="password", help="Kept only in this browser session. Can also be set in .env / environment.")
    embedder = st.selectbox("Embedding model", ["hashing", "minilm", "openai"], index=0,
                            help="hashing = offline & fast (keyword-like). minilm = better meaning search (downloads ~90 MB once). openai = needs key.")
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (characters)", 200, 1500, 500, 50)
    overlap = st.slider("Chunk overlap", 0, 300, 80, 10)
    st.subheader("Retrieval")
    top_k = st.slider("Top-k chunks", 1, 8, 4)
    threshold = st.slider("Minimum retrieval score", 0.0, 0.9, float(DEFAULT_THRESHOLD[embedder]), 0.01,
                          help="If the best match is below this, the system refuses to answer instead of guessing.")
    st.caption("⚠️ Answers can be wrong. Always read the source chunks. Not for legal or medical advice without expert review.")

if "dotenv" not in st.session_state:
    try:
        from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass
    st.session_state["dotenv"] = True


@st.cache_resource(show_spinner="Loading models and vector store ...")
def get_pipe(provider, model, api_key, embedder):
    return RAGPipeline(os.path.join(ROOT, "chroma_db"), embedder, provider, model or None, api_key or None)

try:
    pipe = get_pipe(provider, model, api_key, embedder)
except Exception as e:
    st.error(f"Could not start the pipeline: {e}"); st.stop()

st.title("📚 RAG-Based Document Q&A System")
st.warning("**RAG systems can hallucinate.** Every answer is shown with its source chunks - please verify. "
           "Do not use for legal or medical advice without expert review.", icon="⚠️")
tab_ask, tab_docs, tab_an, tab_eval = st.tabs(["💬 Ask", "📁 Documents", "📊 Analytics", "🧪 Evaluation"])

# ------------------------------------------------------------------ documents
with tab_docs:
    st.subheader("Index documents")
    c1, c2 = st.columns(2)
    with c1:
        files = st.file_uploader("Upload PDF / Word / HTML / Markdown / TXT", type=["pdf", "docx", "html", "htm", "md", "txt"], accept_multiple_files=True)
        if st.button("Index uploaded files", disabled=not files):
            bar = st.progress(0.0)
            for i, f in enumerate(files):
                try:
                    n = pipe.index_file(f, name=f.name, chunk_size=chunk_size, overlap=overlap)
                    st.success(f"{f.name}: {n} chunks indexed")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
                bar.progress((i + 1) / len(files))
    with c2:
        st.write("Or load the bundled sample corpus (50 documents: Indian Constitution, government schemes, technical manuals).")
        if st.button("Index sample corpus (replaces current index)"):
            from scripts.ingest_corpus import ingest
            with st.spinner("Indexing 50 documents ..."):
                n = ingest(pipe, CORPUS, chunk_size, overlap)
            st.success(f"Indexed {n} chunks.")
        if st.button("🗑️ Clear index"):
            pipe.store.reset(); st.info("Index cleared.")
    st.metric("Chunks in vector store", pipe.store.count())
    summ = pipe.store.doc_summary()
    if summ: st.dataframe(pd.DataFrame(summ), width="stretch", hide_index=True)

# ------------------------------------------------------------------ ask
with tab_ask:
    st.subheader("Ask a question about your documents")
    examples = ["When did the Constitution of India come into force?", "How much money do farmers get under PM-KISAN?",
                "What is the difference between a list and a tuple in Python?", "Which command creates a virtual environment?"]
    ex = st.selectbox("Try an example (optional)", [""] + examples)
    q = st.text_input("Your question", value=ex)
    if st.button("Get answer", type="primary", disabled=not q.strip()):
        if pipe.store.count() == 0:
            st.error("The index is empty. Go to the Documents tab first.")
        else:
            with st.spinner("Searching and generating ..."):
                try:
                    ans = pipe.ask(q.strip(), top_k, threshold)
                    st.session_state["last"] = ans
                    st.session_state["row"] = logdb.log(ans, provider, embedder, os.path.join(ROOT, "qa_log.db"))
                except Exception as e:
                    st.error(f"Generation failed: {e}"); st.stop()
    ans = st.session_state.get("last")
    if ans:
        st.markdown("### Answer")
        (st.info if ans.abstained else st.success)(ans.answer)
        if ans.warning: st.warning(ans.warning)
        m1, m2, m3 = st.columns(3)
        m1.metric("Top retrieval score", f"{ans.top_score:.2f}")
        m2.metric("Grounding score", f"{ans.grounding:.2f}", help="Share of answer words found in the sources. Low = possible hallucination.")
        m3.metric("Chunks used", len(ans.sources))
        st.markdown("### Source chunks (always verify against these)")
        for i, s in enumerate(ans.sources, 1):
            with st.expander(f"[{i}] {s['title']}  |  {s['doc_id']} ({s['doc_type']})  |  score {s['score']:.2f}", expanded=(i == 1)):
                st.write(s["text"])
        f1, f2, _ = st.columns([1, 1, 6])
        if f1.button("👍 Helpful"): logdb.set_feedback(st.session_state["row"], "up", os.path.join(ROOT, "qa_log.db")); st.toast("Thanks!")
        if f2.button("👎 Wrong"): logdb.set_feedback(st.session_state["row"], "down", os.path.join(ROOT, "qa_log.db")); st.toast("Noted.")

# ------------------------------------------------------------------ analytics
with tab_an:
    st.subheader("Usage analytics")
    log = logdb.read(os.path.join(ROOT, "qa_log.db"))
    if log.empty:
        st.info("No questions asked yet.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Questions", len(log)); b.metric("Avg retrieval score", f"{log.top_score.mean():.2f}")
        c.metric("Avg grounding", f"{log.grounding.mean():.2f}"); d.metric("Refused (not found)", f"{log.abstained.mean()*100:.0f}%")
        st.line_chart(log.sort_values("id").set_index("id")[["top_score", "grounding"]])
        st.dataframe(log[["ts", "question", "answer", "top_score", "grounding", "abstained", "feedback"]], width="stretch", hide_index=True)
        st.download_button("Download log (CSV)", log.to_csv(index=False), "qa_log.csv")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.subheader("Retrieval evaluation on the 1,000 Q&A pairs")
    st.caption("Run `python scripts/evaluate.py --embedder hashing` in a terminal to refresh these numbers (see README).")
    import json, glob
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "eval_summary_*.json"))):
        j = json.load(open(p))
        st.markdown(f"**Embedder: {j['embedder']}** | provider: {j['provider']} | k={j['k']} | n={j['n_questions']}")
        cols = st.columns(4)
        cols[0].metric("Doc hit@k", j["doc_hit_at_k"]); cols[1].metric("Top-1 accuracy", j["top1_accuracy"])
        cols[2].metric("MRR", j["mrr"]); cols[3].metric("Evidence recall@k", j["evidence_recall_at_k"])
    abl = os.path.join(ROOT, "data", "ablation_results.csv")
    if os.path.exists(abl):
        st.markdown("**Chunk-size / top-k ablation**"); st.dataframe(pd.read_csv(abl), width="stretch", hide_index=True)
