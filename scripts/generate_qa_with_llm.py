"""OPTIONAL: create NEW, independent Q&A pairs with an LLM from the corpus chunks (harder evaluation than the template pairs).
   python scripts/generate_qa_with_llm.py --provider groq --api-key YOUR_KEY --per-doc 5
Output: data/qa_pairs_llm.csv  (same columns as qa_pairs.csv).  Review a sample by hand: LLM-written questions can also contain mistakes."""
import os, sys, json, re, argparse, csv
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from rag.llm import LLM
from rag.loaders import load_file
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--provider", default="groq"); ap.add_argument("--model", default=None)
ap.add_argument("--api-key", default=None); ap.add_argument("--per-doc", type=int, default=5); a = ap.parse_args()
llm = LLM(a.provider, a.model, a.api_key)
man = pd.read_csv(os.path.join(ROOT, "data", "corpus_manifest.csv"))
SYS = "You write exam-style questions. Reply with JSON only."
rows = []
for _, m in man.iterrows():
    text, _ = load_file(os.path.join(ROOT, "data", "corpus", m.file_name))
    user = (f"Read the document and write {a.per_doc} DIFFERENT questions that can be answered from it, in your own wording (do not copy sentences; "
            f"mix factual, analytical, comparative and procedural). Return JSON: [{{\"question\":..., \"answer\":..., \"query_type\":..., \"evidence\":<exact supporting sentence>}}].\n\nDocument:\n{text}")
    try:
        raw = llm.complete(SYS, user); items = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
    except Exception as e:
        print("skip", m.doc_id, e); continue
    for it in items:
        rows.append(dict(qa_id=f"LQ{len(rows)+1:04d}", doc_id=m.doc_id, doc_title=m.title, source=m.source, query_type=it.get("query_type", "factual"),
                         question=it["question"], answer=it["answer"], evidence=it.get("evidence", "")))
    print(m.doc_id, "ok", len(rows))
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "data", "qa_pairs_llm.csv"), index=False); print("saved", len(rows), "pairs")
