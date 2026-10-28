"""Chunk-size / top-k ablation + 'hard' subset (questions that do not mention the document title).  python scripts/ablation.py"""
import os, sys, json
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from rag.pipeline import RAGPipeline
from scripts.ingest_corpus import ingest
from scripts.evaluate import evidence_hit

qa = pd.read_csv(os.path.join(ROOT, "data", "qa_pairs.csv"))
hard = qa[~qa.question.str.startswith(("Regarding", "Using the documents", "Give a short summary"))]
res = []
for emb in ["hashing"]:
    for cs, ov in [(300, 50), (500, 80), (800, 120)]:
        pipe = RAGPipeline(os.path.join(ROOT, "chroma_db_abl"), emb)
        nchunks = ingest(pipe, os.path.join(ROOT, "data", "corpus"), cs, ov)
        for name, sub in [("all_1000", qa), ("hard_no_title", hard)]:
            for k in [1, 3, 5]:
                dh = eh = t1 = rr = 0
                for _, r in sub.iterrows():
                    src = pipe.retrieve(r.question, k); ids = [s["doc_id"] for s in src]
                    rank = ids.index(r.doc_id) + 1 if r.doc_id in ids else 0
                    dh += rank > 0; rr += (1 / rank if rank else 0); eh += evidence_hit(r.evidence, src)
                n = len(sub)
                res.append(dict(embedder=emb, chunk_size=cs, overlap=ov, chunks=nchunks, subset=name, n=n, k=k,
                                doc_hit=round(dh / n, 3), mrr=round(rr / n, 3), evidence_recall=round(eh / n, 3)))
df = pd.DataFrame(res); df.to_csv(os.path.join(ROOT, "data", "ablation_results.csv"), index=False)
print(df.to_string(index=False))
