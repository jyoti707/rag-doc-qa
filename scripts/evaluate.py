"""Evaluate retrieval (and optionally answers) on data/qa_pairs.csv.
   python scripts/evaluate.py --embedder hashing --k 4 --n 1000
   python scripts/evaluate.py --embedder minilm --provider groq --n 100      (adds answer metrics; needs API key)
Outputs: data/eval_results.csv and data/eval_summary.json
"""
import os, sys, json, argparse, time
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from rag.pipeline import RAGPipeline, words
from scripts.ingest_corpus import ingest


def evidence_hit(evidence, chunks):
    ew = set(words(evidence))
    return any(len(ew & set(words(c["text"]))) / max(1, len(ew)) >= 0.8 for c in chunks)


def answer_recall(gold, pred):
    gw = set(words(gold))
    return len(gw & set(words(pred))) / max(1, len(gw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hashing"); ap.add_argument("--provider", default="extractive (no LLM, offline)")
    ap.add_argument("--model", default=None); ap.add_argument("--api-key", default=None)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--chunk-size", type=int, default=500); ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--reindex", action="store_true"); ap.add_argument("--answers", action="store_true", help="also generate answers (needed for extractive mode)")
    a = ap.parse_args()
    pipe = RAGPipeline(os.path.join(ROOT, "chroma_db"), a.embedder, a.provider, a.model, a.api_key)
    if a.reindex or pipe.store.count() == 0:
        print("Indexing corpus ...", ingest(pipe, os.path.join(ROOT, "data", "corpus"), a.chunk_size, a.overlap), "chunks")
    qa = pd.read_csv(os.path.join(ROOT, "data", "qa_pairs.csv")).sample(frac=1, random_state=42).head(a.n)
    use_llm = pipe.llm is not None or a.answers
    rows, t0 = [], time.time()
    for _, r in qa.iterrows():
        src = pipe.retrieve(r.question, a.k)
        ids = [s["doc_id"] for s in src]
        rank = ids.index(r.doc_id) + 1 if r.doc_id in ids else 0
        row = dict(qa_id=r.qa_id, doc_id=r.doc_id, query_type=r.query_type, top_score=src[0]["score"] if src else 0,
                   doc_rank=rank, doc_hit=int(rank > 0), top1_hit=int(rank == 1), evidence_hit=int(evidence_hit(r.evidence, src)))
        if use_llm:
            ans = pipe.ask(r.question, a.k)
            row.update(grounding=ans.grounding, abstained=int(ans.abstained), answer_recall=round(answer_recall(r.answer, ans.answer), 3), answer=ans.answer)
            time.sleep(0.5 if pipe.llm is not None else 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["rr"] = df.doc_rank.apply(lambda x: 1 / x if x else 0)
    summ = dict(embedder=a.embedder, provider=a.provider, k=a.k, n_questions=len(df), chunk_size=a.chunk_size, overlap=a.overlap,
                doc_hit_at_k=round(df.doc_hit.mean(), 4), top1_accuracy=round(df.top1_hit.mean(), 4), mrr=round(df.rr.mean(), 4),
                evidence_recall_at_k=round(df.evidence_hit.mean(), 4), avg_top_score=round(df.top_score.mean(), 4),
                seconds_per_query=round((time.time() - t0) / len(df), 3),
                by_query_type=df.merge(qa[["qa_id"]], on="qa_id").groupby("query_type")[["doc_hit", "top1_hit", "evidence_hit"]].mean().round(3).to_dict("index"))
    if use_llm:
        summ.update(avg_grounding=round(df.grounding.mean(), 3), abstain_rate=round(df.abstained.mean(), 3), avg_answer_recall=round(df.answer_recall.mean(), 3))
    df.to_csv(os.path.join(ROOT, "data", "eval_results.csv"), index=False)
    json.dump(summ, open(os.path.join(ROOT, "data", f"eval_summary_{a.embedder}.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
