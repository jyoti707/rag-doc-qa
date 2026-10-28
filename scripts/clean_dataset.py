"""Clean + pre-process dataset_raw_5000.csv -> data/dataset.csv (+ dataset_features.csv with derived columns, + cleaning_report.json)."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = pd.read_csv(os.path.join(ROOT, "data", "dataset_raw_5000.csv"))
man = pd.read_csv(os.path.join(ROOT, "data", "corpus_manifest.csv")).set_index("doc_id")
rep = {"rows_in": len(df)}

# 1. missing values & duplicates
rep["missing_values"] = int(df.isna().sum().sum()); df = df.dropna()
rep["duplicate_rows"] = int(df.duplicated().sum()); df = df.drop_duplicates()
rep["duplicate_chunk_ids"] = int(df.chunk_id.duplicated().sum()); df = df.drop_duplicates("chunk_id")

# 2. tidy text columns
for c in ["doc_id", "doc_type", "chunk_id", "query_type"]: df[c] = df[c].astype(str).str.strip()
df["doc_id"] = df.doc_id.str.upper(); df["chunk_id"] = df.chunk_id.str.upper(); df["query_type"] = df.query_type.str.lower()

# 3. referential integrity: every doc_id must exist in the corpus and have ONE file format
rep["unknown_doc_ids"] = int((~df.doc_id.isin(man.index)).sum()); df = df[df.doc_id.isin(man.index)]
rep["doc_ids_with_multiple_types_before"] = int((df.groupby("doc_id").doc_type.nunique() > 1).sum())
fixed = df.doc_type != df.doc_id.map(man.doc_type)
rep["doc_type_rows_corrected"] = int(fixed.sum())
df["doc_type"] = df.doc_id.map(man.doc_type)

# 4. range checks
bad = ~df.word_count.between(1, 5000) | ~df.retrieval_score.between(0, 1) | ~df.answer_quality_score.between(0, 1)
rep["out_of_range_rows_dropped"] = int(bad.sum()); df = df[~bad]
df["word_count"] = df.word_count.astype(int)
df = df.sort_values("chunk_id").reset_index(drop=True)
rep["rows_out"] = len(df)
df.to_csv(os.path.join(ROOT, "data", "dataset.csv"), index=False)

# 5. derived features for EDA / modelling
f = df.copy()
f["length_bucket"] = pd.cut(f.word_count, [0, 150, 300, 450, 10**6], labels=["short(<=150)", "medium(151-300)", "long(301-450)", "very_long(>450)"])
f["retrieval_band"] = pd.cut(f.retrieval_score, [0, 0.6, 0.75, 0.9, 1.0], labels=["low(<0.6)", "fair(0.6-0.75)", "good(0.75-0.9)", "excellent(>0.9)"])
f["quality_band"] = pd.cut(f.answer_quality_score, [0, 0.7, 0.85, 1.0], labels=["poor(<0.7)", "ok(0.7-0.85)", "good(>0.85)"])
f["combined_score"] = (0.5 * f.retrieval_score + 0.5 * f.answer_quality_score).round(4)
f["hallucination_risk"] = ((f.retrieval_score < 0.6) | (f.answer_quality_score < 0.65)).astype(int)
f["category"] = f.doc_id.map(lambda d: "Constitution" if int(d[3:]) <= 18 else ("Govt schemes" if int(d[3:]) <= 34 else "Technical manual"))
f.to_csv(os.path.join(ROOT, "data", "dataset_features.csv"), index=False)
rep["hallucination_risk_rate"] = round(float(f.hallucination_risk.mean()), 4)
json.dump(rep, open(os.path.join(ROOT, "data", "cleaning_report.json"), "w"), indent=2); print(json.dumps(rep, indent=2))
