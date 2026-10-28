"""Expand dataset_original_500.csv (500 rows) to 5,000 rows with the SAME 7 columns.
New rows are SYNTHETIC: each column is resampled from the distribution of the original file, doc_type is kept consistent with the
real corpus manifest, chunk_ids continue the sequence (CHK00501 ...).   python scripts/expand_dataset.py
"""
import os, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
rng = np.random.default_rng(42)
orig = pd.read_csv(os.path.join(ROOT, "data", "dataset_original_500.csv"))
man = pd.read_csv(os.path.join(ROOT, "data", "corpus_manifest.csv")).set_index("doc_id")
N_NEW = 5000 - len(orig)

doc_ids = rng.choice(sorted(man.index), size=N_NEW)
new = pd.DataFrame({
    "doc_id": doc_ids,
    "doc_type": [man.loc[d, "doc_type"] for d in doc_ids],              # same format as the real file of that document
    "chunk_id": [f"CHK{n:05d}" for n in range(len(orig) + 1, 5001)],
    "word_count": rng.integers(50, 500, size=N_NEW),                      # original range 50-499 words
    "retrieval_score": np.round(rng.uniform(0.50, 0.999, size=N_NEW), 4), # original range 0.5008-0.9989
    "query_type": rng.choice(orig.query_type, size=N_NEW),               # resample original class proportions
    "answer_quality_score": np.round(rng.uniform(0.60, 1.0, size=N_NEW), 3),
})
full = pd.concat([orig, new], ignore_index=True)
full.to_csv(os.path.join(ROOT, "data", "dataset_raw_5000.csv"), index=False)
print("rows:", len(full), "| new synthetic rows:", N_NEW)
