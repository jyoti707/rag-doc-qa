"""Index every file in data/corpus into ChromaDB.   python scripts/ingest_corpus.py --embedder hashing"""
import os, sys, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.pipeline import RAGPipeline

def ingest(pipe, folder, chunk_size=500, overlap=80, reset=True):
    if reset: pipe.store.reset()
    n = 0
    for f in sorted(os.listdir(folder)):
        n += pipe.index_file(os.path.join(folder, f), name=f, chunk_size=chunk_size, overlap=overlap)
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "minilm", "openai"])
    ap.add_argument("--chunk-size", type=int, default=500); ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--persist", default="chroma_db")
    a = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    t = time.time()
    pipe = RAGPipeline(os.path.join(root, a.persist), a.embedder)
    n = ingest(pipe, os.path.join(root, "data", "corpus"), a.chunk_size, a.overlap)
    print(f"Indexed {n} chunks from corpus using '{a.embedder}' embeddings in {time.time()-t:.1f}s")
