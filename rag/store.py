"""ChromaDB vector store wrapper (cosine similarity, persisted on disk)."""
import chromadb


class VectorStore:
    def __init__(self, persist_dir="chroma_db", collection="rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.name = collection
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids, texts, metadatas, embeddings):
        for i in range(0, len(ids), 500):
            self.col.upsert(ids=ids[i:i+500], documents=texts[i:i+500], metadatas=metadatas[i:i+500], embeddings=embeddings[i:i+500])

    def query(self, embedding, k=4):
        n = self.col.count()
        if n == 0: return []
        r = self.col.query(query_embeddings=[embedding], n_results=min(k, n))
        out = []
        for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            out.append(dict(chunk_id=cid, text=doc, score=round(1.0 - float(dist), 4), **meta))
        return out

    def count(self): return self.col.count()

    def doc_summary(self):
        got = self.col.get(include=["metadatas"])
        summary = {}
        for m in got["metadatas"]:
            key = (m.get("doc_id"), m.get("title"), m.get("doc_type"))
            summary[key] = summary.get(key, 0) + 1
        return [dict(doc_id=k[0], title=k[1], doc_type=k[2], chunks=v) for k, v in sorted(summary.items(), key=lambda x: str(x[0][0]))]

    def reset(self):
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})
