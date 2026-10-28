"""Embedding back-ends. 'hashing' works fully offline (no download). 'minilm' and 'openai' give better semantic search."""
import os
import numpy as np

DEFAULT_THRESHOLD = {"hashing": 0.15, "minilm": 0.30, "openai": 0.30}


class HashingEmbedder:
    name = "hashing"
    def __init__(self, dim=4096):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2",
                                     stop_words="english", ngram_range=(1, 2), lowercase=True)
    def embed(self, texts):
        return self.vec.transform(texts).toarray().astype("float32").tolist()


class MiniLMEmbedder:
    name = "minilm"
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)
    def embed(self, texts):
        return self.m.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


class OpenAIEmbedder:
    name = "openai"
    def __init__(self, api_key=None, model="text-embedding-3-small"):
        from openai import OpenAI
        self.c = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY")); self.model = model
    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            r = self.c.embeddings.create(model=self.model, input=texts[i:i + 64])
            out += [d.embedding for d in r.data]
        return out


def get_embedder(name="hashing", api_key=None):
    if name == "hashing": return HashingEmbedder()
    if name == "minilm": return MiniLMEmbedder()
    if name == "openai": return OpenAIEmbedder(api_key)
    raise ValueError(f"Unknown embedder: {name}")
