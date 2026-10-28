"""RAG pipeline: load -> chunk -> embed -> store -> retrieve -> generate -> guard."""
import re, os
from dataclasses import dataclass, field
from .loaders import load_file
from .embeddings import get_embedder, DEFAULT_THRESHOLD
from .store import VectorStore
from .llm import LLM, SYSTEM_PROMPT

NOT_FOUND = "I could not find this in the documents."
STOP = set("the a an of to in on and or is are was were be by for with as at from that this it its which what who how when where why does do did can".split())


def split_text(text, chunk_size=500, overlap=80):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        sp = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
        return sp.split_text(text)
    except ImportError:  # simple fallback
        out, i = [], 0
        while i < len(text):
            out.append(text[i:i + chunk_size]); i += chunk_size - overlap
        return out


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]


def grounding_score(answer, context):
    """Share of the answer's content words that also occur in the retrieved context (0-1). Low = possible hallucination."""
    aw = set(words(re.sub(r"\[\d+\]", " ", answer)))
    if not aw: return 0.0
    cw = set(words(context))
    return round(len(aw & cw) / len(aw), 3)


@dataclass
class Answer:
    question: str
    answer: str
    sources: list = field(default_factory=list)
    top_score: float = 0.0
    grounding: float = 0.0
    abstained: bool = False
    warning: str = ""


class RAGPipeline:
    def __init__(self, persist_dir="chroma_db", embedder="hashing", provider="extractive (no LLM, offline)",
                 model=None, api_key=None, embed_key=None):
        self.embedder = get_embedder(embedder, embed_key or api_key)
        self.store = VectorStore(persist_dir, collection=f"rag_{self.embedder.name}")
        self.llm = None if provider.startswith("extractive") else LLM(provider, model, api_key)
        self.provider = provider

    # ---------- indexing ----------
    def index_text(self, text, doc_id, title, doc_type="TXT", chunk_size=500, overlap=80):
        chunks = [c.strip() for c in split_text(text, chunk_size, overlap) if len(c.strip()) > 20]
        if not chunks: return 0
        ids = [f"{doc_id}_C{i:03d}" for i in range(len(chunks))]
        metas = [dict(doc_id=doc_id, title=title, doc_type=doc_type, chunk_no=i) for i in range(len(chunks))]
        self.store.add(ids, chunks, metas, self.embedder.embed(chunks))
        return len(chunks)

    def index_file(self, file, name=None, doc_id=None, chunk_size=500, overlap=80):
        text, fname = load_file(file, name)
        ext = os.path.splitext(fname)[1].lower()
        dtype = {".pdf": "PDF", ".docx": "Word", ".html": "HTML", ".htm": "HTML", ".md": "Markdown", ".txt": "TXT"}.get(ext, "TXT")
        title = text.splitlines()[0][:80] if text else fname
        m = re.match(r"(DOC\d+)_", fname)
        doc_id = doc_id or (m.group(1) if m else os.path.splitext(fname)[0])
        return self.index_text(text, doc_id, title, dtype, chunk_size, overlap)

    # ---------- retrieval ----------
    def retrieve(self, question, k=4):
        return self.store.query(self.embedder.embed([question])[0], k)

    # ---------- answering ----------
    def ask(self, question, k=4, threshold=None):
        threshold = DEFAULT_THRESHOLD[self.embedder.name] if threshold is None else threshold
        src = self.retrieve(question, k)
        if not src:
            return Answer(question, "No documents are indexed yet.", abstained=True, warning="Index some documents first.")
        top = src[0]["score"]
        if top < threshold:
            return Answer(question, NOT_FOUND, src, top, 0.0, True,
                          "Best match score is below the threshold, so no answer was generated. Nearest passages are shown for checking.")
        context = "\n\n".join(f"[{i+1}] ({s['title']}) {s['text']}" for i, s in enumerate(src))
        if self.llm is None:
            ans = self._extractive(question, src)
        else:
            user = f"Context passages:\n{context}\n\nQuestion: {question}\nAnswer:"
            ans = self.llm.complete(SYSTEM_PROMPT, user)
        g = grounding_score(ans, context)
        warn = ""
        if ans.strip().startswith("I could not find"):
            return Answer(question, ans, src, top, g, True, "The model could not find the answer in the retrieved passages.")
        if g < 0.6: warn = "Low grounding: parts of this answer may not be supported by the sources. Verify before use."
        elif top < threshold + 0.1: warn = "Weak retrieval match. Please check the source passages."
        return Answer(question, ans, src, top, g, False, warn)

    def _extractive(self, question, src, n=2):
        qw = set(words(question))
        sents = []
        for si, s in enumerate(src):
            for t in re.split(r"(?<=[.!?])\s+", s["text"]):
                ov = len(qw & set(words(t)))
                if len(t) > 25 and ov >= 1 and "Source:" not in t and "Note: This is a simplified" not in t:
                    sents.append((ov + s["score"], si, t.strip()))
        sents.sort(key=lambda x: -x[0])
        best = sents[:n]
        return " ".join(f"{t} [{si+1}]" for _, si, t in best) if best else NOT_FOUND
