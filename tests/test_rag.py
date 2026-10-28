"""Functional tests.  Run:  python -m pytest tests -q     (or: python tests/test_rag.py)"""
import os, sys, tempfile, sqlite3
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from rag.loaders import load_file
from rag.pipeline import RAGPipeline, split_text, grounding_score, NOT_FOUND
from rag import logdb
CORP = os.path.join(ROOT, "data", "corpus")

def test_loaders_all_formats():
    for ext in ["txt", "md", "html", "docx", "pdf"]:
        f = [x for x in sorted(os.listdir(CORP)) if x.endswith("." + ext)][0]
        text, name = load_file(os.path.join(CORP, f))
        assert len(text) > 300 and "<p>" not in text, (ext, len(text))

def test_html_scripts_removed():
    p = os.path.join(tempfile.mkdtemp(), "a.html"); open(p, "w").write("<html><script>var x=1;</script><p>Hello world</p></html>")
    assert "var x" not in load_file(p)[0] and "Hello world" in load_file(p)[0]

def test_unsupported_type():
    try: load_file(os.path.join(tempfile.mkdtemp(), "a.xyz"), "a.xyz"); assert False
    except ValueError: pass

def test_chunking_overlap_and_size():
    text = " ".join(f"word{i}" for i in range(400)); ch = split_text(text, 300, 50)
    assert all(len(c) <= 300 for c in ch) and len(ch) > 5
    assert set(ch[0].split()[-3:]) & set(ch[1].split()[:12])           # neighbouring chunks overlap

def test_grounding_score():
    ctx = "The Constitution came into force on 26 January 1950."
    assert grounding_score("It came into force on 26 January 1950", ctx) > 0.9
    assert grounding_score("The treaty was signed by Napoleon in Paris", ctx) < 0.3

def _pipe():
    p = RAGPipeline(tempfile.mkdtemp(), "hashing")
    for f in sorted(os.listdir(CORP))[:10]: p.index_file(os.path.join(CORP, f), name=f)
    return p

def test_index_retrieve_answer_with_sources():
    p = _pipe(); assert p.store.count() > 20
    a = p.ask("When did the Constitution of India come into force?")
    assert "1950" in a.answer and a.sources and a.sources[0]["doc_id"] == "DOC001" and not a.abstained

def test_refuses_out_of_domain():
    p = _pipe(); a = p.ask("What is the price of Bitcoin today?", threshold=0.2)
    assert a.abstained and a.answer == NOT_FOUND and a.sources            # sources still shown for checking

def test_empty_index_message():
    a = RAGPipeline(tempfile.mkdtemp(), "hashing").ask("anything")
    assert a.abstained and "No documents" in a.answer

def test_metadata_in_chunks():
    p = _pipe(); s = p.retrieve("Directive Principles of State Policy Part IV", 1)[0]
    assert {"doc_id", "title", "doc_type", "chunk_no", "score", "text"} <= set(s)

def test_log_and_feedback():
    path = os.path.join(tempfile.mkdtemp(), "l.db"); p = _pipe(); a = p.ask("Who chaired the Drafting Committee?")
    rid = logdb.log(a, "extractive", "hashing", path); logdb.set_feedback(rid, "up", path)
    df = logdb.read(path); assert len(df) == 1 and df.feedback.iloc[0] == "up"

def test_reindex_idempotent():
    p = _pipe(); n = p.store.count(); f = sorted(os.listdir(CORP))[0]; p.index_file(os.path.join(CORP, f), name=f)
    assert p.store.count() == n                                           # upsert: same chunk ids are not duplicated

if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns: fn(); print("PASS", fn.__name__)
    print(f"{len(fns)} of {len(fns)} tests passed")
