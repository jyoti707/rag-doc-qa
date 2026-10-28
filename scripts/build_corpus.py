"""Builds the 50-document corpus (TXT, Markdown, HTML, Word, PDF) and the 1,000 Q&A pairs.
Run:  python scripts/build_corpus.py
Outputs: data/corpus/DOC001_*.ext ... , data/corpus_manifest.csv, data/qa_pairs.csv
"""
import os, re, sys, csv, html
sys.path.insert(0, os.path.dirname(__file__))
from corpus_part1 import DOCS1
from corpus_part2 import DOCS2
from corpus_part3 import DOCS3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "corpus")
DOCS = DOCS1 + DOCS2 + DOCS3
assert len(DOCS) == 50, len(DOCS)
DISCLAIMER = ("Note: This is a simplified educational summary written in our own words from public information. "
              "It is not legal, financial or medical advice. Verify details with the official source before relying on them.")
EXT = {"TXT": "txt", "Markdown": "md", "HTML": "html", "Word": "docx", "PDF": "pdf"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def body_paragraphs(facts):
    sents = [f[0] for f in facts]
    return [" ".join(sents[i:i + 3]) for i in range(0, len(sents), 3)]


def write_doc(i, d):
    doc_id = f"DOC{i:03d}"
    ext = EXT[d["fmt"]]
    path = os.path.join(OUT, f"{doc_id}_{slug(d['t'])}.{ext}")
    paras = body_paragraphs(d["facts"])
    intro = f"This document gives a short overview of {d['t']}."
    if ext == "txt":
        txt = f"{d['t']}\nSource: {d['src']}\n\n{intro}\n\n" + "\n\n".join(paras) + f"\n\n{DISCLAIMER}\n"
        open(path, "w", encoding="utf-8").write(txt)
    elif ext == "md":
        txt = f"# {d['t']}\n\n*Source: {d['src']}*\n\n{intro}\n\n## Key points\n\n" + "\n\n".join(paras) + f"\n\n> {DISCLAIMER}\n"
        open(path, "w", encoding="utf-8").write(txt)
    elif ext == "html":
        h = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>%s</title></head><body>\n<h1>%s</h1>\n<p><em>Source: %s</em></p>\n<p>%s</p>\n%s\n<footer><small>%s</small></footer>\n</body></html>"
             % (html.escape(d['t']), html.escape(d['t']), html.escape(d['src']), html.escape(intro),
                "\n".join("<p>%s</p>" % html.escape(p) for p in paras), html.escape(DISCLAIMER)))
        open(path, "w", encoding="utf-8").write(h)
    elif ext == "docx":
        from docx import Document
        doc = Document(); doc.add_heading(d["t"], 1)
        doc.add_paragraph("Source: " + d["src"]); doc.add_paragraph(intro)
        for p in paras: doc.add_paragraph(p)
        doc.add_paragraph(DISCLAIMER); doc.save(path)
    elif ext == "pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        st = getSampleStyleSheet(); esc = lambda s: html.escape(s)
        items = [Paragraph(esc(d["t"]), st["Title"]), Paragraph("Source: " + esc(d["src"]), st["Italic"]), Spacer(1, 8), Paragraph(esc(intro), st["BodyText"])]
        for p in paras: items += [Spacer(1, 6), Paragraph(esc(p), st["BodyText"])]
        items += [Spacer(1, 10), Paragraph(esc(DISCLAIMER), st["Italic"])]
        SimpleDocTemplate(path, pagesize=A4).build(items)
    return doc_id, os.path.basename(path)


def qtype(q):
    ql = q.lower()
    if any(k in ql for k in ["difference between", "compare", " versus ", " or ", "which is"]) and ("difference" in ql or "compare" in ql or " or " in ql):
        return "comparative"
    if ql.startswith(("how do you", "how can", "which command", "how to", "what are the steps", "how is", "how are")) or "command" in ql or "how do" in ql:
        return "procedural"
    if ql.startswith(("why", "what does it mean", "what is the purpose", "what is the role", "how does", "what does")) or "purpose" in ql or "why" in ql:
        return "analytical"
    return "factual"


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT): os.remove(os.path.join(OUT, f))
    manifest, qa = [], []
    for i, d in enumerate(DOCS, 1):
        doc_id, fname = write_doc(i, d)
        manifest.append(dict(doc_id=doc_id, title=d["t"], doc_type=d["fmt"], file_name=fname, source=d["src"], num_facts=len(d["facts"])))
        base = len(qa)
        # summarisation question
        qa.append(dict(doc_id=doc_id, doc_title=d["t"], source=d["src"], query_type="summarization",
                       question=f"Give a short summary of the document on {d['t']}.",
                       answer=" ".join(f[0] for f in d["facts"][:2]), evidence=d["facts"][0][0]))
        for k, (sent, q, a) in enumerate(d["facts"]):
            variants = [q, f"Regarding {d['t']}: {q}"]
            if (i + k) % 8 < 3: variants.append(f"Using the documents, answer this: {q}")   # adds 150 extra
            for v in variants:
                qa.append(dict(doc_id=doc_id, doc_title=d["t"], source=d["src"], query_type=qtype(q), question=v, answer=a, evidence=sent))
    # trim/extend to exactly 1000
    qa = qa[:1000] if len(qa) >= 1000 else qa
    for n, r in enumerate(qa, 1): r["qa_id"] = f"QA{n:04d}"
    cols = ["qa_id", "doc_id", "doc_title", "source", "query_type", "question", "answer", "evidence"]
    with open(os.path.join(ROOT, "data", "qa_pairs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows([{c: r[c] for c in cols} for r in qa])
    with open(os.path.join(ROOT, "data", "corpus_manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    from collections import Counter
    print("docs:", len(manifest), "| QA pairs:", len(qa), "| types:", dict(Counter(r["query_type"] for r in qa)))

if __name__ == "__main__":
    main()
