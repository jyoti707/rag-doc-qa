import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); FIG = os.path.join(ROOT, "figures")

def box(ax, x, y, w, h, text, fc="#DCE9F7", ec="#2F5597", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arrow(ax, p1, p2, label="", rad=0.0, fs=7.5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333", connectionstyle=f"arc3,rad={rad}"))
    if label: ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.12, label, ha="center", fontsize=fs, color="#7a1f1f", backgroundcolor="white")

# Fig A: DFD level 0
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
box(ax, 0.2, 1.5, 2.0, 1.2, "USER\n(Student / Employee)", "#FFF2CC", "#BF9000", 10, True)
box(ax, 3.7, 1.2, 2.6, 1.8, "RAG-Based\nDocument Q&A\nSystem", "#DCE9F7", "#2F5597", 11, True)
box(ax, 7.8, 2.7, 2.0, 1.0, "LLM provider\n(OpenAI / Groq / Gemini /\nOllama / Claude)", "#E2F0D9", "#548235", 8.5, True)
box(ax, 7.8, 0.4, 2.0, 1.0, "Vector DB\n(ChromaDB, on disk)", "#E2F0D9", "#548235", 9, True)
arrow(ax, (2.2, 2.4), (3.7, 2.4), "documents, question"); arrow(ax, (3.7, 1.8), (2.2, 1.8), "answer + sources")
arrow(ax, (6.3, 2.8), (7.8, 3.2), "question + context"); arrow(ax, (7.8, 2.9), (6.3, 2.5), "answer text", rad=-0.2)
arrow(ax, (6.3, 1.6), (7.8, 1.0), "chunks + embeddings"); arrow(ax, (7.8, 0.7), (6.3, 1.3), "top-k chunks", rad=-0.2)
plt.savefig(os.path.join(FIG, "arch_dfd0.png"), dpi=150, bbox_inches="tight"); plt.close()

# Fig B: pipeline / DFD level 1
fig, ax = plt.subplots(figsize=(11, 5.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
top = [("1.0 Load\ndocuments", "PDF/Word/HTML/\nMD/TXT"), ("2.0 Clean &\nChunk", "500 chars,\n80 overlap"), ("3.0 Embed", "MiniLM / OpenAI /\nhashing"), ("4.0 Store", "ChromaDB\n(cosine)")]
for i, (t, s) in enumerate(top):
    box(ax, 0.3 + i * 2.7, 3.6, 2.1, 1.1, t + "\n" + s, "#DCE9F7", fs=8.5)
    if i: arrow(ax, (0.3 + i * 2.7 - 0.6, 4.15), (0.3 + i * 2.7, 4.15))
ax.text(0.3, 4.95, "INDEXING (offline, once per document)", fontsize=9, fontweight="bold", color="#2F5597")
bot = [("5.0 Embed\nquestion", ""), ("6.0 Retrieve\ntop-k chunks", "similarity search"), ("7.0 Generate\nanswer", "LLM, context only"), ("8.0 Guard &\ndisplay", "grounding score,\nsources, refusal")]
for i, (t, s) in enumerate(bot):
    box(ax, 0.3 + i * 2.7, 1.2, 2.1, 1.1, t + ("\n" + s if s else ""), "#E2F0D9", "#548235", 8.5)
    if i: arrow(ax, (0.3 + i * 2.7 - 0.6, 1.75), (0.3 + i * 2.7, 1.75))
ax.text(0.3, 2.55, "QUERYING (online, every question)", fontsize=9, fontweight="bold", color="#548235")
arrow(ax, (9.3, 3.6), (4.6, 2.3), "", rad=0.0); ax.text(7.6, 3.15, "stored index is searched by 6.0", fontsize=7.5, color="#7a1f1f", backgroundcolor="white")
box(ax, 8.2, 0.05, 2.5, 0.7, "D2 Query log (SQLite)", "#FFF2CC", "#BF9000", 8.5)
arrow(ax, (9.0, 1.2), (9.2, 0.75), "")
plt.savefig(os.path.join(FIG, "arch_dfd1.png"), dpi=150, bbox_inches="tight"); plt.close()

# Fig C: layered block diagram
fig, ax = plt.subplots(figsize=(10, 5.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
layers = [("Presentation layer - Streamlit (app.py)", ["Ask tab", "Documents tab", "Analytics tab", "Evaluation tab", "Sidebar settings"], "#FFF2CC", "#BF9000", 4.5),
          ("Core logic layer - package rag/", ["loaders.py", "pipeline.py\n(chunk, retrieve, guard)", "embeddings.py", "llm.py", "store.py", "logdb.py"], "#DCE9F7", "#2F5597", 2.95),
          ("External / storage layer", ["Embedding model\n(MiniLM / OpenAI)", "LLM API / Ollama", "ChromaDB\n(chroma_db/)", "qa_log.db\n(SQLite)", "Uploaded files"], "#E2F0D9", "#548235", 1.4)]
for name, items, fc, ec, y in layers:
    ax.add_patch(FancyBboxPatch((0.1, y - 0.1), 9.8, 1.25, boxstyle="round,pad=0.02", fc="white", ec=ec, lw=2)); ax.text(0.25, y + 0.95, name, fontsize=9.5, fontweight="bold", color=ec)
    w = 9.4 / len(items)
    for i, t in enumerate(items): box(ax, 0.3 + i * w, y, w - 0.2, 0.7, t, fc, ec, 8)
for y in (4.5, 2.95): arrow(ax, (5, y - 0.1), (5, y - 0.35)); arrow(ax, (5.3, y - 0.35), (5.3, y - 0.1))
ax.text(5, 0.7, "Offline evaluation scripts (scripts/): ingest_corpus, evaluate, ablation, refusal_test, expand/clean dataset", ha="center", fontsize=8.5, style="italic")
plt.savefig(os.path.join(FIG, "arch_blocks.png"), dpi=150, bbox_inches="tight"); plt.close()
print("diagrams done")
