"""Read text from PDF, Word, HTML, Markdown and TXT files."""
import os
from html.parser import HTMLParser


class _Strip(HTMLParser):
    def __init__(self):
        super().__init__(); self.out = []; self.skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"): self.skip = True
        if tag in ("p", "br", "h1", "h2", "h3", "li", "div"): self.out.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style"): self.skip = False
    def handle_data(self, data):
        if not self.skip: self.out.append(data)


def load_file(path_or_file, name=None):
    """Return (text, file_name). Accepts a path or a file-like object (Streamlit upload)."""
    name = name or getattr(path_or_file, "name", None) or str(path_or_file)
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path_or_file)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    elif ext == ".docx":
        from docx import Document
        d = Document(path_or_file)
        text = "\n".join(p.text for p in d.paragraphs)
    elif ext in (".html", ".htm"):
        raw = _read(path_or_file)
        s = _Strip(); s.feed(raw); text = "".join(s.out)
    elif ext in (".txt", ".md", ".markdown"):
        text = _read(path_or_file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    text = "\n".join(line.strip() for line in text.splitlines())
    while "\n\n\n" in text: text = text.replace("\n\n\n", "\n\n")
    return text.strip(), os.path.basename(name)


def _read(f):
    if hasattr(f, "read"):
        data = f.read()
        return data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else data
    return open(f, encoding="utf-8", errors="ignore").read()
