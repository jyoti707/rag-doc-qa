"""SQLite log of questions, answers, scores and user feedback (used for the Analytics tab)."""
import sqlite3, datetime as dt

SCHEMA = """CREATE TABLE IF NOT EXISTS query_log(
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, question TEXT, answer TEXT, top_score REAL, grounding REAL,
 abstained INTEGER, provider TEXT, embedder TEXT, sources TEXT, feedback TEXT)"""


def _c(path="qa_log.db"):
    c = sqlite3.connect(path); c.execute(SCHEMA); return c


def log(ans, provider, embedder, path="qa_log.db"):
    with _c(path) as c:
        cur = c.execute("INSERT INTO query_log(ts,question,answer,top_score,grounding,abstained,provider,embedder,sources) VALUES(?,?,?,?,?,?,?,?,?)",
            (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ans.question, ans.answer, ans.top_score, ans.grounding,
             int(ans.abstained), provider, embedder, ",".join(s["chunk_id"] for s in ans.sources)))
        return cur.lastrowid


def set_feedback(row_id, fb, path="qa_log.db"):
    with _c(path) as c: c.execute("UPDATE query_log SET feedback=? WHERE id=?", (fb, row_id))


def read(path="qa_log.db"):
    import pandas as pd
    with _c(path) as c: return pd.read_sql_query("SELECT * FROM query_log ORDER BY id DESC", c)
