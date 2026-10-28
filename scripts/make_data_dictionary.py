import os, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
f = pd.read_csv(os.path.join(ROOT, "data", "dataset_features.csv"))
D = {
 "doc_id": ("Identifier", "Unique ID of the source document in the corpus (DOC001-DOC050). 50 documents: 18 Constitution, 16 government schemes, 16 technical manuals."),
 "doc_type": ("Categorical", "File format of the source document: PDF, Word, HTML, Markdown or TXT. Fixed per document (made consistent during cleaning)."),
 "chunk_id": ("Identifier", "Unique ID of one text chunk (CHK00001 onward). A chunk is the unit stored in the vector database and retrieved for a question."),
 "word_count": ("Integer (words)", "Number of words in the chunk. Range 50-499. Controls how much context is given to the LLM."),
 "retrieval_score": ("Float 0-1", "Cosine-similarity score between the question embedding and the chunk embedding. Higher = chunk is more relevant to the question."),
 "query_type": ("Categorical", "Kind of user question: factual, analytical, comparative, procedural or summarization."),
 "answer_quality_score": ("Float 0-1", "Quality rating of the generated answer (correctness, completeness, groundedness) on a 0-1 scale. Higher is better."),
 "length_bucket": ("Derived categorical", "Chunk length group made from word_count: short, medium, long, very_long."),
 "retrieval_band": ("Derived categorical", "Retrieval score group: low (<0.6), fair, good, excellent (>0.9)."),
 "quality_band": ("Derived categorical", "Answer-quality group: poor (<0.7), ok, good (>0.85)."),
 "combined_score": ("Derived float 0-1", "Average of retrieval_score and answer_quality_score; one overall health number per chunk/query."),
 "hallucination_risk": ("Derived binary 0/1", "1 if retrieval_score < 0.6 or answer_quality_score < 0.65. These rows need human review before the answer is trusted."),
 "category": ("Derived categorical", "Corpus group of the document: Constitution, Govt schemes or Technical manual."),
}
rows = []
for c in f.columns:
    s = f[c]; numeric = pd.api.types.is_numeric_dtype(s)
    rows.append(dict(column_name=c, in_original_file="Yes" if c in list(f.columns[:7]) else "No (derived)", data_type=("object (text)" if str(s.dtype) in ("str","object") else str(s.dtype)), kind=D[c][0],
        sample_value_1=s.iloc[0], sample_value_2=s.iloc[1], sample_value_3=s.iloc[2], nullable="No", missing_count=int(s.isna().sum()),
        unique_values=int(s.nunique()), min_value=s.min() if numeric else "", max_value=s.max() if numeric else "",
        mean_value=round(s.mean(), 4) if numeric else "", description=D[c][1]))
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "data_dictionary.csv"), index=False)
print(pd.DataFrame(rows)[["column_name", "data_type", "unique_values", "min_value", "max_value"]])
