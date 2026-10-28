# Retrieval-Augmented Generation (RAG) Explained

*Source: Open technical documentation and standard references (public knowledge, paraphrased)*

This document gives a short overview of Retrieval-Augmented Generation (RAG) Explained.

## Key points

Retrieval-Augmented Generation combines a document search step with a language model so that answers are grounded in retrieved text. A typical RAG pipeline loads documents, splits them into chunks, creates embeddings, stores them, retrieves relevant chunks and generates an answer. RAG reduces hallucination because the model is told to answer only from the retrieved context.

Chunk overlap repeats a few words between neighbouring chunks so that sentences are not cut off from their context. The top-k setting decides how many of the most similar chunks are passed to the language model. Cosine similarity compares the angle between two embedding vectors to measure how similar their meanings are.

The RAG approach was introduced in a 2020 paper by Lewis and colleagues at Facebook AI Research. A good RAG application shows the source chunk next to the answer so that the user can verify it.

> Note: This is a simplified educational summary written in our own words from public information. It is not legal, financial or medical advice. Verify details with the official source before relying on them.
