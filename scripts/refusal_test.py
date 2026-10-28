"""Out-of-domain test: the system should REFUSE questions that are not in the documents.  python scripts/refusal_test.py"""
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import pandas as pd
from rag.pipeline import RAGPipeline
OOD = ["Who won the football world cup in 2018?", "Give me a recipe for butter chicken.", "How tall is Mount Everest?", "What is the price of Bitcoin today?",
 "Who is the CEO of Tesla?", "Explain the plot of Hamlet.", "What is the best smartphone to buy?", "Which planet has the most moons?", "How many runs did Sachin Tendulkar score?",
 "What is the boiling point of mercury?", "Who painted the Mona Lisa?", "How do I train a dog to sit?", "What is the population of Brazil?", "When was the Eiffel Tower built?",
 "Which team won the IPL in 2011?", "What is the speed of light?", "How do airplanes fly?", "What are the symptoms of dengue?", "Who wrote the Ramayana?", "What is the capital of Australia?"]
pipe = RAGPipeline(os.path.join(ROOT, "chroma_db"), "hashing")
qa = pd.read_csv(os.path.join(ROOT, "data", "qa_pairs.csv")).sample(200, random_state=1)
rows = []
for th in [0.10, 0.15, 0.20, 0.25]:
    ood = sum(pipe.ask(q, 4, th).abstained for q in OOD) / len(OOD)
    ind = sum(pipe.ask(q, 4, th).abstained for q in qa.question) / len(qa)
    rows.append(dict(threshold=th, ood_refused=round(ood, 3), in_domain_wrongly_refused=round(ind, 3)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "data", "refusal_test.csv"), index=False); print(df)
