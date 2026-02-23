import json
import math
import re
from nltk.stem import PorterStemmer

INDEX_FILE = "inverted_index_tf.json"
DOC_LEN_FILE = "doc_lengths.json"

ps = PorterStemmer()

def preprocess_query(q: str):
    q = q.lower()
    tokens = re.findall(r"[a-z]+", q)
    return [ps.stem(t) for t in tokens]

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def qlm_laplace_search(query: str, top_k: int = 10):
    """
    Laplace-smoothed Query Likelihood Model:

    score(d,q) = sum_{t in q} log( (tf(t,d)+1) / (|d| + V) )

    Efficient form:
    score(d) = -|q| * log(|d| + V) + sum_{t in q} log(tf(t,d)+1)
    """
    index = load_json(INDEX_FILE)
    doc_lengths = load_json(DOC_LEN_FILE)

    V = len(index)           # vocabulary size
    docs = list(doc_lengths.keys())

    q_terms = preprocess_query(query)
    if not q_terms:
        return []

    q_tf = {}
    for t in q_terms:
        q_tf[t] = q_tf.get(t, 0) + 1

    # Start with base score for every document
    # base(d) = -|q| * log(|d| + V)
    q_len = sum(q_tf.values())
    scores = {}
    for doc_id in docs:
        denom = int(doc_lengths[doc_id]) + V
        scores[doc_id] = -q_len * math.log(denom)

    # Add term contributions where tf(t,d) exists
    for term, count_in_query in q_tf.items():
        postings = index.get(term)
        if not postings:
            continue

        for doc_id, tf in postings.items():
            # term appears tf times in doc, repeated count_in_query times in query
            # add count_in_query * log(tf+1)
            scores[doc_id] += count_in_query * math.log(tf + 1)

    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    while True:
        q = input("\nQLM Query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        out = qlm_laplace_search(q, top_k=10)
        for rank, (doc_id, score) in enumerate(out, start=1):
            print(f"{rank:02d}. {doc_id}  logscore={score:.6f}")