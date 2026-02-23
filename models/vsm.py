import json
import math
import re
from nltk.stem import PorterStemmer

INDEX_FILE = "inverted_index_tf.json"
DOC_LEN_FILE = "doc_lengths.json"
DOC_NORMS_FILE = "doc_norms_vsm.json"  # cache to speed up next runs

ps = PorterStemmer()

def preprocess_query(q: str):
    q = q.lower()
    tokens = re.findall(r"[a-z]+", q)
    return [ps.stem(t) for t in tokens]

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def compute_doc_norms(index: dict, N_docs: int):
    """
    Computes ||d|| for each document using TF-IDF weights:
    w(t,d) = (1 + log(tf)) * idf, where idf = log(N/df)
    Returns dict doc_id -> norm
    """
    norm_sq = {}  # doc -> sum(w^2)

    for term, postings in index.items():
        df = len(postings)
        if df == 0:
            continue

        idf = math.log(N_docs / df)

        for doc_id, tf in postings.items():
            if tf <= 0:
                continue
            w = (1.0 + math.log(tf)) * idf
            norm_sq[doc_id] = norm_sq.get(doc_id, 0.0) + (w * w)

    # sqrt
    norms = {doc_id: math.sqrt(v) for doc_id, v in norm_sq.items()}
    return norms

def vsm_search(query: str, top_k: int = 10):
    index = load_json(INDEX_FILE)
    doc_lengths = load_json(DOC_LEN_FILE)  # mainly for N docs
    N = len(doc_lengths)

    # Load or compute doc norms (big speed-up after first run)
    try:
        doc_norms = load_json(DOC_NORMS_FILE)
    except FileNotFoundError:
        print("Computing document norms (first time only)...")
        doc_norms = compute_doc_norms(index, N)
        save_json(DOC_NORMS_FILE, doc_norms)
        print("Saved doc norms cache:", DOC_NORMS_FILE)

    q_terms = preprocess_query(query)
    if not q_terms:
        return []

    # Query term frequencies
    q_tf = {}
    for t in q_terms:
        q_tf[t] = q_tf.get(t, 0) + 1

    # Accumulator for dot product: doc -> sum(w_q * w_d)
    scores = {}

    # Build query vector norm too
    q_norm_sq = 0.0

    for term, tf_q in q_tf.items():
        postings = index.get(term)
        if not postings:
            continue

        df = len(postings)
        idf = math.log(N / df)

        w_q = (1.0 + math.log(tf_q)) * idf
        q_norm_sq += (w_q * w_q)

        for doc_id, tf_d in postings.items():
            w_d = (1.0 + math.log(tf_d)) * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + (w_q * w_d)

    q_norm = math.sqrt(q_norm_sq)
    if q_norm == 0.0:
        return []

    # Cosine similarity
    results = []
    for doc_id, dot in scores.items():
        d_norm = float(doc_norms.get(doc_id, 0.0))
        if d_norm == 0.0:
            continue
        sim = dot / (q_norm * d_norm)
        results.append((doc_id, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    while True:
        q = input("\nVSM Query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        out = vsm_search(q, top_k=10)
        for rank, (doc_id, score) in enumerate(out, start=1):
            print(f"{rank:02d}. {doc_id}  score={score:.6f}")