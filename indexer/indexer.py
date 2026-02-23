import os
import json
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def build_tf_index(crawl_dir="corpus_html",
                   index_file="inverted_index_tf.json",
                   doclen_file="doc_lengths.json"):

    inverted_index = {}   # term -> {doc_id: tf}
    doc_lengths = {}      # doc_id -> length

    for filename in os.listdir(crawl_dir):
        if not filename.endswith(".html"):
            continue

        file_path = os.path.join(crawl_dir, filename)
        doc_id = filename.split(".")[0]

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ").lower()

        # words only (no digits)
        tokens = re.findall(r"[a-z]+", text)

        doc_lengths[doc_id] = len(tokens)

        for tok in tokens:
            term = ps.stem(tok)
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = inverted_index[term].get(doc_id, 0) + 1

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=2)

    with open(doclen_file, "w", encoding="utf-8") as f:
        json.dump(doc_lengths, f, indent=2)

    return len(doc_lengths), len(inverted_index)