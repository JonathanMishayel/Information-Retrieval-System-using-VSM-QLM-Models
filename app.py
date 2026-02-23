import streamlit as st
import os

from crawler_module import crawl_site
from indexer_unified import build_tf_index
from vsm import vsm_search
from qlm import qlm_laplace_search

st.set_page_config(page_title="IR System - VSM & QLM", layout="wide")

st.title("Information Retrieval System (VSM + QLM)")

import json

def load_meta():
    try:
        with open("doc_meta.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

doc_meta = load_meta()

# -----------------------------
# Sidebar: Crawl + Index Controls
# -----------------------------
st.sidebar.header("Pipeline Controls")

st.sidebar.subheader("1) Crawl (optional)")
seed_url = st.sidebar.text_input("Seed URL", value="https://edition.cnn.com/")
max_pages = st.sidebar.number_input("Max pages", min_value=1, max_value=2000, value=50, step=10)
delay = st.sidebar.number_input("Delay (seconds)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
save_dir = st.sidebar.text_input("Save folder", value="crawled_temp")



if st.sidebar.button("Run Crawler"):
    with st.spinner("Crawling..."):
        n = crawl_site(seed_url, int(max_pages), float(delay), save_dir)
    st.sidebar.success(f"Crawled & saved: {n} pages into '{save_dir}'")

st.sidebar.subheader("2) Index")
corpus_dir = st.sidebar.text_input("Corpus folder for indexing", value="corpus_html")

if st.sidebar.button("Build TF Index"):
    if not os.path.isdir(corpus_dir):
        st.sidebar.error("Corpus folder not found.")
    else:
        with st.spinner("Indexing..."):
            docs, terms = build_tf_index(crawl_dir=corpus_dir)
        st.sidebar.success(f"Indexed {docs} docs | {terms} unique terms")

st.sidebar.subheader("Files check")
st.sidebar.write("inverted_index_tf.json exists:", os.path.exists("inverted_index_tf.json"))
st.sidebar.write("doc_lengths.json exists:", os.path.exists("doc_lengths.json"))

# -----------------------------
# Main Search UI
# -----------------------------
st.subheader("Search")

query = st.text_input("Enter your query", value=" ")
top_k = st.slider("Top-K results", min_value=5, max_value=50, value=10, step=5)

run = st.button("Search")

tab1, tab2 = st.tabs(["VSM Results", "QLM Results"])

if run:
    if not os.path.exists("inverted_index_tf.json") or not os.path.exists("doc_lengths.json"):
        st.error("Index files not found. Please build the TF index first.")
    else:
        # VSM
        with tab1:
            st.write("### Vector Space Model (TF-IDF + Cosine)")
            with st.spinner("Running VSM..."):
                vsm_results = vsm_search(query, top_k=top_k)

            if not vsm_results:
                st.warning("No results found for VSM.")

            else:
                for i, (doc_id, score) in enumerate(vsm_results, start=1):
                   
                    m = doc_meta.get(doc_id, {})
                    title = m.get("title", doc_id)
                    url = m.get("url", "")

                    if url:
                        st.markdown(f"**{i:02d}. [{title}]({url})**")
                        st.caption(url)
                    else:
                        st.markdown(f"**{i:02d}. {title}**")
                        st.caption(doc_id)

                    st.write(f"Score: `{score:.6f}`")
                    st.divider()

        # QLM
        with tab2:
            st.write("### Query Likelihood Model (Laplace Smoothing)")
            with st.spinner("Running QLM..."):
                qlm_results = qlm_laplace_search(query, top_k=top_k)

            if not qlm_results:
                st.warning("No results found for QLM.")
            else:
                for i, (doc_id, logscore) in enumerate(qlm_results, start=1):
                    m = doc_meta.get(doc_id, {})
                    title = m.get("title", doc_id)
                    url = m.get("url", "")

                    if url:
                         st.markdown(f"**{i:02d}. [{title}]({url})**")
                         st.caption(url)
                    else:
                         st.markdown(f"**{i:02d}. {title}**")
                         st.caption(doc_id)

                    st.write(f"logscore: `{logscore:.6f}`")
                    st.divider()