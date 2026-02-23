# Information Retrieval System using Vector Space & Query Likelihood Models
## Project Overview

This project implements a complete Information Retrieval (IR) system that retrieves relevant documents from a unified news corpus using:

- Vector Space Model (TF-IDF + Cosine Similarity)

- Query Likelihood Model (Jelinek–Mercer Smoothing)

- Streamlit Web Application Interface

The system is built as an extension of a web crawling and indexing pipeline developed in Previous Project.

## System Architecture

The project follows this pipeline:

- Crawl multiple news websites

- Create a unified HTML corpus

- Build an inverted index with term frequencies

- Compute document statistics (lengths, norms, metadata)

- Implement VSM retrieval model

- Implement QLM retrieval model

- Build a Streamlit application for querying and displaying ranked results
----
## Retrieval Models
Vector Space Model (VSM)

- Term Frequency (TF)

- Inverse Document Frequency (IDF)

- TF-IDF weighting

- Cosine Similarity ranking
```text
Cosine Similarity= A⋅B / ∣∣A∣∣ ∣∣B∣∣​
````
Documents are ranked based on similarity to the query vector.

Query Likelihood Model (QLM)

- Document language model

- laplace smoothing
```text
P(t∣D)= (TF(t,D) + 1 / ∣D∣ ​+ |V|​
````
Documents are ranked based on log-likelihood scores of generating the query.

----
## Web Application

The Streamlit application provides:

- Query input box

- Two result tabs:

     -- VSM ranking

     -- QLM ranking

- Display of document title, URL, and score

----
## How to Run ?
1. Install Dependencies
```text
pip install -r requirements.txt
````
2. Run the Application
```text
streamlit run app.py
````
## Academic Context

This project was developed as part of the Information Retrieval coursework module.

It demonstrates practical implementation of:

- Text preprocessing

- Index construction

- Probabilistic language modeling

- Vector-based retrieval

- Web-based search interfaces

----
🧑‍💻 Author

Jonathan Mishayel

Academic Project | Information Retrieval Systems

@2025
