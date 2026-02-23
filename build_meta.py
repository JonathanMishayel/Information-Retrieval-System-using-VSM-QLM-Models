import os, json
from bs4 import BeautifulSoup

CORPUS_DIR = "corpus_html"
META_FILE = "doc_meta.json"

meta = {}
count = 0

for filename in os.listdir(CORPUS_DIR):
    if not filename.endswith(".html"):
        continue

    doc_id = filename.split(".")[0]
    path = os.path.join(CORPUS_DIR, filename)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else doc_id
    canonical = soup.find("link", rel="canonical")
    url = canonical["href"] if canonical and canonical.get("href") else ""

    meta[doc_id] = {"title": title, "url": url}

    count += 1
    if count % 100 == 0:
        print(f"Processed {count} documents...")

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Done. Total:", count)