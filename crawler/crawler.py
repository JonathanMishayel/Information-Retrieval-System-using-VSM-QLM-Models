import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag

def crawl_site(seed_url: str, max_pages: int, delay: float, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    allowed_root = urlparse(seed_url).netloc.split(".", 1)[-1]  # e.g., cnn.com, nytimes.com

    visited = set()
    to_visit = [seed_url]

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue

            visited.add(url)
            page_id = len(visited)

            file_path = os.path.join(save_dir, f"page_{page_id}.html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                next_url = urljoin(url, a["href"])
                next_url, _ = urldefrag(next_url)  # remove #fragment

                # keep within same root domain
                if urlparse(next_url).netloc.endswith(allowed_root):
                    if next_url not in visited and next_url not in to_visit:
                        to_visit.append(next_url)

            time.sleep(delay)

        except Exception:
            continue

    return len(visited)

# if __name__ == "__main__":
#     seed_url = "https://www.nytimes.com/section/world"
#     max_pages = 550         
#     delay = 2
#     save_dir = "corpus_html"

#     total = crawl_site(seed_url, max_pages, delay, save_dir)
#     print(f"Crawled and saved {total} pages into: {save_dir}")