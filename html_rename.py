import os
import shutil

# Your four crawler folders
folders = {
    "cnn": "crawled_pages_list2",
    "nyt": "crawled_pages_listx",
    "bbc": "crawled_pages_list3",
    "aj": "crawled_pages_list4"
}

output_folder = "corpus_html"
os.makedirs(output_folder, exist_ok=True)

for prefix, folder in folders.items():
    for i, filename in enumerate(os.listdir(folder), start=1):
        if filename.endswith(".html"):
            old_path = os.path.join(folder, filename)
            new_filename = f"{prefix}_{i}.html"
            new_path = os.path.join(output_folder, new_filename)

            shutil.copy(old_path, new_path)

    print(f"Finished merging {prefix}")

print("All files merged and renamed successfully.")