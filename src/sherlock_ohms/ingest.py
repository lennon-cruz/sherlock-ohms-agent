"""Download the source PDFs (if missing) and embed them into local Chroma collections.

Usage: uv run ingest
"""

import os

import requests
from dotenv import load_dotenv

from sherlock_ohms import vectorstore
from sherlock_ohms.config import CORPORA, DATA_DIR


def download(url: str, dest) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; sherlock-ohms-agent/0.1)"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)


def main() -> None:
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, corpus in CORPORA.items():
        download(corpus["url"], corpus["pdf_path"])
        collection = vectorstore.get_or_create_collection(corpus["collection_name"], api_key)
        added = vectorstore.load_pdf_into_collection(str(corpus["pdf_path"]), collection)
        print(f"[{key}] {added} new page(s) embedded into '{corpus['collection_name']}'")


if __name__ == "__main__":
    main()
