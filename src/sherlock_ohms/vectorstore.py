"""Thin wrapper around a persistent Chroma collection with OpenAI embeddings."""

from pathlib import Path
from typing import List

import chromadb
import pdfplumber
from chromadb.utils import embedding_functions

PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma"


def _client() -> chromadb.ClientAPI:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(PERSIST_DIR))


def get_or_create_collection(name: str, openai_api_key: str):
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key)
    return _client().get_or_create_collection(name=name, embedding_function=embedding_fn)


def load_pdf_into_collection(pdf_path: str, collection) -> int:
    """Chunk a PDF page-by-page and add any pages missing from the collection. Returns pages added."""
    existing_ids = set(collection.get(include=[])["ids"])
    ids, docs = [], []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_id = f"{Path(pdf_path).stem}-p{page_num}"
            if page_id in existing_ids:
                continue
            text = page.extract_text()
            if text:
                ids.append(page_id)
                docs.append(text)

    if docs:
        collection.add(documents=docs, ids=ids)
    return len(docs)


def query(collection, question: str, n_results: int = 3) -> List[str]:
    results = collection.query(query_texts=[question], n_results=n_results, include=["documents"])
    return results["documents"][0] if results["documents"] else []
