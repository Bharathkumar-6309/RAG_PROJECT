"""
embeddings.py
Handles:
1) Loading documents
2) Splitting them into chunks
3) Creating embeddings
4) Storing embeddings in FAISS
"""

import os
from pathlib import Path
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embedding_model(provider: str | None = None):
    """
    Return embedding model based on MODEL_PROVIDER env var.
    - openai (default): uses OpenAIEmbeddings
    - ollama: uses local Ollama embeddings
    """
    provider = (provider or os.getenv("MODEL_PROVIDER", "openai")).strip().lower()
    if provider == "ollama":
        model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model_name)

    return OpenAIEmbeddings()


def load_documents(data_dir: str = "data") -> List:
    """Load all .txt files from the data directory."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs = []
    for file_path in path.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def split_documents(documents: Iterable, chunk_size: int = 500, chunk_overlap: int = 80) -> List:
    """Split long documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def create_vector_store(chunks: List, provider: str | None = None) -> FAISS:
    """Create a FAISS vector store from chunks using selected embedding provider."""
    embedding_model = get_embedding_model(provider=provider)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


def build_vector_store_from_data(data_dir: str = "data", provider: str | None = None) -> FAISS:
    """
    End-to-end helper:
    - Load docs
    - Split docs
    - Build FAISS store
    """
    docs = load_documents(data_dir=data_dir)
    if not docs:
        raise ValueError("No .txt documents found in the data directory.")

    chunks = split_documents(docs)
    return create_vector_store(chunks, provider=provider)
