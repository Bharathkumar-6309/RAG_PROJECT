"""
retriever.py
Handles retrieval from FAISS vector store.
"""

from typing import List, Tuple

from langchain_community.vectorstores import FAISS


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 4,
) -> List[Tuple]:
    """
    Retrieve top_k relevant chunks with similarity score.
    Lower score generally means closer match for FAISS L2 distance.
    """
    return vector_store.similarity_search_with_score(query, k=top_k)
