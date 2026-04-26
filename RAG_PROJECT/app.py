"""
app.py
Streamlit interface for the RAG application.
Features:
- Ask questions about AI
- See retrieved chunks
- Upload custom .txt documents (bonus)
- Show similarity scores (bonus)
"""

from copy import deepcopy
import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from embeddings import build_vector_store_from_data, get_embedding_model
from generator import generate_answer
from retriever import retrieve_relevant_chunks


def _texts_from_uploads(uploaded_files) -> Tuple[List[str], List[dict]]:
    """Convert uploaded .txt files to plain texts + metadata."""
    texts: List[str] = []
    metadatas: List[dict] = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        if text.strip():
            texts.append(text)
            metadatas.append({"source": uploaded_file.name})
    return texts, metadatas


@st.cache_resource
def _base_vector_store(provider: str):
    """Cache base vector store from local /data folder."""
    return build_vector_store_from_data(data_dir="data", provider=provider)


def main():
    load_dotenv()
    default_provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()

    st.set_page_config(page_title="Simple AI RAG", page_icon="🤖", layout="wide")
    st.title("🤖 AI Basics RAG Chatbot")
    st.write("Ask questions about AI history, types, and applications.")
    st.sidebar.header("Settings")
    provider = st.sidebar.selectbox(
        "Model provider",
        options=["openai", "ollama"],
        index=0 if default_provider == "openai" else 1,
        help="Choose which backend to use for embeddings and answer generation.",
    )
    st.caption(f"Active provider: {provider}")

    # OpenAI mode requires API key. Ollama mode does not.
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY. Create a .env file and set OPENAI_API_KEY=your_key, then restart Streamlit.")
        st.stop()

    # Build base vector store from the provided knowledge base.
    try:
        base_store = _base_vector_store(provider)
    except Exception as exc:
        if provider == "ollama":
            st.error(
                "Could not connect to Ollama. Install and start Ollama, then try again.\n\n"
                "1) Install from https://ollama.com/download\n"
                "2) Run: ollama pull llama3.1\n"
                "3) Run: ollama pull nomic-embed-text\n"
                "4) Keep Ollama app/service running"
            )
            st.info(f"Technical detail: {exc}")
            st.stop()
        raise

    top_k = st.sidebar.slider("Number of retrieved chunks (top_k)", min_value=1, max_value=8, value=4)
    st.sidebar.markdown("Optional: upload custom `.txt` files to extend context.")

    uploaded_files = st.sidebar.file_uploader(
        "Upload custom text files",
        type=["txt"],
        accept_multiple_files=True,
    )

    active_store = base_store
    if uploaded_files:
        texts, metadatas = _texts_from_uploads(uploaded_files)
        if texts:
            custom_store = FAISS.from_texts(
                texts=texts,
                embedding=get_embedding_model(provider=provider),
                metadatas=metadatas,
            )

            # Merge user-uploaded vectors with the base store for retrieval.
            active_store = deepcopy(base_store)
            active_store.merge_from(custom_store)
            st.sidebar.success(f"Loaded {len(uploaded_files)} custom file(s).")

    query = st.text_input("Your question", placeholder="Example: What is Narrow AI?")

    if st.button("Ask") and query.strip():
        with st.spinner("Retrieving relevant context and generating answer..."):
            retrieved = retrieve_relevant_chunks(active_store, query=query, top_k=top_k)
            answer = generate_answer(query=query, retrieved_chunks_with_scores=retrieved, provider=provider)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks")
        for idx, (doc, score) in enumerate(retrieved, start=1):
            source = doc.metadata.get("source", "unknown")
            with st.expander(f"Chunk {idx} | Similarity score: {score:.4f} | Source: {Path(source).name}"):
                st.write(doc.page_content)


if __name__ == "__main__":
    main()
