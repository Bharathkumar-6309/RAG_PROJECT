"""
main.py
Command-line entry point for the RAG pipeline.
"""

import os

from dotenv import load_dotenv

from embeddings import build_vector_store_from_data
from generator import generate_answer
from retriever import retrieve_relevant_chunks


def run_cli() -> None:
    """Run a simple terminal-based RAG chat loop."""
    load_dotenv()
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY. Create a .env file with OPENAI_API_KEY=your_key and run again.")
        return

    print(f"Using provider: {provider}")
    print("Building vector store from data files...")
    vector_store = build_vector_store_from_data(data_dir="data")
    print("Ready! Ask questions about AI (type 'exit' to quit).")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        retrieved = retrieve_relevant_chunks(vector_store, query, top_k=4)
        answer = generate_answer(query, retrieved)

        print("\nAnswer:")
        print(answer)
        print("\nRetrieved Chunks:")
        for idx, (doc, score) in enumerate(retrieved, start=1):
            print(f"\n[{idx}] Score: {score:.4f}")
            print(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))


if __name__ == "__main__":
    run_cli()
