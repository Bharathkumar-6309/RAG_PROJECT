"""
generator.py
Uses an LLM to generate a final answer from retrieved context.
"""

import os
from typing import List, Tuple

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_chat_model(provider: str | None = None):
    """
    Return chat model based on MODEL_PROVIDER env var.
    - openai (default): ChatOpenAI
    - ollama: ChatOllama
    """
    provider = (provider or os.getenv("MODEL_PROVIDER", "openai")).strip().lower()
    if provider == "ollama":
        model_name = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
        return ChatOllama(model=model_name, temperature=0)

    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0)


def build_prompt(query: str, retrieved_chunks_with_scores: List[Tuple]) -> str:
    """Build a simple RAG prompt with context + user question."""
    context_parts = []
    for idx, (doc, score) in enumerate(retrieved_chunks_with_scores, start=1):
        context_parts.append(
            f"Chunk {idx} (similarity score: {score:.4f}):\n{doc.page_content}"
        )

    context_text = "\n\n".join(context_parts)
    prompt = f"""
You are an AI tutor assistant.
Answer the user question using ONLY the context below.
If the answer is not in context, say clearly: "I don't have enough context to answer that."

Context:
{context_text}

Question:
{query}

Answer in a clear, beginner-friendly style.
"""
    return prompt.strip()


def generate_answer(query: str, retrieved_chunks_with_scores: List[Tuple], provider: str | None = None) -> str:
    """Generate response using configured chat provider."""
    llm = get_chat_model(provider=provider)
    prompt = build_prompt(query, retrieved_chunks_with_scores)
    response = llm.invoke(prompt)
    return response.content
