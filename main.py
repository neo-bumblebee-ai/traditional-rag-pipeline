"""
Traditional RAG Pipeline
Ingest → Chunk → Embed → Store → Retrieve → Generate

Documents : PDF and text files under data/
Embeddings: sentence-transformers (all-MiniLM-L6-v2)
Vector DB : ChromaDB (persisted to data/vector_store/)
Generation: Ollama (local LLM)
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ── Configuration ──────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = "data/vector_store"
PDF_DIR          = "data/pdf_files"
TEXT_DIR         = "data/text_files"
COLLECTION_NAME  = "pdf_chunks"       # reuses the collection already ingested by the notebook
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
OLLAMA_MODEL     = "llama3.2"         # change to any model you have pulled (e.g. mistral, llama3)
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
TOP_K            = 5


# ── Document loading ───────────────────────────────────────────────────────────
def load_documents() -> list[Document]:
    docs: list[Document] = []

    pdf_path = Path(PDF_DIR)
    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        print(f"  Loading PDF : {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        for page in pages:
            page.metadata.update({"source_file": pdf_file.name, "file_type": "pdf"})
        docs.extend(pages)

    text_path = Path(TEXT_DIR)
    for txt_file in sorted(text_path.glob("*.txt")):
        print(f"  Loading TXT : {txt_file.name}")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        pages = loader.load()
        for page in pages:
            page.metadata.update({"source_file": txt_file.name, "file_type": "text"})
        docs.extend(pages)

    print(f"Loaded {len(docs)} document pages total.")
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# ── Embeddings ─────────────────────────────────────────────────────────────────
class EmbeddingManager:
    """Wraps SentenceTransformer for generating embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        print(f"Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name)
        print(f"Embedding dimension : {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=show_progress)


# ── Vector store ───────────────────────────────────────────────────────────────
class VectorStore:
    """ChromaDB-backed persistent vector store."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_dir: str = VECTOR_STORE_DIR,
    ) -> None:
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(
            f"Vector store ready — {self.collection.count()} docs "
            f"in collection '{collection_name}'."
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def add_documents(self, chunks: list[Document], embeddings: np.ndarray) -> None:
        BATCH = 500
        ids, texts, metas, embeds = [], [], [], []

        for i, (doc, emb) in enumerate(zip(chunks, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            texts.append(doc.page_content)
            # ChromaDB metadata values must be str / int / float / bool
            meta = {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in doc.metadata.items()
            }
            meta["content_length"] = len(doc.page_content)
            metas.append(meta)
            embeds.append(emb.tolist())

        for start in range(0, len(ids), BATCH):
            self.collection.add(
                ids=ids[start : start + BATCH],
                documents=texts[start : start + BATCH],
                metadatas=metas[start : start + BATCH],
                embeddings=embeds[start : start + BATCH],
            )

        print(f"Stored {len(ids)} chunks. Total in collection: {self.collection.count()}.")

    def query(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        # all-MiniLM-L6-v2 produces unit vectors, so L2 distance ≈ 2*(1-cosine_sim)
        # convert to a 0-1 similarity score for display
        return [
            {
                "document": doc,
                "metadata": meta,
                "score": max(0.0, 1.0 - dist / 2.0),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


# ── Generation ─────────────────────────────────────────────────────────────────
def generate_answer(
    question: str,
    context_docs: list[dict[str, Any]],
    model: str = OLLAMA_MODEL,
) -> str:
    context_parts = []
    for d in context_docs:
        source = d["metadata"].get("source_file", "unknown")
        page   = d["metadata"].get("page", "?")
        context_parts.append(f"[Source: {source}, page {page}]\n{d['document']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant. Answer the question using only the context below.\n"
        "If the answer is not found in the context, say: "
        "\"I don't have enough information to answer that.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ── Pipeline helpers ───────────────────────────────────────────────────────────
def ingest(vector_store: VectorStore, embedding_manager: EmbeddingManager) -> None:
    """Load, chunk, embed, and store all documents. Skips if already populated."""
    if vector_store.count > 0:
        print("Vector store already populated — skipping ingestion.\n")
        return

    print("\n── Ingesting documents ──────────────────────────────────────────")
    documents = load_documents()
    chunks    = split_documents(documents)
    print("Generating embeddings …")
    embeddings = embedding_manager.embed([c.page_content for c in chunks])
    vector_store.add_documents(chunks, embeddings)
    print("Ingestion complete.\n")


def rag_query(
    question: str,
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
) -> str:
    """Embed the question, retrieve context, and generate an answer."""
    q_emb   = embedding_manager.embed([question], show_progress=False)[0]
    results = vector_store.query(q_emb, top_k=TOP_K)

    print(f"\nTop {TOP_K} retrieved chunks:")
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source_file", "?")
        pg  = r["metadata"].get("page", "?")
        snippet = r["document"][:80].replace("\n", " ").strip()
        print(f"  {i}. [{r['score']:.3f}] {src} p.{pg} — {snippet}…")

    print("\nGenerating answer with Ollama …")
    return generate_answer(question, results)


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Traditional RAG Pipeline ===\n")

    embedding_manager = EmbeddingManager()
    vector_store      = VectorStore()

    ingest(vector_store, embedding_manager)

    print(f"Ollama model : {OLLAMA_MODEL}")
    print("Type a question or 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        answer = rag_query(question, vector_store, embedding_manager)
        print(f"\nAnswer:\n{answer}\n")
        print("─" * 60)


if __name__ == "__main__":
    main()
