# Traditional RAG Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-blueviolet?style=for-the-badge)](https://www.trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange?style=for-the-badge)](https://ollama.com)
[![Embeddings](https://img.shields.io/badge/Embeddings-sentence--transformers-yellow?style=for-the-badge)](https://www.sbert.net)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A fully local, end-to-end Retrieval-Augmented Generation pipeline built from first principles — no cloud dependencies, no API keys, every component explicit and swappable.**

---

## What This Repository Is

This is a ground-up implementation of the classical RAG pattern in Python, built to understand and demonstrate how retrieval-augmented generation actually works beneath the abstractions that hosted services provide.

Every stage of the pipeline — document loading, chunking, embedding, vector storage, retrieval, and generation — is implemented explicitly, without delegating to opaque managed services. The intent is to make each design decision visible and each component independently replaceable.

> Drop in your own PDFs or text files, ask questions in plain English, and get grounded answers backed by your documents — entirely on local hardware.

---

## Live Demo

```
=== Traditional RAG Pipeline ===

Vector store ready — 847 docs in collection 'pdf_chunks'.
Vector store already populated — skipping ingestion.

Ollama model : llama3.2
Type a question or 'quit' to exit.

Question: What are the daily cleaning steps for the coffee machine?

Top 5 retrieved chunks:
  1. [0.772] Franke_A400_User_Manual_EN.pdf p.42 — The machine automatically rinses…
  2. [0.754] Franke_A400_User_Manual_EN.pdf p.43 — Milk system cleaning procedure…

Generating answer with Ollama …

Answer:
The daily cleaning steps include: 1) Run the automatic rinse cycle at the
end of each day by navigating to the Cleaning menu. 2) Clean the milk
system using the dedicated cleaning programme...
```

---

## Pipeline Architecture

```
Your Documents (PDFs / .txt)
        │
        ▼
┌───────────────────────┐
│    Document Loader    │  LangChain PyPDFLoader / TextLoader
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│     Text Splitter     │  RecursiveCharacterTextSplitter
│   chunk=1000 chars    │  overlap=200 chars
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│    Embedding Model    │  sentence-transformers / all-MiniLM-L6-v2
│    (runs locally)     │  → 384-dimensional dense vectors
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│       ChromaDB        │  Persisted to disk — no re-embedding on restart
│     Vector Store      │  Cosine similarity via L2 distance conversion
└──────────┬────────────┘
           │  (at query time)
           ▼
┌───────────────────────┐
│       Retriever       │  Top-K chunk retrieval with similarity scores
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│      Ollama LLM       │  llama3.2 (or any locally pulled model)
│    (runs locally)     │  Context-grounded answer generation
└───────────────────────┘
```

**On first run:** documents are loaded, chunked, embedded, and persisted to ChromaDB.
**On subsequent runs:** embeddings are reloaded from disk — ingestion is skipped entirely.

---

## Key Engineering Capabilities Demonstrated

| Area | Implementation |
|---|---|
| **Document ingestion** | LangChain loaders for PDF and plain text with per-page metadata tagging |
| **Chunking strategy** | `RecursiveCharacterTextSplitter` with configurable size and overlap |
| **Local embeddings** | `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional, CPU-friendly |
| **Vector persistence** | ChromaDB with disk-backed `PersistentClient` — no re-embedding on restart |
| **Similarity retrieval** | L2 distance → cosine similarity conversion for human-readable scores |
| **Local LLM generation** | Ollama integration with context-grounded prompting and source citation |
| **Idempotent ingestion** | Collection count check — documents are never re-embedded unnecessarily |
| **Batch embedding** | 500-document batch inserts to ChromaDB to handle large corpora efficiently |
| **Swappable components** | Every major component (embedder, LLM, vector store, chunk config) is configurable |

---

## Design Decisions

**Why `RecursiveCharacterTextSplitter` with overlap?**
Splitting on newlines and spaces before falling back to characters preserves semantic boundaries. Overlap ensures that context spanning a chunk boundary is not silently lost during retrieval.

**Why `all-MiniLM-L6-v2` for embeddings?**
It is lightweight enough to run comfortably on CPU, produces 384-dimensional vectors that ChromaDB handles efficiently, and offers a strong quality-to-latency trade-off for document retrieval tasks. For higher accuracy at the cost of speed, `all-mpnet-base-v2` is a direct drop-in replacement.

**Why ChromaDB over FAISS?**
ChromaDB provides a persistent, disk-backed store with a clean collection API. FAISS requires manual index serialisation and deserialisation. For local development and reproducible retrieval, ChromaDB reduces operational overhead significantly.

**Why Ollama over an API-hosted LLM?**
Zero network dependency, no token cost, no data leaving the machine. The pipeline is designed to work entirely on local hardware — this is a deliberate constraint, not a limitation.

**Why percentile similarity scores instead of raw L2 distances?**
L2 distances from ChromaDB are not interpretable without context. Converting to a 0–1 similarity scale makes retrieval quality immediately readable in the terminal output.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Managed via `.python-version` |
| [uv](https://docs.astral.sh/uv/) | Fast Python package and environment manager |
| [Ollama](https://ollama.com/download) | Local LLM runtime — must be running before `main.py` is executed |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/neo-bumblebee-ai/traditional-rag-pipeline.git
cd traditional-rag-pipeline
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Pull an Ollama model

```bash
ollama pull llama3.2        # ~2 GB — recommended default
ollama pull llama3.2:1b     # ~800 MB — faster on lower-end hardware
ollama pull mistral         # alternative
```

### 4. Add your documents

```
data/
├── pdf_files/      ← drop any PDF here
└── text_files/     ← or plain .txt files
```

### 5. Run the pipeline

```bash
uv run python main.py
```

First run ingests, embeds, and persists all documents automatically. Subsequent runs skip ingestion and go directly to the query loop.

---

## Configuration

All settings are centralised at the top of `main.py`:

```python
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # any sentence-transformers model
OLLAMA_MODEL     = "llama3.2"            # any model from `ollama list`
CHUNK_SIZE       = 1000                   # characters per chunk
CHUNK_OVERLAP    = 200                    # overlap between consecutive chunks
TOP_K            = 5                      # chunks retrieved per query
```

### Swap the LLM

```bash
ollama pull mistral
```

```python
OLLAMA_MODEL = "mistral"
```

### Use a higher-quality embedding model

```python
EMBEDDING_MODEL = "all-mpnet-base-v2"   # better quality, higher latency
```

> If the embedding model is changed, delete `data/vector_store/` and re-run to rebuild embeddings with the new model dimensions.

### Add new documents

1. Drop files into `data/pdf_files/` or `data/text_files/`
2. Delete `data/vector_store/` to trigger re-ingestion
3. Run `uv run python main.py`

---

## Project Structure

```
traditional-rag-pipeline/
│
├── main.py                      # Full pipeline — entry point
│
├── notebooks/
│   ├── document.ipynb           # Step-by-step: document loading and structure
│   └── pdf_load.ipynb           # Step-by-step: chunking, embedding, retrieval
│
├── data/
│   ├── pdf_files/               # Input PDFs (not committed)
│   ├── text_files/              # Sample plain-text documents
│   └── vector_store/            # ChromaDB persisted index (not committed)
│
├── .github/
│   ├── ISSUE_TEMPLATE/          # Bug report and feature request templates
│   └── pull_request_template.md
│
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # pip-compatible dependency list
└── uv.lock                      # Locked dependency graph
```

---

## Troubleshooting

**`Failed to connect to Ollama`**
Ollama is not running. Start it with:
```bash
ollama serve
```

**`Collection has 0 docs` on restart**
The vector store was deleted or the path was changed. Verify `VECTOR_STORE_DIR` in `main.py` points to the correct location.

**Slow responses**
- Use a smaller model: `ollama pull llama3.2:1b`
- Reduce `TOP_K` from 5 to 3 to limit the context sent to the LLM

**Poor retrieval quality**
- Reduce `CHUNK_SIZE` to 500 for more granular retrieval
- Switch to `all-mpnet-base-v2` for higher-quality embeddings (delete vector store first)

---

## Roadmap

- [ ] Re-ranking with a cross-encoder model
- [ ] Hybrid search (dense + BM25 sparse retrieval)
- [ ] Streamlit web UI
- [ ] Conversation memory for multi-turn follow-up questions
- [ ] RAGAS evaluation suite for retrieval and generation quality scoring

---

## Contributing

Contributions are welcome. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the [MIT License](LICENSE).
