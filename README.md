# Traditional RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) pipeline built from scratch — no managed vector database, no cloud LLM, no black-box framework. Every component is explicit and controllable.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                         │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────────┐ │
│  │  PDFs &  │───▶│  Chunk   │───▶│  Embed                │ │
│  │  .txt    │    │ (1000ch/ │    │  (all-MiniLM-L6-v2)   │ │
│  │  files   │    │  200 ov) │    │  sentence-transformers│ │
│  └──────────┘    └──────────┘    └──────────┬────────────┘ │
│                                             │               │
│                                             ▼               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │  Answer  │◀───│  Ollama  │◀───│  ChromaDB            │  │
│  │          │    │ llama3.2 │    │  (persisted locally) │  │
│  └──────────┘    └──────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Stack

| Component | Tool |
|-----------|------|
| Document loading | LangChain (`PyPDFLoader`, `TextLoader`) |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector store | ChromaDB (local, persisted) |
| LLM | Ollama — `llama3.2` (fully local, no API key needed) |
| Package manager | `uv` |

## Setup

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download) with `llama3.2` pulled

```bash
ollama pull llama3.2
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Add documents

Place your files in:

```
data/
├── pdf_files/      # PDF documents
└── text_files/     # Plain text files
```

### 4. Run

```bash
uv run python main.py
```

On first run the pipeline ingests and embeds all documents automatically. Subsequent runs load from the persisted vector store and go straight to the query loop.

```
=== Traditional RAG Pipeline ===

Loading embedding model: all-MiniLM-L6-v2 …
Vector store ready — 847 docs in collection 'pdf_chunks'.
Vector store already populated — skipping ingestion.

Ollama model : llama3.2
Type a question or 'quit' to exit.

Question: How do I clean the Franke A400 coffee machine?

Top 5 retrieved chunks:
  1. [0.772] Franke_A400_User_Manual_EN.pdf p.42 — The machine automatically rinses…
  ...

Generating answer with Ollama …

Answer:
To clean the Franke A400, navigate to the cleaning menu and select...
```

## Project structure

```
traditional-rag-pipeline/
├── main.py              # End-to-end pipeline (ingest → retrieve → generate)
├── notebooks/
│   ├── document.ipynb   # Document loading exploration
│   └── pdf_load.ipynb   # PDF ingestion, chunking, embedding, retrieval
├── data/
│   └── text_files/      # Sample text documents (committed)
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## How it works

1. **Ingest** — PDFs and text files are loaded and split into overlapping chunks
2. **Embed** — each chunk is encoded to a 384-dimensional vector using a local sentence-transformer model
3. **Store** — vectors are persisted to ChromaDB on disk (no re-embedding on restart)
4. **Retrieve** — at query time, the question is embedded and the top-5 closest chunks are fetched by cosine similarity
5. **Generate** — retrieved chunks are passed as context to a local Ollama LLM which produces a grounded answer

## Sample documents

The notebooks were developed and tested against:

- Franke A400 Espresso Machine User Manual
- WMF 1500S User Manual
- Nuova Simonelli Appia II (2–3 Group) Manual
- Rancilio Classe 5 User Manual

## Licence

MIT
