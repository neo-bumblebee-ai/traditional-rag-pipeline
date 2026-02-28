# Traditional RAG Pipeline

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.com)
[![Vector DB](https://img.shields.io/badge/vectordb-ChromaDB-blueviolet.svg)](https://www.trychroma.com)
[![Embeddings](https://img.shields.io/badge/embeddings-sentence--transformers-yellow.svg)](https://www.sbert.net)

A fully local, end-to-end Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python. No cloud services, no API keys, no black-box abstractions — every component is explicit and swappable.

> Drop in your own PDFs or text files, ask questions in plain English, and get grounded answers backed by your documents.

---

## Demo

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

## Architecture

```
Your Documents (PDFs / .txt)
        │
        ▼
┌───────────────────┐
│   Document Loader │  LangChain PyPDFLoader / TextLoader
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Text Splitter    │  RecursiveCharacterTextSplitter
│  chunk=1000       │  overlap=200 chars
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Embedding Model  │  sentence-transformers/all-MiniLM-L6-v2
│  (runs locally)   │  → 384-dimensional vectors
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   ChromaDB        │  Persisted to disk — no re-embedding on restart
│  Vector Store     │
└────────┬──────────┘
         │  (at query time)
         ▼
┌───────────────────┐
│   Retriever       │  Cosine similarity → top-5 chunks
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Ollama LLM       │  llama3.2 (or any model you pull)
│  (runs locally)   │  Context-grounded answer generation
└───────────────────┘
```

**On first run:** documents are loaded, chunked, embedded, and stored in ChromaDB.
**On subsequent runs:** embeddings are reloaded from disk instantly — no re-processing.

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| Document loading | LangChain | `PyPDFLoader`, `TextLoader` |
| Text splitting | `RecursiveCharacterTextSplitter` | Configurable chunk size & overlap |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2`, CPU-friendly |
| Vector store | ChromaDB | Persisted locally to disk |
| LLM | Ollama | `llama3.2` by default, fully local |
| Package manager | `uv` | Fast Python package manager |

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- [Ollama](https://ollama.com/download) — local LLM runner

---

## Quick start

### 1. Clone the repo

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
ollama pull llama3.2        # ~2GB — recommended
ollama pull llama3.2:1b     # ~800MB — faster on low-end hardware
ollama pull mistral         # alternative
```

### 4. Add your documents

```
data/
├── pdf_files/      ← drop any PDF here
└── text_files/     ← or plain .txt files
```

### 5. Run

```bash
uv run python main.py
```

First run ingests and embeds everything automatically. Subsequent runs skip straight to the query loop.

---

## Configuration

All settings live at the top of `main.py`:

```python
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # any sentence-transformers model
OLLAMA_MODEL     = "llama3.2"            # any model from `ollama list`
CHUNK_SIZE       = 1000                   # characters per chunk
CHUNK_OVERLAP    = 200                    # overlap between chunks
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
EMBEDDING_MODEL = "all-mpnet-base-v2"   # better quality, slower
```

> If you change the embedding model, delete `data/vector_store/` and re-run to rebuild embeddings.

### Add new documents

1. Drop files into `data/pdf_files/` or `data/text_files/`
2. Delete `data/vector_store/` to trigger re-ingestion
3. Run `uv run python main.py`

---

## Project structure

```
traditional-rag-pipeline/
│
├── main.py                      # Full pipeline — run this
│
├── notebooks/
│   ├── document.ipynb           # Step-by-step: document loading & structure
│   └── pdf_load.ipynb           # Step-by-step: chunking, embedding, retrieval
│
├── data/
│   ├── pdf_files/               # Your PDFs go here (not committed)
│   ├── text_files/              # Sample .txt documents
│   └── vector_store/            # ChromaDB persisted here (not committed)
│
├── .github/
│   ├── ISSUE_TEMPLATE/          # Bug report & feature request templates
│   └── pull_request_template.md
│
├── CONTRIBUTING.md              # How to contribute
├── LICENSE                      # MIT
├── pyproject.toml               # Project metadata & dependencies
├── requirements.txt             # pip-compatible dependency list
└── uv.lock                      # Locked dependency versions
```

---

## Troubleshooting

**`Failed to connect to Ollama`**
Ollama isn't running. Start it:
```bash
ollama serve
```

**`Collection has 0 docs` on restart**
The vector store was deleted or the path changed. Check `VECTOR_STORE_DIR` in `main.py`.

**Slow responses**
- Use a smaller model: `ollama pull llama3.2:1b`
- Reduce `TOP_K` from 5 to 3 to send less context

**Poor retrieval quality**
- Reduce `CHUNK_SIZE` to `500` for more granular retrieval
- Try `all-mpnet-base-v2` for higher-quality embeddings

---

## Roadmap

- [ ] Re-ranking with a cross-encoder
- [ ] Hybrid search (dense + BM25)
- [ ] Streamlit web UI
- [ ] Conversation memory for follow-up questions
- [ ] RAGAS evaluation suite

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the [MIT License](LICENSE).
