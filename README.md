# Traditional RAG Pipeline

A fully local, end-to-end Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python. No cloud services, no API keys, no black-box abstractions — every component is explicit and swappable.

Ask questions about your own PDF and text documents. The pipeline finds the most relevant passages and generates a grounded answer using a local LLM.

---

## How it works

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
**On subsequent runs:** embeddings are loaded from disk instantly — no re-processing.

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| Document loading | LangChain | `PyPDFLoader`, `TextLoader` |
| Text splitting | `RecursiveCharacterTextSplitter` | Configurable chunk size & overlap |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2`, runs on CPU |
| Vector store | ChromaDB | Persisted locally to disk |
| LLM | Ollama | `llama3.2` by default, fully local |
| Package manager | `uv` | Fast Python package manager |

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- [Ollama](https://ollama.com/download) — local LLM runner

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/neo-bumblebee-ai/traditional-rag-pipeline.git
cd traditional-rag-pipeline
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Install and start Ollama

Download from [ollama.com/download](https://ollama.com/download), then pull a model:

```bash
ollama pull llama3.2        # ~2GB, good balance of speed and quality
# or
ollama pull mistral         # alternative option
# or
ollama pull llama3.2:1b     # ~800MB, faster on low-end machines
```

Ollama runs as a background service automatically after install.

### 4. Add your documents

Place your files in the `data/` directory:

```
data/
├── pdf_files/          ← drop any PDF here
│   └── your_doc.pdf
└── text_files/         ← or plain .txt files
    └── your_notes.txt
```

### 5. Run

```bash
uv run python main.py
```

That's it. On first run the pipeline ingests and embeds everything automatically.

---

## Usage

```
=== Traditional RAG Pipeline ===

Loading embedding model: all-MiniLM-L6-v2 …
Vector store ready — 847 docs in collection 'pdf_chunks'.
Vector store already populated — skipping ingestion.

Ollama model : llama3.2
Type a question or 'quit' to exit.

Question: What are the daily cleaning steps for the coffee machine?

Top 5 retrieved chunks:
  1. [0.772] Franke_A400_User_Manual_EN.pdf p.42 — The machine automatically rinses…
  2. [0.754] Franke_A400_User_Manual_EN.pdf p.43 — Milk system cleaning procedure…
  ...

Generating answer with Ollama …

Answer:
The daily cleaning steps include: 1) Run the automatic rinse cycle at the
end of each day by navigating to the Cleaning menu...

────────────────────────────────────────────────
Question: quit
Goodbye.
```

---

## Configuration

All settings are at the top of `main.py`:

```python
VECTOR_STORE_DIR = "data/vector_store"   # where ChromaDB persists
PDF_DIR          = "data/pdf_files"       # PDF source folder
TEXT_DIR         = "data/text_files"      # text file source folder
COLLECTION_NAME  = "pdf_chunks"           # ChromaDB collection name
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"    # sentence-transformers model
OLLAMA_MODEL     = "llama3.2"            # any model from `ollama list`
CHUNK_SIZE       = 1000                   # characters per chunk
CHUNK_OVERLAP    = 200                    # overlap between chunks
TOP_K            = 5                      # chunks retrieved per query
```

### Changing the LLM

Pull any model with Ollama and update `OLLAMA_MODEL`:

```bash
ollama pull mistral
```

```python
OLLAMA_MODEL = "mistral"
```

### Using different documents

1. Drop new files into `data/pdf_files/` or `data/text_files/`
2. Delete `data/vector_store/` to force re-ingestion
3. Run `uv run python main.py` — it will re-embed everything

### Changing the embedding model

Any model from [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) works:

```python
EMBEDDING_MODEL = "all-mpnet-base-v2"   # higher quality, slower
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # default — fast, good quality
```

> **Note:** If you change the embedding model, delete `data/vector_store/` and re-ingest, since vectors from different models are incompatible.

---

## Project structure

```
traditional-rag-pipeline/
│
├── main.py                     # Full pipeline — run this
│
├── notebooks/
│   ├── document.ipynb          # Step-by-step: document loading & structure
│   └── pdf_load.ipynb          # Step-by-step: chunking, embedding, retrieval
│
├── data/
│   ├── pdf_files/              # Your PDFs go here (gitignored)
│   ├── text_files/             # Your .txt files go here
│   │   ├── python_intro.txt    # Sample document
│   │   └── machine_learning.txt
│   └── vector_store/           # ChromaDB persisted here (gitignored)
│
├── pyproject.toml              # Dependencies
├── requirements.txt            # pip-compatible dependency list
└── uv.lock                     # Locked dependency versions
```

---

## Troubleshooting

**`Failed to connect to Ollama`**
Ollama isn't running. Start it:
```bash
ollama serve
```

**`Collection has 0 docs` on restart**
The vector store path changed or was deleted. Check that `VECTOR_STORE_DIR` points to the same folder used during ingestion.

**Slow responses**
- Use a smaller model: `ollama pull llama3.2:1b`
- Reduce `TOP_K` from 5 to 3 to send less context to the LLM
- `all-MiniLM-L6-v2` is already the fastest sentence-transformer — don't change this unless you need higher quality

**Poor retrieval quality (wrong chunks returned)**
- Reduce `CHUNK_SIZE` to `500` for more granular retrieval
- Increase `CHUNK_OVERLAP` to `100` to preserve more context across chunk boundaries
- Try a higher-quality embedding model like `all-mpnet-base-v2`

**Non-ASCII characters display as `?` on Windows**
Already handled — the pipeline reconfigures stdout to UTF-8 on startup.

---

## Extending the pipeline

Ideas for taking this further:

- **Re-ranking** — add a cross-encoder to re-score retrieved chunks before generation
- **Hybrid search** — combine embedding search with BM25 keyword search
- **Web UI** — wrap in Streamlit for a chat interface
- **Conversation memory** — track chat history for follow-up questions
- **Evaluation** — use [RAGAS](https://docs.ragas.io) to measure retrieval and answer quality automatically

---

## Licence

MIT — use it however you like.
