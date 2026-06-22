# Local Brain RAG

[![CI](https://github.com/rrg1225/Local-Brain-Retrieval-Augmented-Generation/actions/workflows/ci.yml/badge.svg)](https://github.com/rrg1225/Local-Brain-Retrieval-Augmented-Generation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-6A5ACD)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F61)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)

Local Brain RAG is a privacy-first retrieval-augmented generation workspace for local codebases, documents, PDFs, Word files, and architecture images. It is designed as a personal "second brain" with local indexing, hybrid retrieval, workspace isolation, and Gemini/Qwen provider configuration.

> Resume and interview brief: [PORTFOLIO.md](PORTFOLIO.md)
> Enterprise architecture: [docs/ENTERPRISE_ARCHITECTURE.md](docs/ENTERPRISE_ARCHITECTURE.md)

## Highlights

- Local Chroma persistence for private project knowledge.
- Hybrid retrieval with code-aware tokenization, BM25, vector search, RRF fusion, and reranking.
- Java and Vue structure-aware chunking.
- PDF, Word, Markdown, JSON, YAML, Python, Java, Vue, and image-caption ingestion paths.
- Workspace watching with debounce and incremental updates.
- Gemini and Qwen configuration slots for provider failover.
- Lightweight CI syntax validation that avoids model downloads.

## RAG Pipeline

```text
workspace files
  -> parser / captioner
  -> chunker with metadata
  -> embedding
  -> Chroma persistence
  -> BM25 + vector retrieval
  -> RRF fusion
  -> rerank
  -> context assembly
  -> answer generation
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python doctor.py
streamlit run main.py
```

## Configuration

Copy `.env.example` to `.env` and set the values you need.

```env
GEMINI_API_KEY=your_gemini_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
WORKSPACE_PATHS=./workspace
CHROMA_PERSIST_DIR=./.chroma_second_brain
TOP_K=5
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

## Important Paths

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit UI |
| `rag_engine.py` | Ingestion, chunking, retrieval, rerank, and chat engine |
| `config.py` | Environment loading and workspace persistence |
| `watcher.py` | File watching and incremental refresh |
| `doctor.py` | Lightweight environment diagnostics |

## Verification

```bash
python -m py_compile app.py config.py doctor.py main.py rag_engine.py watcher.py
python doctor.py
```

## Privacy Notes

- Local files are indexed into a local Chroma directory.
- API keys belong in `.env`, not in committed files.
- `dynamic_workspace/`, `.chroma_second_brain/`, chat history, and index state files are ignored by Git.

## License

MIT
