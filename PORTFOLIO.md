# Portfolio Brief: Local Brain RAG

## Resume Bullets

- Built a privacy-first local RAG system for code and technical documents with ChromaDB persistence, hybrid retrieval, workspace isolation, file watching, and multi-provider LLM failover.
- Implemented code-aware indexing across PDFs, Word files, source code, and architecture images with incremental updates and source trace visibility.
- Designed the system for local GPU inference and developer knowledge retrieval, keeping sensitive code and documents on the user's machine.

## What This Proves

- Retrieval-augmented generation architecture and ingestion pipelines.
- Hybrid search, reranking, workspace isolation, and incremental indexing.
- Practical privacy and resilience design for developer tools.

## Verification

```bash
python -m py_compile app.py config.py doctor.py main.py rag_engine.py watcher.py
```

Full runtime validation requires local model credentials and GPU/runtime dependencies from `.env.example`.

## Interview Talking Points

- Why hybrid retrieval is useful for codebases where exact identifiers matter.
- How workspace isolation prevents cross-project context leakage.
- How file watching keeps a local knowledge base fresh without full reindexing.
