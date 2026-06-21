# Enterprise Architecture

## Enterprise Positioning

Local Brain RAG is a privacy-first developer knowledge system. It is designed for environments where source code, design documents, architecture diagrams, and operational notes must remain local while still being searchable through conversational retrieval.

## Architecture Boundaries

- **UI layer**: Streamlit workspace for chat, uploads, source traces, and workspace switching.
- **Configuration layer**: `.env` and workspace registry for provider and filesystem settings.
- **Ingestion layer**: file parsing for code, PDFs, Word documents, images, and ZIP uploads.
- **Retrieval layer**: ChromaDB vector search, BM25 lexical search, and reciprocal rank fusion.
- **Reranking layer**: local reranker for precision improvement.
- **Watcher layer**: incremental file updates and deletion cleanup.

## Enterprise Extension Path

1. Replace local-only workspace registry with a signed workspace manifest and policy engine.
2. Add document-level ACLs and per-space encryption.
3. Add ingestion queues for large repositories and background reindex jobs.
4. Add retrieval evaluation sets for code Q&A, architecture Q&A, and incident support.
5. Add observability around chunk counts, retrieval latency, reranker latency, and answer source coverage.

## SLO and Observability

- **Freshness target**: changed files indexed within 60 seconds after filesystem events.
- **Retrieval latency target**: p95 hybrid retrieval under 2 seconds for ordinary workspaces.
- **Privacy target**: local files never leave the machine except explicit LLM provider calls configured by the user.
- **Core dashboards**: indexed documents, chunk count, vector store size, retrieval latency, reranker failures, provider failover rate.

## Security Model

- `.env` contains provider keys and stays excluded from Git.
- Workspace isolation prevents accidental cross-project retrieval.
- Dynamic upload workspaces are treated as scratch data and excluded from Git.
- Future enterprise mode should enforce ACL-aware retrieval and source-level redaction.

## Interview-Level Design Rationale

The architecture combines lexical and semantic retrieval because code search often depends on exact identifiers while document search benefits from embeddings. Hybrid retrieval gives better coverage than either path alone.
