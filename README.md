<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/GPU-RTX_5060-ED1C24?style=for-the-badge&logo=nvidia&logoColor=white" alt="RTX 5060">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Privacy-100%25_Local-2EA44F?style=for-the-badge&logo=lock&logoColor=white" alt="Privacy">
</p>

<h1 align="center">🧠 Local Brain RAG</h1>
<h3 align="center">Privacy-First Code &amp; Knowledge Retrieval — Powered by Your Own GPU</h3>

<p align="center">
  <b>100% Local Inference · Zero Data Leakage · Built for Developers</b><br>
  An industrial-grade Retrieval-Augmented Generation (RAG) system that turns your<br>
  entire codebase, technical docs, and architecture diagrams into a conversational <i>second brain</i>.
</p>

<p align="center">
  <sub>English · <a href="#中文">中文</a></sub>
</p>

> Resume and interview brief: [PORTFOLIO.md](PORTFOLIO.md)

---

## ✨ Highlights

<table>
  <tr>
    <td width="50%">
      <h4>🔒 Privacy Isolation</h4>
      <p>All embeddings and reranking run on <b>your local GPU</b>. Code snippets, documents, and conversations never leave your machine. ChromaDB multi-collection architecture provides physical isolation per project space.</p>
    </td>
    <td width="50%">
      <h4>⚡ Hybrid Retrieval</h4>
      <p><b>BM25 (code-aware tokenizer) + Vector (BGE) + RRF fusion</b>. The custom <code>code_aware_tokenize</code> combines jieba Chinese segmentation with regex code identifier extraction — hitting class names, method signatures, and stack traces with surgical precision.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4>🚀 Hardware-Optimized</h4>
      <p>Tuned for <b>NVIDIA RTX 5060 · PyTorch 2.6 · CUDA 12.8</b>. Concurrent batch ingestion with thread-pool workers, incremental mtime-based scanning, and BM25 instance caching that avoids O(N) vocabulary rebuilds on every chat turn.</p>
    </td>
    <td width="50%">
      <h4>📄 Multi-Format Parsing</h4>
      <p>PyMuPDF for PDF, python-docx for Word, Gemini Vision for architecture diagrams/screenshots, and tree-sitter <code>CodeSplitter</code> for structural Java/Vue chunking that preserves class and method boundaries.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4>🛡️ Graceful Degradation</h4>
      <p>LLM quota exhausted? The system <b>auto-fails over</b> from Gemini to Qwen (and back) mid-conversation without losing context. BM25 init failure? Falls back to pure vector retrieval. Resilient by design.</p>
    </td>
    <td width="50%">
      <h4>🪝 Live File Watching</h4>
      <p>Watchdog monitors workspace directories with configurable debounce. Files are incrementally re-indexed on change; deleted files are pruned from the vector store. Ghost node cleanup runs on every startup.</p>
    </td>
  </tr>
</table>

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/LlamaIndex-0.12+-6A5ACD?style=flat-square&logo=llamaindex&logoColor=white" alt="LlamaIndex">
  <img src="https://img.shields.io/badge/ChromaDB-0.5+-FF6F61?style=flat-square&logo=chroma&logoColor=white" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/HuggingFace-BGE_Large-brightgreen?style=flat-square&logo=huggingface&logoColor=white" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/Qwen-Plus%2FMax-FF6A00?style=flat-square&logo=alibabacloud&logoColor=white" alt="Qwen">
</p>

| Layer | Technology | Role |
|-------|-----------|------|
| **Orchestration** | LlamaIndex 0.12+ | Index pipeline, Condense+Context chat engine, RRF fusion |
| **LLM** | Gemini 2.5 Flash · Qwen-Plus/Max | Conversational generation with seamless failover |
| **Embedding** | `BAAI/bge-large-zh-v1.5` (HuggingFace) | Local GPU vector embeddings |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (SentenceTransformer) | Local GPU cross-encoder reranking |
| **Vector Store** | ChromaDB (Persistent Client) | On-disk persistence, multi-collection isolation |
| **Hybrid Search** | BM25 + Vector + Reciprocal Rank Fusion | Code-aware tokenizer + semantic dual-pathway |
| **UI** | Streamlit 1.40+ | Streaming chat, source trace panel, file upload, ZIP extraction |
| **File Watching** | Watchdog 6.0+ | Debounced incremental indexing on filesystem events |
| **Doc Parsing** | PyMuPDF · python-docx · tree-sitter | PDF, Word, structural code splitting |
| **Tokenization** | Jieba + custom regex | Chinese semantic + code identifier extraction |

---

## 📂 Project Structure

```
bendiRAG/
├── main.py                 # Entry point — loads .env, sets cache paths, launches Streamlit
├── app.py                  # Streamlit UI — streaming chat, space management, file upload
├── config.py               # Configuration — .env loading, workspace persistence
├── rag_engine.py           # RAG core — indexing, BM25+Vector retrieval, chat engine
├── watcher.py              # Watchdog daemon — debounced incremental file indexing
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template (safe to commit)
├── .env                    # Local secrets (excluded from Git)
├── .gitignore
├── .chroma_second_brain/   # ChromaDB persistent storage
├── dynamic_workspace/      # Uploaded files & ZIP extraction scratch space
├── workspaces.json         # Per-space workspace path registry
└── .index_state_*.json     # Per-space mtime index state (auto-generated)
```

| File | Responsibility |
|------|---------------|
| `main.py` | Boot: loads `.env`, syncs `HF_HOME` / cache env vars, delegates to `run_app()` |
| `app.py` | Full UI lifecycle: space switching, streaming output, source trace expander, chat persistence, file upload/ZIP extraction, workspace CRUD |
| `config.py` | `AppConfig` dataclass, `dotenv` loading, `workspaces.json` persistence, HTTP proxy injection |
| `rag_engine.py` | RAG infrastructure: GPU embedding/reranker init, BM25+Vector hybrid retrieval, ChromaDB collection management, ghost node cleanup, incremental scan, code-aware chunking, image captioning |
| `watcher.py` | Watchdog event handler: debounced batch upsert, file deletion sync to vector store |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** · **CUDA 12.8** · **NVIDIA GPU** (RTX 3060 or above recommended)
- [Google AI Studio API Key](https://aistudio.google.com/) (free tier available)
- (Optional) [Aliyun DashScope API Key](https://bailian.console.aliyun.com/) for Qwen failover

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n bendirag python=3.11 -y
conda activate bendirag

# Install PyTorch 2.6 with CUDA 12.8 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install project dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the template
cp .env.example .env

# Edit .env with your API keys
# GEMINI_API_KEY=...          (required)
# DASHSCOPE_API_KEY=...       (optional — Qwen failover)
# WORKSPACE_PATHS=...         (directories to index)
```

### 3. Launch

```bash
streamlit run main.py
# or: python main.py
```

Open `http://localhost:8501` and start conversing with your codebase.

### 4. Lightweight Doctor

Run this before installing the full GPU stack or launching Streamlit:

```bash
python doctor.py
```

It checks API key placeholders, workspace paths, Chroma persistence paths, and required project files without importing heavy ML packages.

---

## 💡 Usage

### First Run
On startup, the app scans all `WORKSPACE_PATHS` directories, chunks every indexable file (code, docs, images), and ingests them into the vector store. A progress bar shows batch insertion status in real time.

### Project Spaces
| Action | How | Effect |
|--------|-----|--------|
| **Switch space** | Sidebar dropdown | Switches to an isolated ChromaDB collection + chat history |
| **Create space** | `＋` button | Spawns a blank collection |
| **Destroy space** | `🗑️` button | Physically deletes collection + index state + chat history |

### Knowledge Base
- **File upload**: Drag & drop files or ZIP archives. ZIPs auto-extract (with Chinese filename encoding fix).
- **Watch directory**: Enter a local folder path — one-click full scan + continuous watchdog monitoring.
- **Remove workspace**: Click `✕` to purge all vector nodes under a directory.

### Chat & Source Tracing
Ask natural-language questions like:
- *"Explain the overall architecture and tech stack of this project."*
- *"What does the recent change to UserService do?"*
- *"Map out the database table dependencies."*

Each response includes a **source trace panel** showing the originating file path and a 300-character code snippet preview.

---

## 🔧 Model Switching & Failover

| Model | Provider | Required Key |
|-------|----------|-------------|
| `gemini-2.5-flash` | Google | `GEMINI_API_KEY` |
| `qwen-plus` | Aliyun DashScope | `DASHSCOPE_API_KEY` |
| `qwen-max` | Aliyun DashScope | `DASHSCOPE_API_KEY` |

**Auto-failover**: If the active model returns a quota/resource error (HTTP 429), the system automatically retries with the alternate provider — no manual intervention needed.

---

## 🧪 Under the Hood

### Code-Aware Tokenizer
```python
def code_aware_tokenize(text: str) -> list[str]:
    code_tokens = re.findall(r"[a-zA-Z0-9_]+", text)        # camelCase / snake_case
    chinese_tokens = [w for w in jieba.lcut(text) if ...]   # Chinese semantics
    return code_tokens + chinese_tokens
```
BM25 keeps identifiers like `getUserById` intact instead of splitting them into `["get", "User", "By", "Id"]` — a game-changer for code search.

### Data Lifecycle
```
File created   → incremental_upsert_file → update index_state
File modified  → delete_ref_doc → re-insert → update index_state mtime
File deleted   → watchdog on_deleted → delete_ref_doc
Workspace rm   → delete_workspace_nodes → batch purge via index_state
Space destroy  → delete_collection → wipe Chroma + state + chat history
Ghost cleanup  → cleanup_ghost_nodes → scan index_state for vanished files
```

### BM25 Cache Strategy
BM25 retriever construction requires an O(N) scan over the full document vocabulary. The system caches `(index_state_mtime, BM25Retriever)` globally and only rebuilds when the space's index state file changes — reuse across chat turns, no per-message penalty.

---

## 📜 License

MIT License — use freely. Your data never leaves your machine.

---

<h2 id="中文">🇨🇳 中文说明</h2>

<h3 align="center">Local Brain RAG（bendiRAG）</h3>
<h4 align="center">隐私优先的本地代码与知识库检索助手</h4>

**核心定位**：面向软件工程师的工业级 RAG 系统。所有 Embedding 和 Rerank 推理均在本地 GPU 上完成，代码片段绝不离开你的机器。

**五大亮点**：
- **隐私隔离** — ChromaDB 多 Collection 物理隔离，销毁空间即完整擦除
- **混合检索** — BM25（代码分词）+ Vector（BGE 语义）+ RRF 倒数排序融合
- **硬件压榨** — 针对 RTX 5060 / PyTorch 2.6 / CUDA 12.8 调优，并发批写入 + BM25 缓存复用
- **多格式解析** — PyMuPDF 解析 PDF、python-docx 解析 Word、Gemini 视觉模型描述截图、tree-sitter 结构化切分 Java/Vue
- **自动容灾** — Gemini 配额耗尽时自动切换千问接力，对话不中断

**快速安装**：
```bash
conda create -n bendirag python=3.11 -y && conda activate bendirag
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
cp .env.example .env   # 编辑填入 GEMINI_API_KEY
python doctor.py       # 轻量环境预检
streamlit run main.py
```

**技术栈**：LlamaIndex 编排 · ChromaDB 向量存储 · Streamlit 界面 · HuggingFace BGE 本地 Embedding/Rerank · Gemini 2.5 Flash + 千问 Plus/Max 双引擎 · Watchdog 文件监听 · Jieba 中文分词

---

<p align="center">
  <sub>Built with ❤️ for developers who care about privacy and performance.</sub>
</p>
