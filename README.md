# Local Brain RAG

[简体中文](#简体中文) | [English](#english)

Local Brain RAG is a privacy-first retrieval-augmented generation workspace for local codebases and knowledge folders. It combines LlamaIndex, ChromaDB, hybrid retrieval, workspace watching, Gemini/Qwen provider configuration, and a Streamlit UI.

> Resume and interview brief: [PORTFOLIO.md](PORTFOLIO.md)
> Enterprise architecture: [docs/ENTERPRISE_ARCHITECTURE.md](docs/ENTERPRISE_ARCHITECTURE.md)

---

## 简体中文

### 项目亮点

- **本地知识库优先**：面向代码仓库、技术文档、PDF、Word 和图片说明的个人第二大脑。
- **混合检索路径**：代码感知 tokenizer、BM25、向量检索和 RRF 融合，兼顾关键字与语义召回。
- **Chroma 持久化**：本地向量库按空间隔离，便于不同项目独立索引。
- **结构化切分**：Java/Vue 有专门切分策略，普通文本使用句子切分。
- **工作区监听**：Watchdog 监听文件变化，支持增量重建和删除清理。
- **配置诊断**：`doctor.py` 可在安装完整 GPU 依赖前检查 `.env`、工作区和 Chroma 路径。
- **轻量 CI**：CI 做 Python 语法检查和 doctor 单元测试，不触发模型下载。

### 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python doctor.py
streamlit run main.py
```

### 环境变量

复制 `.env.example` 为 `.env`，再按需配置：

```env
GEMINI_API_KEY=your_gemini_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
WORKSPACE_PATHS=./workspace
CHROMA_PERSIST_DIR=./.chroma_second_brain
TOP_K=5
```

### RAG 路径

```text
workspace files
-> parse PDF / Word / code / image captions
-> chunk with metadata
-> embed and persist in Chroma
-> BM25 + vector retrieval
-> RRF fusion and rerank
-> context assembly
-> Gemini / Qwen answer
```

### 验证

```bash
python -m py_compile app.py config.py doctor.py main.py rag_engine.py watcher.py
python -m unittest discover -s test
```

---

## English

### Highlights

- **Local-first knowledge workspace** for codebases, docs, PDFs, Word files, and image captions.
- **Hybrid retrieval** with code-aware tokenization, BM25, vector search, and RRF fusion.
- **Persistent Chroma collections** for project-space isolation.
- **Structure-aware chunking** for Java and Vue, plus sentence splitting for general text.
- **Workspace watching** for incremental re-indexing and delete cleanup.
- **Environment doctor** that checks configuration before installing the full GPU/model stack.
- **Lightweight CI** that avoids model downloads while still checking syntax and diagnostics.

### Scripts

```bash
python doctor.py
streamlit run main.py
python -m unittest discover -s test
```

### Repository Topics

`rag`, `llamaindex`, `chromadb`, `streamlit`, `gemini`, `qwen`, `hybrid-search`
