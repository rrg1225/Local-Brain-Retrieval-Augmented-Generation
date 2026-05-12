import os
from pathlib import Path
from dotenv import load_dotenv

# 最先加载 .env，确保后续 os.getenv 能读到用户配置
load_dotenv()

# --- AI 模型缓存路径：优先从 .env 读取，未配置则使用系统默认 ---
cache_dir = os.getenv("HF_HOME", "").strip()
if cache_dir:
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
    os.environ.setdefault("LLAMA_INDEX_CACHE_DIR", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

# 禁用 HTTP/2 协议，防止代理环境下的 SSL UNEXPECTED_EOF 报错
os.environ.setdefault("HTTPX_NO_HTTP2", "1")

"""
兼容入口：推荐使用 `streamlit run app.py`。
"""

from app import run_app

if __name__ == "__main__":
    run_app()
