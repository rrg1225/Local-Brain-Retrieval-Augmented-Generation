"""Application configuration for Local Brain RAG.

The module loads provider keys, retrieval settings, model names, and workspace
paths from environment variables. Runtime workspace changes are persisted in a
small JSON file protected by a file lock so the Streamlit app and watcher can
share state safely.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from filelock import FileLock


@dataclass
class AppConfig:
    api_key: str
    """Gemini / Google GenAI API key. Mirrored to GOOGLE_API_KEY when present."""

    dashscope_api_key: str
    """DashScope / Qwen API key used by optional provider failover."""

    proxy_base: str
    """Optional Google GenAI compatible base URL, for example a proxy endpoint."""

    http_proxy: str | None
    """Optional local HTTP proxy. Applied to HTTP_PROXY and HTTPS_PROXY."""

    workspaces: list[Path]
    chroma_dir: str
    llm_model: str
    local_embed_model: str
    local_rerank_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    debounce: float
    memory_token_limit: int


def _workspace_file() -> Path:
    return Path.cwd() / "workspaces.json"


def _load_persistent_workspaces() -> list[Path]:
    """Load saved workspace paths from the current workspace state file."""
    ws_file = _workspace_file()
    if not ws_file.exists():
        return []

    try:
        data = json.loads(ws_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    paths: list[Path] = []
    if isinstance(data, dict):
        for space_paths in data.values():
            if isinstance(space_paths, list):
                paths.extend(Path(p).resolve() for p in space_paths if p and Path(p).resolve().is_dir())
    elif isinstance(data, list):
        paths.extend(Path(p).resolve() for p in data if p and Path(p).resolve().is_dir())

    return paths


def save_persistent_workspace(space_name: str, new_path: Path) -> None:
    """Append a workspace path to a named space and invalidate config cache."""
    ws_file = _workspace_file()
    lock = FileLock(str(ws_file) + ".lock")
    resolved = str(new_path.resolve())

    with lock:
        data: dict[str, list[str]] = {}
        if ws_file.exists():
            try:
                loaded = json.loads(ws_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (json.JSONDecodeError, OSError):
                data = {}

        space_paths = data.setdefault(space_name, [])
        if resolved not in space_paths:
            space_paths.append(resolved)
            ws_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            reset_config_cache()


def remove_persistent_workspace(space_name: str, target: Path) -> None:
    """Remove a workspace path from a named space and invalidate config cache."""
    ws_file = _workspace_file()
    if not ws_file.exists():
        return

    lock = FileLock(str(ws_file) + ".lock")
    resolved = str(target.resolve())

    with lock:
        try:
            data = json.loads(ws_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(data, dict):
            return

        space_paths: list[str] = data.get(space_name, [])
        if resolved not in space_paths:
            return

        space_paths.remove(resolved)
        if space_paths:
            data[space_name] = space_paths
        else:
            data.pop(space_name, None)

        ws_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        reset_config_cache()


def apply_http_proxy_env(http_proxy: str | None) -> None:
    """Apply a configured proxy to common Python HTTP client environment names."""
    if not http_proxy or not str(http_proxy).strip():
        return

    proxy = str(http_proxy).strip()
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        os.environ.setdefault("GOOGLE_API_KEY", gemini_key)

    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

    http_proxy = os.getenv("HTTP_PROXY", "").strip() or None
    apply_http_proxy_env(http_proxy)

    workspaces_raw = os.getenv("WORKSPACE_PATHS", "./workspace")
    workspaces = [Path(p.strip()).resolve() for p in workspaces_raw.split(",") if p.strip()]
    workspaces.extend(_load_persistent_workspaces())

    dynamic_dir = Path("./dynamic_workspace").resolve()
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    workspaces.append(dynamic_dir)
    workspaces = list({path.resolve() for path in workspaces})

    return AppConfig(
        api_key=gemini_key or os.getenv("GOOGLE_API_KEY", "").strip(),
        dashscope_api_key=dashscope_key,
        proxy_base=os.getenv("PROXY_URL", "").strip().rstrip("/"),
        http_proxy=http_proxy,
        workspaces=workspaces,
        chroma_dir=os.getenv("CHROMA_PERSIST_DIR", "./.chroma_second_brain"),
        llm_model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
        local_embed_model=os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-large-zh-v1.5"),
        local_rerank_model=os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        chunk_size=_int_env("CHUNK_SIZE", 512, minimum=128, maximum=4096),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 64, minimum=0, maximum=1024),
        top_k=_int_env("TOP_K", 5, minimum=1, maximum=50),
        debounce=_float_env("DEBOUNCE_SECONDS", 60.0, minimum=1.0, maximum=3600.0),
        memory_token_limit=_int_env("MEMORY_TOKEN_LIMIT", 8192, minimum=1024, maximum=65536),
    )


def reset_config_cache() -> None:
    """Clear cached configuration after local workspace state changes."""
    get_config.cache_clear()
