"""
应用配置：从 .env 加载 API Key、本地 Embedding/Rerank 模型、工作区路径等，并注入系统级 HTTP(S) 代理环境变量。
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
    """Gemini / Google GenAI API Key（可与 GOOGLE_API_KEY 同步）。"""

    dashscope_api_key: str
    """阿里云 DashScope / 百炼 API Key（通义千问对话）。"""

    proxy_base: str
    """Cloudflare Worker 根 URL（Google GenAI HttpOptions.base_url）。"""

    http_proxy: str | None
    """本地 HTTP 隧道代理，如 Clash；写入 HTTP_PROXY / HTTPS_PROXY 等。"""

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


def _load_persistent_workspaces() -> list[Path]:
    """从 workspaces.json 加载所有空间的工作区路径（并集），兼容旧版列表格式。"""
    ws_file = Path.cwd() / "workspaces.json"
    if not ws_file.exists():
        return []
    try:
        with open(ws_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 新版：{"space_name": ["/path1", ...]}
            paths: list[Path] = []
            for space_paths in data.values():
                if isinstance(space_paths, list):
                    for p in space_paths:
                        if p and Path(p).resolve().is_dir():
                            paths.append(Path(p).resolve())
            return paths
        if isinstance(data, list):
            # 旧版：["/path1", ...] — 自动迁移到新版格式
            return [Path(p).resolve() for p in data if p and Path(p).resolve().is_dir()]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_persistent_workspace(space_name: str, new_path: Path) -> None:
    """将路径追加到指定空间的工作区列表并保存（文件锁防并发）。"""
    ws_file = Path.cwd() / "workspaces.json"
    lock = FileLock(str(ws_file) + ".lock")
    resolved = str(new_path.resolve())

    with lock:
        data: dict = {}
        if ws_file.exists():
            try:
                data = json.loads(ws_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    # 旧格式自动迁移
                    data = {}
            except (json.JSONDecodeError, OSError):
                data = {}

        space_paths: list[str] = data.setdefault(space_name, [])
        if resolved not in space_paths:
            space_paths.append(resolved)
            ws_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reset_config_cache()


def remove_persistent_workspace(space_name: str, target: Path) -> None:
    """从指定空间的工作区列表中移除路径并保存（文件锁防并发）；若空间列表为空则清理该 key。"""
    ws_file = Path.cwd() / "workspaces.json"
    if not ws_file.exists():
        return
    lock = FileLock(str(ws_file) + ".lock")

    with lock:
        try:
            data = json.loads(ws_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
        except (json.JSONDecodeError, OSError):
            return

        resolved = str(target.resolve())
        space_paths: list[str] = data.get(space_name, [])
        if resolved not in space_paths:
            return
        space_paths.remove(resolved)
        if not space_paths:
            data.pop(space_name, None)
        ws_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reset_config_cache()


def apply_http_proxy_env(http_proxy: str | None) -> None:
    """将 .env 中的 HTTP_PROXY 同步到进程环境，供 httpx/urllib 等客户端走本地代理隧道。"""
    if not http_proxy or not str(http_proxy).strip():
        return
    p = str(http_proxy).strip()
    os.environ["HTTP_PROXY"] = p
    os.environ["HTTPS_PROXY"] = p
    os.environ["http_proxy"] = p
    os.environ["https_proxy"] = p


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        os.environ.setdefault("GOOGLE_API_KEY", gemini_key)

    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

    http_proxy = os.getenv("HTTP_PROXY", "").strip() or None
    apply_http_proxy_env(http_proxy)

    proxy = os.getenv("PROXY_URL", "").strip().rstrip("/")
    workspaces_raw = os.getenv("WORKSPACE_PATHS", "./workspace")
    workspaces = [Path(p.strip()).resolve() for p in workspaces_raw.split(",") if p.strip()]

    # 加载持久化的工作空间路径
    persistent_workspaces = _load_persistent_workspaces()
    workspaces.extend(persistent_workspaces)

    # 去重（以 resolved 绝对路径为准）
    workspaces = list({p.resolve() for p in workspaces})

    dynamic_dir = Path("./dynamic_workspace").resolve()
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    if dynamic_dir not in workspaces:
        workspaces.append(dynamic_dir)

    api_key = gemini_key or os.getenv("GOOGLE_API_KEY", "").strip()

    return AppConfig(
        api_key=api_key,
        dashscope_api_key=dashscope_key,
        proxy_base=proxy,
        http_proxy=http_proxy,
        workspaces=workspaces,
        chroma_dir=os.getenv("CHROMA_PERSIST_DIR", "./.chroma_second_brain"),
        llm_model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
        local_embed_model=os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-large-zh-v1.5"),
        local_rerank_model=os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
        top_k=int(os.getenv("TOP_K", "5")),
        debounce=float(os.getenv("DEBOUNCE_SECONDS", "60")),
        memory_token_limit=int(os.getenv("MEMORY_TOKEN_LIMIT", "8192")),
    )


def reset_config_cache() -> None:
    """测试或热重载 .env 时可调用。"""
    get_config.cache_clear()
