"""
工作区文件监听：Watchdog + 可配置秒级防抖（默认 60s），触发增量索引。
支持多空间隔离 — 每个空间的 handler 更新对应 Collection 的状态文件。
忽略 node_modules / target / .git / .idea 下的路径，避免无意义刷新。
"""

from __future__ import annotations

import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import AppConfig
from rag_engine import incremental_upsert_file, path_should_skip

from llama_index.core import VectorStoreIndex


class DebouncedIndexHandler(FileSystemEventHandler):
    """在静默 debounce 秒后，批量刷新待处理文件至指定空间的向量库。"""

    def __init__(self, index: VectorStoreIndex, cfg: AppConfig,
                 collection_name: str = "default_space"):
        self._index = index
        self._cfg = cfg
        self._collection_name = collection_name
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._pending: set[Path] = set()

    def _flush(self) -> None:
        with self._lock:
            paths = list(self._pending)
            self._pending.clear()
            self._timer = None
        for p in paths:
            incremental_upsert_file(self._index, self._cfg, p,
                                    collection_name=self._collection_name)

    def _schedule(self, path: Path) -> None:
        if path_should_skip(path):
            return
        with self._lock:
            self._pending.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._cfg.debounce, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def on_created(self, event):
        if event.is_directory:
            return
        self._schedule(Path(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            return
        self._schedule(Path(event.src_path))

    def on_moved(self, event):
        if event.is_directory:
            return
        self._schedule(Path(event.dest_path))

    def on_deleted(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if path_should_skip(p):
            return
        doc_id = str(p.resolve())
        try:
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
        except Exception:
            pass


def start_observer(handler: DebouncedIndexHandler, workspaces: list[Path]) -> Observer:
    obs = Observer()
    for root in workspaces:
        if root.is_dir():
            obs.schedule(handler, str(root), recursive=True)
    obs.start()
    return obs
