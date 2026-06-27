"""Lightweight environment doctor for Local Brain RAG.

This script intentionally avoids importing Streamlit, LlamaIndex, torch, or
model packages. It is safe to run before installing the full GPU stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PLACEHOLDERS = {
    "",
    "your_gemini_api_key_here",
    "your_dashscope_api_key_here",
}


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def status(ok: bool, message: str) -> None:
    marker = "OK" if ok else "WARN"
    text = f"[{marker}] {message}"
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def path_status(root: Path, raw_path: str) -> tuple[bool, Path]:
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return path.exists(), path


def main() -> None:
    root = Path(__file__).resolve().parent
    env = {**load_dotenv(root / ".env.example"), **load_dotenv(root / ".env"), **os.environ}

    gemini_key = env.get("GEMINI_API_KEY", "")
    dashscope_key = env.get("DASHSCOPE_API_KEY", "")
    workspace_paths = [p.strip() for p in env.get("WORKSPACE_PATHS", "./workspace").split(",") if p.strip()]
    chroma_dir = Path(env.get("CHROMA_PERSIST_DIR", "./.chroma_second_brain"))
    if not chroma_dir.is_absolute():
        chroma_dir = root / chroma_dir

    gemini_ok = gemini_key not in PLACEHOLDERS
    dashscope_ok = dashscope_key not in PLACEHOLDERS
    status(gemini_ok, "GEMINI_API_KEY ready" if gemini_ok else "GEMINI_API_KEY missing or still a placeholder")
    status(dashscope_ok, "DASHSCOPE_API_KEY ready for Qwen failover" if dashscope_ok else "DASHSCOPE_API_KEY missing or still a placeholder")
    status(len(workspace_paths) > 0, f"{len(workspace_paths)} workspace path(s) configured")

    for raw_path in workspace_paths:
        exists, path = path_status(root, raw_path)
        status(exists, f"workspace exists: {path}")

    status(chroma_dir.parent.exists(), f"Chroma parent exists: {chroma_dir.parent}")
    status((root / "requirements.txt").exists(), "requirements.txt present")
    status(not (root / ".env").exists(), ".env is local-only and should not be committed")
    status(not (root / "workspaces.json").exists(), "workspace state is local-only and should not be committed")
    print("\nNext: pip install -r requirements.txt && streamlit run main.py")


if __name__ == "__main__":
    main()
