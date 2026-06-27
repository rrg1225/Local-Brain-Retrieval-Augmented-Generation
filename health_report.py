from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
    "app.py",
    "config.py",
    "rag_engine.py",
    "watcher.py",
    "doctor.py",
]


def main() -> None:
    problems: list[str] = []
    for rel_path in REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            problems.append(f"missing {rel_path}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for section in ("Highlights", "RAG Pipeline", "Operational Readiness", "Privacy Notes"):
        if section not in readme:
            problems.append(f"README missing section: {section}")

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", ".chroma_second_brain/", "dynamic_workspace/", ".chat_history_*.json"):
        if pattern not in ignored:
            problems.append(f".gitignore missing pattern: {pattern}")

    if problems:
        for problem in problems:
            print(f"[health] {problem}")
        raise SystemExit(1)

    print("[health] local-brain-rag repository checks passed")


if __name__ == "__main__":
    main()
