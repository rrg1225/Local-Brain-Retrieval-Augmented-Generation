from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROBLEMS: list[str] = []

for rel_path in (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/ENTERPRISE_READINESS.md",
    "docs/OPERATIONS.md",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    "requirements.txt",
):
    if not (ROOT / rel_path).exists():
        PROBLEMS.append(f"missing {rel_path}")

ci_file = ROOT / ".github" / "workflows" / "ci.yml"
if ci_file.exists() and "python ops_check.py" not in ci_file.read_text(encoding="utf-8"):
    PROBLEMS.append("CI does not run python ops_check.py")

readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
if "Enterprise Readiness" not in readme:
    PROBLEMS.append("README missing Enterprise Readiness")

if PROBLEMS:
    for problem in PROBLEMS:
        print(f"[ops] {problem}")
    raise SystemExit(1)

print("[ops] Local Brain RAG operational readiness checks passed")
