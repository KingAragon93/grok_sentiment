#!/usr/bin/env python3
"""Generate AGENT_SYMBOL_INDEX.md from top-level Python symbols (AST-based)."""

from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "AGENT_SYMBOL_INDEX.md"
MAX_SYMBOLS_PER_FILE = 20

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}


def classify_file(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if rel.startswith("scripts/"):
        return "tooling"
    if rel in {"main.py", "gcs_logger.py"}:
        return "primary-runtime"
    if rel in {"analyze_recommendations.py", "sentiment_analyzer.py"}:
        return "analysis"
    if rel.endswith("_example.py"):
        return "reference"
    return "secondary"


def iter_python_files() -> List[Path]:
    files: List[Path] = []
    for p in REPO_ROOT.rglob("*.py"):
        rel_parts = p.relative_to(REPO_ROOT).parts
        if any(part in EXCLUDE_DIR_NAMES for part in rel_parts):
            continue
        if p.name.startswith("_"):
            # scratch files excluded from index
            continue
        files.append(p)
    return sorted(files, key=lambda x: x.relative_to(REPO_ROOT).as_posix())


def top_level_symbols(path: Path) -> List[Tuple[str, str, int]]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    symbols: List[Tuple[str, str, int]] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(("class", node.name, node.lineno))
        elif isinstance(node, ast.FunctionDef):
            symbols.append(("def", node.name, node.lineno))
        elif isinstance(node, ast.AsyncFunctionDef):
            symbols.append(("async def", node.name, node.lineno))

    return symbols


def build_markdown(index: Dict[str, Dict[str, List[Tuple[str, str, int]]]]) -> str:
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines.append("# AGENT_SYMBOL_INDEX")
    lines.append("")
    lines.append("Auto-generated top-level symbol index for retrieval-efficient code navigation.")
    lines.append("")
    lines.append(f"Generated (UTC): {ts}")
    lines.append(f"Symbol cap per file: {MAX_SYMBOLS_PER_FILE}")
    lines.append("")

    tier_order = ["primary-runtime", "secondary", "analysis", "reference", "tooling"]

    for tier in tier_order:
        files = index.get(tier, {})
        if not files:
            continue
        lines.append(f"## {tier}")
        lines.append("")
        for rel in sorted(files.keys()):
            symbols = files[rel]
            lines.append(f"### {rel}")
            if not symbols:
                lines.append("- (no top-level def/class symbols)")
                lines.append("")
                continue

            shown = symbols[:MAX_SYMBOLS_PER_FILE]
            for kind, name, lineno in shown:
                lines.append(f"- `{kind}` `{name}` (L{lineno})")
            if len(symbols) > MAX_SYMBOLS_PER_FILE:
                lines.append(f"- ... {len(symbols) - MAX_SYMBOLS_PER_FILE} more symbols omitted")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    index: Dict[str, Dict[str, List[Tuple[str, str, int]]]] = {}
    for py_file in iter_python_files():
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        tier = classify_file(py_file)
        symbols = top_level_symbols(py_file)
        index.setdefault(tier, {})[rel] = symbols

    OUTPUT_FILE.write_text(build_markdown(index), encoding="utf-8")
    print(f"PASS: wrote {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
