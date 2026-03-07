#!/usr/bin/env python3
"""Validate agent/repo docs for broken local links and secret leakage patterns."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [
    REPO_ROOT / "AGENT_START_HERE.md",
    REPO_ROOT / "AGENT_RETRIEVAL_MAP.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / "AGENT_SYMBOL_INDEX.md",
]

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

SECRET_PATTERNS = [
    re.compile(r"xai-[A-Za-z0-9]{20,}"),
    re.compile(r"https://discord\.com/api/webhooks/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+"),
    re.compile(r"\bPK[A-Z0-9]{10,}\b"),
]


def check_links(path: Path, content: str) -> List[str]:
    errors: List[str] = []
    for match in LINK_RE.finditer(content):
        target = match.group(1).strip()
        if target.startswith("http://") or target.startswith("https://") or target.startswith("mailto:"):
            continue
        local = target.split("#", 1)[0]
        if not local:
            continue
        if local.startswith("/"):
            errors.append(f"{path.name}: absolute local link not allowed: {target}")
            continue
        resolved = (path.parent / local).resolve()
        if not resolved.exists():
            errors.append(f"{path.name}: broken link target: {target}")
    return errors


def check_secrets(path: Path, content: str) -> List[str]:
    errs: List[str] = []
    for pat in SECRET_PATTERNS:
        if pat.search(content):
            errs.append(f"{path.name}: potential secret pattern matched: {pat.pattern}")
    return errs


def validate_required_policy_text(agent_map: Path) -> List[str]:
    errs: List[str] = []
    content = agent_map.read_text(encoding="utf-8")
    required_snippets = [
        "Include first (high confidence)",
        "Include second (documentation)",
        "Include with caution",
        "Exclude by default",
        "_*.py",
        "_*.txt",
        "Source-of-truth config rule",
    ]
    for snippet in required_snippets:
        if snippet not in content:
            errs.append(f"{agent_map.name}: missing required policy snippet: {snippet}")
    return errs


def main() -> int:
    errors: List[str] = []

    for doc in DOC_FILES:
        if not doc.exists():
            errors.append(f"missing required doc: {doc.relative_to(REPO_ROOT)}")
            continue
        text = doc.read_text(encoding="utf-8")
        errors.extend(check_links(doc, text))
        errors.extend(check_secrets(doc, text))

    agent_map = REPO_ROOT / "AGENT_RETRIEVAL_MAP.md"
    if agent_map.exists():
        errors.extend(validate_required_policy_text(agent_map))

    if errors:
        print("FAIL: doc validation errors")
        for err in errors:
            print(f" - {err}")
        return 1

    print("PASS: documentation references and policy checks are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
