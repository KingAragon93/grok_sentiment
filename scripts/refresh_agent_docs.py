#!/usr/bin/env python3
"""Refresh agent docs: regenerate symbol index and validate references."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(args: list[str]) -> int:
    print(f"RUN: {' '.join(args)}")
    proc = subprocess.run(args, cwd=REPO_ROOT)
    return proc.returncode


def main() -> int:
    steps = [
        [sys.executable, "scripts/generate_agent_symbol_index.py"],
        [sys.executable, "scripts/validate_doc_references.py"],
    ]

    for step in steps:
        code = run_step(step)
        if code != 0:
            print(f"FAIL: {' '.join(step)}")
            return code

    print("PASS: agent docs refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
