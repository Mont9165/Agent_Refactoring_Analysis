#!/usr/bin/env python3
"""Compute readability impact deltas for refactoring commits using configured tool (e.g., CoRed)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.phase3_code_quality.readability_impact import run_readability_impact


def main() -> None:
    max_commits_env = os.environ.get("READABILITY_MAX_COMMITS")
    max_commits = int(max_commits_env) if max_commits_env else None

    try:
        df = run_readability_impact(max_commits=max_commits)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        sys.exit(1)

    if df.empty:
        print("No readability deltas computed. Check READABILITY_TOOL_CMD configuration and repository clones.")
    else:
        print(f"Computed readability deltas for {len(df)} file instances across {df['commit_sha'].nunique()} commits.")
        print("Outputs: data/analysis/readability/readability_deltas.* and *_summary.*")


if __name__ == "__main__":
    main()
