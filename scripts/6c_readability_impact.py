#!/usr/bin/env python3
"""Compute readability impact deltas for refactoring commits using configured tool (e.g., CoRed)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.phase3_code_quality.readability_impact import (  # noqa: E402
    READABILITY_DIR,
    run_readability_impact,
)

READABILITY_PARQUET = READABILITY_DIR / "readability_deltas.parquet"
READABILITY_CSV = READABILITY_DIR / "readability_deltas.csv"
SUMMARY_PARQUET = READABILITY_DIR / "readability_delta_summary.parquet"
SUMMARY_CSV = READABILITY_DIR / "readability_delta_summary.csv"


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    """Return parquet data or an empty frame when the cache is unavailable."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully if the cache is corrupt
        print(f"Warning: failed to read {path}: {exc}. Ignoring cached readability deltas.")
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-commits", type=int, default=None, help="Optional limit on refactoring commits")
    parser.add_argument("--workers", type=int, default=None, help="Repository workers for parallel processing")
    parser.add_argument("--timeout", type=int, default=None, help="Per-invocation timeout in seconds")
    args = parser.parse_args()

    max_commits_env = os.environ.get("READABILITY_MAX_COMMITS")
    max_commits = args.max_commits if args.max_commits is not None else (
        int(max_commits_env) if max_commits_env else None
    )

    existing_df = _safe_read_parquet(READABILITY_PARQUET)
    skip_commits = set(existing_df["commit_sha"].astype(str)) if not existing_df.empty else set()

    workers_env = os.environ.get("READABILITY_WORKERS")
    workers = args.workers if args.workers is not None else (
        int(workers_env) if workers_env else None
    )

    timeout_env = os.environ.get("READABILITY_TIMEOUT")
    timeout = args.timeout if args.timeout is not None else (
        int(timeout_env) if timeout_env else None
    )

    try:
        df = run_readability_impact(
            max_commits=max_commits,
            skip_commits=skip_commits,
            workers=workers,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        sys.exit(1)

    if df.empty:
        if existing_df.empty:
            print("No readability deltas computed. Check READABILITY_TOOL_CMD configuration and repository clones.")
        else:
            print("No new readability deltas; existing outputs already cover all commits.")
        return

    combined_df = pd.concat([existing_df, df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(
        subset=["commit_sha", "refactoring_type", "before_path", "after_path"],
        keep="last",
    )

    combined_df.to_parquet(READABILITY_PARQUET, index=False)
    combined_df.to_csv(READABILITY_CSV, index=False)

    if not combined_df.empty:
        summary_df = (
            combined_df.groupby("refactoring_type")["delta"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )
        summary_df.to_parquet(SUMMARY_PARQUET, index=False)
        summary_df.to_csv(SUMMARY_CSV, index=False)

    new_commits = df["commit_sha"].nunique()
    print(
        f"Computed readability deltas for {len(df)} file instances across {new_commits} new commits."
    )
    print("Outputs: data/analysis/readability/readability_deltas.* and *_summary.*")


if __name__ == "__main__":
    main()
