#!/usr/bin/env python3
"""Compute readability impact deltas for refactoring commits using configured tool (e.g., CoRed)."""
from __future__ import annotations

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


def main() -> None:
    max_commits_env = os.environ.get("READABILITY_MAX_COMMITS")
    max_commits = int(max_commits_env) if max_commits_env else None

    existing_df = pd.read_parquet(READABILITY_PARQUET) if READABILITY_PARQUET.exists() else pd.DataFrame()
    skip_commits = set(existing_df["commit_sha"].astype(str)) if not existing_df.empty else set()

    workers_env = os.environ.get("READABILITY_WORKERS")
    workers = int(workers_env) if workers_env else None

    try:
        df = run_readability_impact(max_commits=max_commits, skip_commits=skip_commits, workers=workers)
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
