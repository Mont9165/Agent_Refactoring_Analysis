#!/usr/bin/env python3
"""Compute Designite before/after metric deltas for refactoring entities."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.research_questions.designite_entity_delta import (  # noqa: E402
    DesigniteDeltaCalculator,
    aggregate_deltas,
    load_tool_config,
)

COMMITS_PATH = Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet")
REFMINER_PATH = Path("data/analysis/refactoring_instances/refminer_refactorings.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Optional limit on number of refactoring commits to process (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not COMMITS_PATH.exists():
        raise FileNotFoundError(f"Missing commits dataset: {COMMITS_PATH}")
    if not REFMINER_PATH.exists():
        raise FileNotFoundError(f"Missing RefactoringMiner dataset: {REFMINER_PATH}")

    commits_df = pd.read_parquet(COMMITS_PATH)
    refminer_df = pd.read_parquet(REFMINER_PATH)

    cfg = load_tool_config()
    calculator = DesigniteDeltaCalculator(cfg, max_commits=args.max_commits)
    type_df, method_df = calculator.process(commits_df, refminer_df)

    frames = [df for df in (type_df, method_df) if not df.empty]
    if not frames:
        print("No deltas computed. Ensure Designite outputs are available and REPOS_BASE points to cloned repositories.")
        return

    combined = pd.concat(frames, ignore_index=True)

    summary = aggregate_deltas(combined)
    print("Computed Designite deltas for", len(combined), "entity metrics")
    if not summary.empty:
        print(summary.head())


if __name__ == "__main__":
    main()
