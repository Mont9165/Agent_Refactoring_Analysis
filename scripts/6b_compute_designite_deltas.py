#!/usr/bin/env python3
"""Compute Designite before/after metric deltas for refactoring entities."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.phase3_code_quality.designite_entity_delta import (  # noqa: E402
    DESIGNITE_OUTPUT_ROOT,
    DELTA_Output_DIR,
    DesigniteDeltaCalculator,
    aggregate_deltas,
    load_tool_config,
)

COMMITS_PATH = Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet")
REFMINER_PATH = Path("data/analysis/refactoring_instances/refminer_refactorings.parquet")
TYPE_DELTA_PATH = DELTA_Output_DIR / "type_metric_deltas.parquet"
METHOD_DELTA_PATH = DELTA_Output_DIR / "method_metric_deltas.parquet"


def _parse_owner_repo(url: str) -> Optional[tuple[str, str]]:
    if not isinstance(url, str):
        return None
    parts = url.split("/")
    if len(parts) < 5 or "github.com" not in parts[2]:
        return None
    return parts[3], parts[4]


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

    existing_type = pd.read_parquet(TYPE_DELTA_PATH) if TYPE_DELTA_PATH.exists() else pd.DataFrame()
    existing_method = pd.read_parquet(METHOD_DELTA_PATH) if METHOD_DELTA_PATH.exists() else pd.DataFrame()

    processed_shas: set[str] = set()
    if not existing_type.empty:
        processed_shas.update(existing_type["commit_sha"].astype(str))
    if not existing_method.empty:
        processed_shas.update(existing_method["commit_sha"].astype(str))

    if processed_shas:
        commits_df = commits_df[~commits_df["sha"].astype(str).isin(processed_shas)]

    if commits_df.empty:
        print("No new commits to process; existing deltas are up to date.")
        existing_frames = [df for df in (existing_type, existing_method) if not df.empty]
        if existing_frames:
            aggregate_deltas(pd.concat(existing_frames, ignore_index=True))
        return

    if args.max_commits is not None:
        commits_df = commits_df.head(args.max_commits)

    cfg = load_tool_config()
    calculator = DesigniteDeltaCalculator(cfg, max_commits=None)
    type_df, method_df = calculator.process(commits_df, refminer_df)

    final_frames = []
    if not type_df.empty:
        combined_type = pd.concat([existing_type, type_df], ignore_index=True)
        combined_type = combined_type.drop_duplicates(
            subset=[
                "commit_sha",
                "child_sha",
                "parent_sha",
                "entity_kind",
                "metric",
                "before_key",
                "after_key",
            ],
            keep="last",
        )
        combined_type.to_parquet(TYPE_DELTA_PATH, index=False)
        combined_type.to_csv(TYPE_DELTA_PATH.with_suffix(".csv"), index=False)
        final_frames.append(combined_type)
    elif not existing_type.empty:
        final_frames.append(existing_type)

    if not method_df.empty:
        combined_method = pd.concat([existing_method, method_df], ignore_index=True)
        combined_method = combined_method.drop_duplicates(
            subset=[
                "commit_sha",
                "child_sha",
                "parent_sha",
                "entity_kind",
                "metric",
                "before_key",
                "after_key",
            ],
            keep="last",
        )
        combined_method.to_parquet(METHOD_DELTA_PATH, index=False)
        combined_method.to_csv(METHOD_DELTA_PATH.with_suffix(".csv"), index=False)
        final_frames.append(combined_method)
    elif not existing_method.empty:
        final_frames.append(existing_method)

    if not final_frames:
        print("No Designite deltas produced. Verify Designite outputs exist for the target commits.")
        return

    combined = pd.concat(final_frames, ignore_index=True)
    summary = aggregate_deltas(combined)
    print("Computed Designite deltas for", len(combined), "entity metrics")
    if not summary.empty:
        print(summary.head())


if __name__ == "__main__":
    main()
