#!/usr/bin/env python3
"""Recompute self-affirmation flags for refactoring commits using the project regex."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phase3_refactoring_analysis.self_affirmation import (  # noqa: E402
    SELF_AFFIRMATION_PATTERN,
)

DEFAULT_COMMITS_WITH_REFACTORING = Path(
    "data/analysis/refactoring_instances/commits_with_refactoring.parquet"
)
DEFAULT_REFACTORING_COMMITS_PARQUET = Path(
    "data/analysis/refactoring_instances/refactoring_commits.parquet"
)
DEFAULT_REFACTORING_COMMITS_CSV = Path(
    "data/analysis/refactoring_instances/refactoring_commits.csv"
)

def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute is_self_affirmed flags using the canonical regex."
    )
    parser.add_argument(
        "--commits-with-refactoring",
        type=Path,
        default=DEFAULT_COMMITS_WITH_REFACTORING,
        help=f"Path to commits_with_refactoring.parquet (default: {DEFAULT_COMMITS_WITH_REFACTORING})",
    )
    parser.add_argument(
        "--refactoring-commits",
        type=Path,
        default=None,
        help=(
            "Input refactoring_commits file (parquet or CSV). "
            "Defaults to the parquet file if present, otherwise the CSV."
        ),
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_REFACTORING_COMMITS_PARQUET,
        help=f"Output parquet path (default: {DEFAULT_REFACTORING_COMMITS_PARQUET})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_REFACTORING_COMMITS_CSV,
        help=f"Output CSV path (default: {DEFAULT_REFACTORING_COMMITS_CSV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Recompute flags and print summary without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    commits_path = args.commits_with_refactoring
    refactoring_input: Optional[Path] = args.refactoring_commits

    if refactoring_input is None:
        if DEFAULT_REFACTORING_COMMITS_PARQUET.exists():
            refactoring_input = DEFAULT_REFACTORING_COMMITS_PARQUET
        elif DEFAULT_REFACTORING_COMMITS_CSV.exists():
            refactoring_input = DEFAULT_REFACTORING_COMMITS_CSV
        else:
            raise FileNotFoundError(
                "Could not locate refactoring_commits.parquet or refactoring_commits.csv. "
                "Specify --refactoring-commits manually."
            )

    commits_df = _load_dataframe(commits_path).copy()
    if "sha" not in commits_df.columns or "message" not in commits_df.columns:
        raise ValueError("commits_with_refactoring file must include 'sha' and 'message' columns.")

    commit_columns = list(commits_df.columns)
    original_commit_dtype = commits_df["is_self_affirmed"].dtype if "is_self_affirmed" in commits_df.columns else bool

    commits_df["message"] = commits_df["message"].astype(str)
    commits_df["is_self_affirmed"] = commits_df["message"].apply(
        lambda text: bool(SELF_AFFIRMATION_PATTERN.search(text))
    )
    if commits_df["is_self_affirmed"].dtype != original_commit_dtype:
        commits_df["is_self_affirmed"] = commits_df["is_self_affirmed"].astype(original_commit_dtype)
    commits_df = commits_df[commit_columns]
    commits_df.to_parquet(commits_path, index=False)

    new_flags = commits_df.set_index("sha")["is_self_affirmed"]

    ref_df = _load_dataframe(refactoring_input).copy()
    if "sha" not in ref_df.columns or "is_self_affirmed" not in ref_df.columns:
        raise ValueError("Input refactoring_commits file must include 'sha' and 'is_self_affirmed' columns.")

    ref_columns = list(ref_df.columns)
    original_dtype = ref_df["is_self_affirmed"].dtype

    old_flags = ref_df["is_self_affirmed"].copy()
    before_counts = old_flags.value_counts(dropna=False).to_dict()

    ref_df["is_self_affirmed"] = ref_df["sha"].map(new_flags).fillna(False)
    if ref_df["is_self_affirmed"].dtype != original_dtype:
        ref_df["is_self_affirmed"] = ref_df["is_self_affirmed"].astype(original_dtype)

    after_counts = ref_df["is_self_affirmed"].value_counts(dropna=False).to_dict()
    changed = (ref_df["is_self_affirmed"] != old_flags).sum()
    ref_df = ref_df[ref_columns]

    print("Self-affirmation recomputed:")
    print(f"  Total refactoring commits: {len(ref_df)}")
    print(f"  Matches in commits_with_refactoring: {new_flags.shape[0]}")
    print(f"  is_self_affirmed counts before: {before_counts}")
    print(f"  is_self_affirmed counts after:  {after_counts}")
    print(f"  Updated rows: {changed}")

    if args.dry_run:
        print("Dry run requested; no files written.")
        return 0

    parquet_out = args.output_parquet
    csv_out = args.output_csv
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_parquet(parquet_out, index=False)
    ref_df.to_csv(csv_out, index=False)

    print(f"Wrote updated parquet to {parquet_out}")
    print(f"Wrote updated CSV to {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
