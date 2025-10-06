#!/usr/bin/env python3
"""Compute Designite before/after metric deltas for refactoring entities."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.phase3_code_quality.designite_entity_delta import (  # noqa: E402
    DELTA_Output_DIR,
    DesigniteDeltaCalculator,
    aggregate_deltas,
    load_tool_config,
)

COMMITS_PATH = Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet")
REFMINER_PATH = Path("data/analysis/refactoring_instances/refminer_refactorings.parquet")
TYPE_DELTA_PATH = DELTA_Output_DIR / "type_metric_deltas.parquet"
METHOD_DELTA_PATH = DELTA_Output_DIR / "method_metric_deltas.parquet"


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    """Return existing deltas if readable, otherwise fall back to an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - log and continue without cached data
        print(f"Warning: failed to read {path}: {exc}. Ignoring existing cached deltas.")
        return pd.DataFrame()


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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of repositories to process in parallel (defaults to DESIGNITE_WORKERS or 1).",
    )
    return parser.parse_args()


def _group_commits_by_repo(commits_df: pd.DataFrame) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for _, row in commits_df.iterrows():
        owner_repo = _parse_owner_repo(row.get("html_url"))
        if not owner_repo:
            continue
        groups.setdefault(owner_repo, []).append(row.to_dict())
    return groups


def _process_repo_commits(
    owner: str,
    repo: str,
    rows: List[dict],
    refminer_df: pd.DataFrame,
    cfg,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = pd.DataFrame(rows)
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()
    subset_refminer = refminer_df[refminer_df["commit_sha"].isin(subset["sha"])]
    calculator = DesigniteDeltaCalculator(cfg, max_commits=None, persist_outputs=False)
    return calculator.process(subset, subset_refminer)


def main() -> None:
    args = parse_args()

    if not COMMITS_PATH.exists():
        raise FileNotFoundError(f"Missing commits dataset: {COMMITS_PATH}")
    if not REFMINER_PATH.exists():
        raise FileNotFoundError(f"Missing RefactoringMiner dataset: {REFMINER_PATH}")

    commits_df = pd.read_parquet(COMMITS_PATH)
    refminer_df = pd.read_parquet(REFMINER_PATH)

    existing_type = _safe_read_parquet(TYPE_DELTA_PATH)
    existing_method = _safe_read_parquet(METHOD_DELTA_PATH)

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

    repo_groups = _group_commits_by_repo(commits_df)
    if not repo_groups:
        print("No commits with GitHub repository information to process.")
        existing_frames = [df for df in (existing_type, existing_method) if not df.empty]
        if existing_frames:
            aggregate_deltas(pd.concat(existing_frames, ignore_index=True))
        return

    cfg = load_tool_config()
    workers_env = os.environ.get("DESIGNITE_WORKERS")
    workers = args.workers or (int(workers_env) if workers_env else 1)
    workers = max(1, min(workers, len(repo_groups)))

    print(f"Processing {len(repo_groups)} repositories with {workers} worker(s)...")

    type_frames: List[pd.DataFrame] = []
    method_frames: List[pd.DataFrame] = []
    errors: List[Tuple[str, str, Exception]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_repo = {
            executor.submit(_process_repo_commits, owner, repo, rows, refminer_df, cfg): (owner, repo)
            for (owner, repo), rows in repo_groups.items()
        }
        for future in as_completed(future_to_repo):
            owner, repo = future_to_repo[future]
            try:
                repo_type_df, repo_method_df = future.result()
                if not repo_type_df.empty:
                    type_frames.append(repo_type_df)
                if not repo_method_df.empty:
                    method_frames.append(repo_method_df)
                print(
                    f"  ✓ {owner}/{repo}: type_rows={len(repo_type_df)}, method_rows={len(repo_method_df)}"
                )
            except Exception as exc:  # noqa: BLE001
                errors.append((owner, repo, exc))
                print(f"  ✗ {owner}/{repo}: {exc}")

    if errors:
        print("Encountered errors for the following repositories:")
        for owner, repo, exc in errors:
            print(f"  - {owner}/{repo}: {exc}")

    type_df = pd.concat(type_frames, ignore_index=True) if type_frames else pd.DataFrame()
    method_df = pd.concat(method_frames, ignore_index=True) if method_frames else pd.DataFrame()

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
