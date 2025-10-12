#!/usr/bin/env python3
"""Compute Designite before/after metric deltas for refactoring entities."""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

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

logger = logging.getLogger(__name__)
LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    """Return existing deltas if readable, otherwise fall back to an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - log and continue without cached data
        logger.warning("Failed to read %s: %s. Ignoring existing cached deltas.", path, exc)
        return pd.DataFrame()


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
    parser.add_argument(
        "--log-level",
        type=lambda value: value.upper(),
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def _normalize_commits(
    commits_df: pd.DataFrame,
    *,
    fallback_repo: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    result = commits_df.copy()
    placeholder_url: Optional[str] = None
    if fallback_repo:
        placeholder_url = f"https://github.com/{fallback_repo[0]}/{fallback_repo[1]}"

    if "html_url" not in result.columns:
        result["html_url"] = placeholder_url
    elif placeholder_url:
        html_series = result["html_url"].astype("string")
        parts_length = html_series.str.split("/").str.len()
        valid_mask = html_series.str.contains("github.com", na=False) & (parts_length >= 5)
        result.loc[~valid_mask.fillna(False), "html_url"] = placeholder_url

    html_series = result["html_url"].astype("string")
    parts = html_series.str.split("/")
    result["owner"] = parts.str.get(3)
    result["repo"] = parts.str.get(4)

    if fallback_repo:
        result["owner"] = result["owner"].fillna(fallback_repo[0])
        result["repo"] = result["repo"].fillna(fallback_repo[1])

    return result


def _process_commit(
    commit_record: Dict[str, Any],
    refminer_by_sha: Dict[str, pd.DataFrame],
    cfg,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    sha = str(commit_record.get("sha", ""))
    refminer_subset = refminer_by_sha.get(sha)
    if refminer_subset is None or refminer_subset.empty:
        return sha, pd.DataFrame(), pd.DataFrame()

    commit_df = pd.DataFrame([commit_record])
    calculator = DesigniteDeltaCalculator(cfg, max_commits=None, persist_outputs=False)
    type_df, method_df = calculator.process(commit_df, refminer_subset)
    return sha, type_df, method_df


def main() -> None:
    args = parse_args()

    log_level = getattr(logging, args.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger.debug("Parsed arguments: %s", args)

    if not COMMITS_PATH.exists():
        raise FileNotFoundError(f"Missing commits dataset: {COMMITS_PATH}")
    if not REFMINER_PATH.exists():
        raise FileNotFoundError(f"Missing RefactoringMiner dataset: {REFMINER_PATH}")

    commits_df = pd.read_parquet(COMMITS_PATH)
    commits_df = commits_df.drop_duplicates(subset="sha").reset_index(drop=True)
    logger.info("Loaded %d unique commits from %s.", len(commits_df), COMMITS_PATH)
    refminer_df = pd.read_parquet(REFMINER_PATH)
    logger.info("Loaded %d RefactoringMiner rows from %s.", len(refminer_df), REFMINER_PATH)

    existing_type = _safe_read_parquet(TYPE_DELTA_PATH)
    existing_method = _safe_read_parquet(METHOD_DELTA_PATH)

    processed_shas: set[str] = set()
    if not existing_type.empty:
        processed_shas.update(existing_type["commit_sha"].astype(str))
    if not existing_method.empty:
        processed_shas.update(existing_method["commit_sha"].astype(str))

    if processed_shas:
        logger.info("Skipping %d commits already present in cached deltas.", len(processed_shas))
        commits_df = commits_df[~commits_df["sha"].astype(str).isin(processed_shas)]
        commits_df = commits_df.drop_duplicates(subset="sha").reset_index(drop=True)

    if commits_df.empty:
        logger.info("No new commits to process; existing deltas are up to date.")
        existing_frames = [df for df in (existing_type, existing_method) if not df.empty]
        if existing_frames:
            logger.debug("Aggregating existing deltas without new commits.")
            aggregate_deltas(pd.concat(existing_frames, ignore_index=True))
        return

    if args.max_commits is not None:
        commits_df = commits_df.head(args.max_commits)
        logger.info("Limiting to %d commits based on --max-commits.", len(commits_df))

    cfg = load_tool_config()
    fallback_repo: Optional[Tuple[str, str]] = None
    if cfg.local_repo:
        fallback_repo = ("local", cfg.local_repo.name)

    commits_df["sha"] = commits_df["sha"].astype("string")
    refminer_df["commit_sha"] = refminer_df["commit_sha"].astype("string")

    normalized_commits = _normalize_commits(commits_df, fallback_repo=fallback_repo)
    normalized_commits = normalized_commits[normalized_commits["sha"].notna()].reset_index(drop=True)

    missing_repo_mask = normalized_commits["owner"].isna() | normalized_commits["repo"].isna()
    if missing_repo_mask.any():
        logger.warning(
            "Dropping %d commits without resolvable repository information.",
            int(missing_repo_mask.sum()),
        )
        normalized_commits = normalized_commits[~missing_repo_mask].reset_index(drop=True)

    if normalized_commits.empty:
        logger.warning("No commits available after repository normalization.")
        existing_frames = [df for df in (existing_type, existing_method) if not df.empty]
        if existing_frames:
            logger.debug("Aggregating existing deltas after normalization left no commits.")
            aggregate_deltas(pd.concat(existing_frames, ignore_index=True))
        return

    repo_commit_counts = (
        normalized_commits.groupby(["owner", "repo"])["sha"]
        .nunique()
        .reset_index(name="commit_count")
    )
    logger.info(
        "Processing %d commits across %d repositories.",
        len(normalized_commits),
        len(repo_commit_counts),
    )

    refminer_by_sha: Dict[str, pd.DataFrame] = {
        sha: group for sha, group in refminer_df.groupby("commit_sha")
    }
    commits_with_refminer = normalized_commits[normalized_commits["sha"].isin(refminer_by_sha)]
    missing_refminer = len(normalized_commits) - len(commits_with_refminer)
    if missing_refminer:
        logger.info("Skipping %d commits without RefactoringMiner coverage.", missing_refminer)
    commits_with_refminer = commits_with_refminer.reset_index(drop=True)

    if commits_with_refminer.empty:
        logger.warning("No commits have associated RefactoringMiner data; exiting early.")
        existing_frames = [df for df in (existing_type, existing_method) if not df.empty]
        if existing_frames:
            logger.debug("Aggregating existing deltas before exiting due to missing RefactoringMiner data.")
            aggregate_deltas(pd.concat(existing_frames, ignore_index=True))
        return

    workers_env = os.environ.get("DESIGNITE_WORKERS")
    default_workers = int(workers_env) if workers_env else os.cpu_count() or 1
    configured_workers = args.workers or default_workers
    total_commits = len(commits_with_refminer)
    workers = max(1, min(configured_workers, total_commits))

    logger.info(
        "Processing commits with %d worker(s); max queue size %d.",
        workers,
        total_commits,
    )

    commit_records: List[Dict[str, Any]] = commits_with_refminer.to_dict("records")
    type_frames: List[pd.DataFrame] = []
    method_frames: List[pd.DataFrame] = []
    errors: List[Tuple[str, str, str, Exception]] = []

    tqdm_disabled = not sys.stderr.isatty()
    with ThreadPoolExecutor(max_workers=workers) as executor, tqdm(
        total=total_commits,
        desc="Processing commits",
        unit="commit",
        disable=tqdm_disabled,
    ) as progress:
        future_to_commit = {
            executor.submit(_process_commit, record, refminer_by_sha, cfg): record
            for record in commit_records
        }
        for future in as_completed(future_to_commit):
            record = future_to_commit[future]
            owner = str(record.get("owner", "unknown"))
            repo = str(record.get("repo", "unknown"))
            sha = str(record.get("sha", ""))
            try:
                _, commit_type_df, commit_method_df = future.result()
                if not commit_type_df.empty:
                    type_frames.append(commit_type_df)
                if not commit_method_df.empty:
                    method_frames.append(commit_method_df)
                logger.debug(
                    "✓ %s/%s@%s: type_rows=%d, method_rows=%d",
                    owner,
                    repo,
                    sha[:10],
                    len(commit_type_df),
                    len(commit_method_df),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append((owner, repo, sha, exc))
                logger.exception("✗ %s/%s@%s failed", owner, repo, sha[:10])
            finally:
                progress.update(1)

    if errors:
        logger.error("Encountered errors for %d commits.", len(errors))
        for owner, repo, sha, exc in errors:
            logger.error("- %s/%s@%s: %s", owner, repo, sha[:10], exc)

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
        logger.warning(
            "No Designite deltas produced. Verify Designite outputs exist for the target commits."
        )
        return

    combined = pd.concat(final_frames, ignore_index=True)
    summary = aggregate_deltas(combined)
    logger.info("Computed Designite deltas for %d entity metrics.", len(combined))
    if not summary.empty:
        logger.info("Delta summary head:\n%s", summary.head())


if __name__ == "__main__":
    main()
