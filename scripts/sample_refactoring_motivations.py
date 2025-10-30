#!/usr/bin/env python3
"""Randomly sample refactoring motivation commits per category."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

DEFAULT_INPUT = Path("data/analysis/refactoring_instances/gpt_refactoring_motivation_update.csv")
DEFAULT_OUTPUT = Path("data/analysis/refactoring_instances/gpt_refactoring_motivation_sample.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the motivations CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--existing-sample",
        type=Path,
        default=None,
        help="Optional CSV of existing samples to retain and top up.",
    )
    parser.add_argument(
        "--id-column",
        default="sha",
        help="Unique identifier column used to avoid duplicate sampling (default: sha).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination CSV for the sampled rows (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--category-column",
        default="type",
        help="Column used to stratify samples (default: type).",
    )
    parser.add_argument(
        "--min-per-category",
        type=int,
        default=10,
        help="Minimum number of commits to sample for each category (default: 10).",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=10,
        help="Maximum number of commits to sample for each category (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


def _choose_sample_size(
    rng: random.Random,
    available: int,
    min_per_category: int,
    max_per_category: int,
) -> Tuple[int, bool]:
    if available <= 0:
        return 0, False

    if available < min_per_category:
        return available, True

    upper_bound = min(max_per_category, available)
    if upper_bound <= min_per_category:
        return upper_bound, False

    return rng.randint(min_per_category, upper_bound), False


def sample_per_category(
    df: pd.DataFrame,
    category_column: str,
    min_per_category: int,
    max_per_category: int,
    seed: int,
    existing_records: Dict[str, Set[str]] | None = None,
    *,
    id_column: str = "sha",
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, int]], set[str]]:
    rng = random.Random(seed)
    sampled_frames: List[pd.DataFrame] = []
    warnings: List[str] = []
    summary: Dict[str, Dict[str, int]] = {}
    undersized_categories: set[str] = set()
    existing_records = existing_records or {}

    for category, group in sorted(df.groupby(category_column), key=lambda item: str(item[0])):
        category_key = str(category)
        group = group.copy()
        group_ids = group[id_column].astype(str)
        already_selected: Set[str] = {str(value) for value in existing_records.get(category_key, set())}
        available_pool = group.loc[~group_ids.isin(already_selected)]
        available_total = len(available_pool) + len(already_selected)
        size, undersized_flag = _choose_sample_size(rng, available_total, min_per_category, max_per_category)
        summary[category_key] = {
            "available": int(available_total),
            "requested": int(size),
            "existing": int(len(already_selected)),
            "added": 0,
        }

        if available_total == 0:
            warnings.append(f"Category '{category}' has no rows to sample.")
            continue

        if undersized_flag:
            warnings.append(
                f"Category '{category}' has only {available_total} rows (including existing samples), "
                f"less than requested minimum {min_per_category}."
            )
            undersized_categories.add(category_key)

        if size <= len(already_selected):
            # Existing samples already satisfy the requested count.
            continue

        needed = size - len(already_selected)
        if needed <= 0:
            continue

        sample_state = rng.randint(0, 2**32 - 1)
        sampled = available_pool.sample(n=needed, random_state=sample_state).copy()
        summary[category_key]["added"] = int(len(sampled))
        sampled_frames.append(sampled)

    if not sampled_frames:
        # Ensure categories present only in existing_records are tracked in summary.
        for category_key, ids in existing_records.items():
            if category_key not in summary:
                size_existing = int(len(ids))
                summary[category_key] = {
                    "available": size_existing,
                    "requested": size_existing,
                    "existing": size_existing,
                    "added": 0,
                }
                if size_existing < min_per_category:
                    undersized_categories.add(category_key)
                    warnings.append(
                        f"Category '{category_key}' has only {size_existing} existing rows, "
                        f"less than requested minimum {min_per_category}."
                    )
        return pd.DataFrame(), warnings, summary, undersized_categories

    combined = pd.concat(sampled_frames, ignore_index=True)

    for category_key, ids in existing_records.items():
        if category_key not in summary:
            size_existing = int(len(ids))
            summary[category_key] = {
                "available": size_existing,
                "requested": size_existing,
                "existing": size_existing,
                "added": 0,
            }
            if size_existing < min_per_category:
                undersized_categories.add(category_key)
                warnings.append(
                    f"Category '{category_key}' has only {size_existing} existing rows, "
                    f"less than requested minimum {min_per_category}."
                )

    return combined, warnings, summary, undersized_categories


def main() -> None:
    args = _parse_args()
    if args.min_per_category <= 0 or args.max_per_category <= 0:
        raise ValueError("Sample sizes must be positive integers.")
    if args.max_per_category < args.min_per_category:
        raise ValueError("--max-per-category cannot be smaller than --min-per-category.")

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.category_column not in df.columns:
        raise KeyError(f"Column '{args.category_column}' not found in {args.input}")
    if args.id_column not in df.columns:
        raise KeyError(f"Column '{args.id_column}' not found in {args.input}")

    commits_meta_path = Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet")
    if not commits_meta_path.exists():
        raise FileNotFoundError(
            f"Commits metadata parquet not found at {commits_meta_path}; cannot determine SAR commits."
        )
    commits_meta = pd.read_parquet(commits_meta_path, columns=["sha", "is_self_affirmed"])
    commits_meta = commits_meta.dropna(subset=["sha"])
    commits_meta["sha"] = commits_meta["sha"].astype(str)
    sar_mapping = commits_meta.set_index("sha")["is_self_affirmed"]

    df[args.id_column] = df[args.id_column].astype(str)
    if "is_self_affirmed" not in df.columns:
        df["is_self_affirmed"] = df[args.id_column].map(sar_mapping)
    missing_affirmed = df["is_self_affirmed"].isna().sum()
    if missing_affirmed:
        print(
            f"Warning: {missing_affirmed} rows in {args.input} lacked SAR metadata; they will be dropped."
        )
    df = df[df["is_self_affirmed"] == True].copy()
    if df.empty:
        raise RuntimeError("Input data contains no self-affirmed (SAR) commits after filtering.")

    df = df.copy()
    df[args.id_column] = df[args.id_column].astype(str)

    existing_records: Dict[str, Set[str]] = {}
    existing_df: pd.DataFrame | None = None
    extra_warnings: List[str] = []

    if args.existing_sample:
        if not args.existing_sample.exists():
            extra_warnings.append(f"Existing sample file not found at {args.existing_sample}; ignoring.")
        else:
            existing_df = pd.read_csv(args.existing_sample)
            if args.category_column not in existing_df.columns:
                raise KeyError(f"Column '{args.category_column}' not found in {args.existing_sample}")
            if args.id_column not in existing_df.columns:
                raise KeyError(f"Column '{args.id_column}' not found in {args.existing_sample}")
            existing_df = existing_df.dropna(subset=[args.id_column]).copy()
            before_count = len(existing_df)
            if "is_self_affirmed" in existing_df.columns:
                existing_df = existing_df[existing_df["is_self_affirmed"] == True].copy()
                dropped = before_count - len(existing_df)
                if dropped > 0:
                    extra_warnings.append(
                        f"Dropped {dropped} non-SAR rows from existing sample {args.existing_sample}."
                    )
            else:
                existing_df[args.id_column] = existing_df[args.id_column].astype(str)
                existing_df["is_self_affirmed"] = existing_df[args.id_column].map(sar_mapping)
                missing_affirmed_existing = existing_df["is_self_affirmed"].isna().sum()
                if missing_affirmed_existing:
                    extra_warnings.append(
                        f"{missing_affirmed_existing} rows in {args.existing_sample} lacked SAR metadata and were dropped."
                    )
                existing_df = existing_df[existing_df["is_self_affirmed"] == True].copy()
            if existing_df.empty:
                existing_df = None
            else:
                existing_df[args.id_column] = existing_df[args.id_column].astype(str)
                existing_df[args.category_column] = existing_df[args.category_column].astype(str)
                for category, group in existing_df.groupby(args.category_column):
                    existing_records[str(category)] = set(group[args.id_column])
                input_ids = set(df[args.id_column])
                existing_ids = {identifier for ids in existing_records.values() for identifier in ids}
                missing_ids = existing_ids - input_ids
                if missing_ids:
                    extra_warnings.append(
                        f"{len(missing_ids)} existing sample rows are not present in the input dataset; "
                        "they will be retained but cannot be refreshed."
                    )

    sampled_df, warnings, summary, undersized_categories = sample_per_category(
        df,
        category_column=args.category_column,
        min_per_category=args.min_per_category,
        max_per_category=args.max_per_category,
        seed=args.seed,
        existing_records=existing_records,
        id_column=args.id_column,
    )
    warnings = extra_warnings + warnings

    if existing_df is not None:
        combined_df = pd.concat([existing_df, sampled_df], ignore_index=True, sort=False)
        combined_df = combined_df.drop_duplicates(subset=[args.id_column], keep="first").reset_index(drop=True)
    else:
        combined_df = sampled_df

    commits_path = Path("data/analysis/refactoring_instances/commits_with_refactoring.parquet")
    commit_urls: Dict[str, str] = {}
    pr_urls: Dict[str, str] = {}
    if commits_path.exists():
        try:
            commits_df = pd.read_parquet(commits_path, columns=["sha", "html_url"])
            commits_df = commits_df.dropna(subset=["html_url"]).copy()
            commits_df["html_url"] = commits_df["html_url"].astype(str)
            commits_df = commits_df.drop_duplicates(subset="sha", keep="first")

            def _extract_owner_repo(url: str) -> Tuple[str, str]:
                parts = url.split("/")
                if len(parts) < 5:
                    raise ValueError
                return parts[3], parts[4]

            commit_urls = {}
            pr_urls = {}
            for sha_value, url in zip(commits_df["sha"], commits_df["html_url"]):
                try:
                    owner, repo = _extract_owner_repo(url)
                except ValueError:
                    continue
                commit_urls[str(sha_value)] = f"https://github.com/{owner}/{repo}/commit/{sha_value}"
                pr_urls[str(sha_value)] = url
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Warning: could not read commit metadata from {commits_path}: {exc}")

    if args.category_column in combined_df.columns:
        combined_df[args.category_column] = combined_df[args.category_column].astype(str)
    combined_df[args.id_column] = combined_df[args.id_column].astype(str)

    if commit_urls:
        mapped_commit = combined_df[args.id_column].map(commit_urls)
        if "commit_url" in combined_df.columns:
            combined_df["commit_url"] = combined_df["commit_url"].fillna(mapped_commit)
        else:
            combined_df["commit_url"] = mapped_commit
    if pr_urls:
        mapped_pr = combined_df[args.id_column].map(pr_urls)
        if "pr_url" in combined_df.columns:
            combined_df["pr_url"] = combined_df["pr_url"].fillna(mapped_pr)
        else:
            combined_df["pr_url"] = mapped_pr

    if combined_df.empty:
        raise RuntimeError("No rows were sampled. Check input data and parameters.")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Sampled dataset contains {len(combined_df)} rows across {len(summary)} categories.")
    if existing_df is not None:
        retained = len(combined_df) - len(sampled_df)
        print(f"Retained {retained} existing rows and added {len(sampled_df)} new rows.")
    else:
        print(f"Added {len(sampled_df)} new rows (no prior samples provided).")
    print(f"Saved sample to {output_path}")
    print()
    print("Category summary:")
    for category, stats in summary.items():
        status = []
        if category in undersized_categories:
            status.append("undersized")
        existing_count = stats.get("existing")
        added_count = stats.get("added")
        details = (
            f"  {category}: available={stats['available']} target={stats['requested']}"
            f" existing={existing_count} added={added_count}"
        )
        if status:
            details += f" ({', '.join(status)})"
        print(details)

    if warnings:
        print()
        print("Warnings:")
        for note in warnings:
            print(f"- {note}")


if __name__ == "__main__":
    main()
