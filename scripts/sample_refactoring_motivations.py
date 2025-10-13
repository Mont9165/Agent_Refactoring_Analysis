#!/usr/bin/env python3
"""Randomly sample refactoring motivation commits per category."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
        default=15,
        help="Maximum number of commits to sample for each category (default: 15).",
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
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, int]], set[str]]:
    rng = random.Random(seed)
    sampled_frames: List[pd.DataFrame] = []
    warnings: List[str] = []
    summary: Dict[str, Dict[str, int]] = {}
    undersized_categories: set[str] = set()

    for category, group in sorted(df.groupby(category_column), key=lambda item: str(item[0])):
        available = len(group)
        size, undersized_flag = _choose_sample_size(rng, available, min_per_category, max_per_category)
        summary[str(category)] = {
            "available": int(available),
            "requested": int(size),
        }

        if size == 0:
            warnings.append(f"Category '{category}' has no rows to sample.")
            continue

        if undersized_flag:
            warnings.append(
                f"Category '{category}' has only {available} rows (less than requested minimum {min_per_category})."
            )
            undersized_categories.add(str(category))

        sample_state = rng.randint(0, 2**32 - 1)
        sampled = group.sample(n=size, random_state=sample_state).copy()
        sampled_frames.append(sampled)

    if not sampled_frames:
        return pd.DataFrame(), warnings, summary, undersized_categories

    combined = pd.concat(sampled_frames, ignore_index=True)
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

    sampled_df, warnings, summary, undersized_categories = sample_per_category(
        df,
        category_column=args.category_column,
        min_per_category=args.min_per_category,
        max_per_category=args.max_per_category,
        seed=args.seed,
    )

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

    if commit_urls:
        sampled_df["commit_url"] = sampled_df["sha"].map(commit_urls)
    if pr_urls:
        sampled_df["pr_url"] = sampled_df["sha"].map(pr_urls)

    if sampled_df.empty:
        raise RuntimeError("No rows were sampled. Check input data and parameters.")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False)

    print(f"Sampled {len(sampled_df)} rows across {len(summary)} categories.")
    print(f"Saved sample to {output_path}")
    print()
    print("Category summary:")
    for category, stats in summary.items():
        status = []
        if category in undersized_categories:
            status.append("undersized")
        print(
            f"  {category}: available={stats['available']} sampled={stats['requested']}"
            + (f" ({', '.join(status)})" if status else "")
        )

    if warnings:
        print()
        print("Warnings:")
        for note in warnings:
            print(f"- {note}")


if __name__ == "__main__":
    main()
