#!/usr/bin/env python3
"""Stratified proportional sampling of GPT-labeled repositories for manual validation."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/filtered/java_repositories/gpt_repository_labels_updated.csv")
DEFAULT_OUTPUT_DIR = Path("data/analysis/project_filtering")

RANDOM_SEED = 42


def required_sample_size(
    population: int,
    confidence_z: float = 1.96,
    margin: float = 0.05,
    p: float = 0.5,
) -> int:
    """Cochran's formula with finite population correction."""
    numerator = population * (confidence_z ** 2) * p * (1 - p)
    denominator = (margin ** 2) * (population - 1) + (confidence_z ** 2) * p * (1 - p)
    return math.ceil(numerator / denominator)


def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    total_n: int,
    seed: int,
) -> pd.DataFrame:
    """Proportional stratified random sampling."""
    counts = df[label_col].value_counts()
    fractions = counts / len(df)
    allocations = (fractions * total_n).round().astype(int)

    # Adjust rounding so total matches
    diff = total_n - allocations.sum()
    if diff != 0:
        remainders = (fractions * total_n) - allocations.astype(float)
        if diff > 0:
            top = remainders.nlargest(abs(diff)).index
            allocations[top] += 1
        else:
            top = remainders.nsmallest(abs(diff)).index
            allocations[top] -= 1

    samples = []
    for label, n in allocations.items():
        stratum = df[df[label_col] == label]
        sampled = stratum.sample(n=min(n, len(stratum)), random_state=seed)
        samples.append(sampled)

    return pd.concat(samples, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to GPT repository labels CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margin of error (default: 0.05)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    # Migrate legacy 4-category labels to binary scheme
    _LEGACY_LABEL_MAP = {
        "production_grade": "non_toy",
        "specialized_project": "non_toy",
        "toy_or_example": "toy",
        "uncertain": "uncertain",
    }
    if "label" in df.columns:
        migrated = df["label"].map(_LEGACY_LABEL_MAP)
        changed = migrated.notna() & (migrated != df["label"])
        if changed.any():
            df["label"] = migrated.fillna(df["label"])
            print(f"Migrated {changed.sum()} legacy labels to ternary scheme (non_toy/toy/uncertain).")

    population = len(df)

    confidence_z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(args.confidence, 1.96)
    total_n = required_sample_size(population, confidence_z=confidence_z, margin=args.margin)

    print(f"Population: {population}")
    print(f"Required sample size ({args.confidence*100:.0f}% confidence, {args.margin*100:.0f}% margin): {total_n}")
    print(f"Label distribution:")
    for label, count in df["label"].value_counts().items():
        alloc = round(count / population * total_n)
        print(f"  {label}: {count} ({count/population*100:.1f}%) -> sample {alloc}")
    print()

    sampled = stratified_sample(df, "label", total_n, seed=args.seed)

    # Build GitHub URL
    sampled["github_url"] = "https://github.com/" + sampled["owner"] + "/" + sampled["repo"]

    # Reorder and rename columns for full version
    full_cols = [
        "repo_name", "owner", "repo", "github_url",
        "repo_stars", "repo_forks",
        "summary_context", "readme_excerpt",
        "label", "reason", "confidence",
    ]
    full_df = sampled[[c for c in full_cols if c in sampled.columns]].copy()
    full_df = full_df.rename(columns={
        "label": "gpt_label",
        "reason": "gpt_reason",
        "confidence": "gpt_confidence",
    })

    # Add empty annotation columns
    full_df["annotator1_label"] = ""
    full_df["annotator1_reason"] = ""
    full_df["annotator2_label"] = ""
    full_df["annotator2_reason"] = ""
    full_df["conflict"] = ""
    full_df["third_label"] = ""
    full_df["final_human_label"] = ""

    # Save full version (master)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "project_filtering_validation_full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"Saved full version ({len(full_df)} rows): {full_path}")

    # Build blind version (no GPT labels)
    blind_cols = [
        "repo_name", "owner", "repo", "github_url",
        "repo_stars", "repo_forks",
        "summary_context", "readme_excerpt",
        "annotator1_label", "annotator1_reason",
        "annotator2_label", "annotator2_reason",
    ]
    blind_df = full_df[[c for c in blind_cols if c in full_df.columns]].copy()
    blind_path = args.output_dir / "project_filtering_validation_blind.csv"
    blind_df.to_csv(blind_path, index=False)
    print(f"Saved blind version ({len(blind_df)} rows): {blind_path}")

    # Print summary
    print(f"\nSample distribution:")
    print(full_df["gpt_label"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
