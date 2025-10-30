#!/usr/bin/env python3
"""
Filter existing parquet or CSV artifacts so they only include whitelisted repositories.
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Set

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.repo_filters import (
    COMMIT_ID_COLUMNS,
    REPO_ID_COLUMNS,
    REPO_NAME_COLUMNS,
    build_commit_repo_lookup,
    extract_repo_identifiers,
    filter_dataframe_by_repo_list,
    load_repo_whitelist,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-list",
        default="data/filtered/java_repositories/high_star_repositories.csv",
        help="Path to the repository whitelist (CSV or parquet).",
    )
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional repository labels file (CSV or parquet) to further restrict repositories.",
    )
    parser.add_argument(
        "--allowed-labels",
        nargs="*",
        default=None,
        help=(
            "Labels to keep from --labels-file. Defaults to production_grade and specialized_project "
            "when --labels-file is provided."
        ),
    )
    parser.add_argument(
        "--exclude-labels",
        nargs="*",
        default=None,
        help="Labels to drop from --labels-file.",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=None,
        help="Minimum confidence required when --labels-file includes a confidence column.",
    )
    parser.add_argument(
        "--combine-mode",
        choices=["intersection", "union"],
        default="intersection",
        help="How to combine --repo-list and --labels-file repositories (default: intersection).",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files (parquet or csv) to filter.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for filtered outputs. Defaults to alongside each input.",
    )
    parser.add_argument(
        "--suffix",
        default="_filtered",
        help="Suffix to append when writing filtered files (ignored with --in-place).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files instead of writing separate copies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matched rows without writing files.",
    )
    parser.add_argument(
        "--pr-stats",
        default="data/filtered/java_repositories/simple_java_prs.parquet",
        help="Optional PR metadata file used to backfill repo columns when inputs lack them.",
    )
    parser.add_argument(
        "--commit-stats",
        default="data/filtered/java_repositories/java_file_commits_for_refactoring.parquet",
        help="Optional commit metadata file to backfill repo columns from commit SHA.",
    )
    return parser.parse_args()


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_dataframe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def load_optional_dataframe(path_str: str | None) -> pd.DataFrame | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        print(f"Warning: PR stats file not found at {path}; skipping metadata augmentation")
        return None
    return read_dataframe(path)


def load_repo_labels(
    path_str: str | None,
    allowed_labels: Optional[Set[str]],
    exclude_labels: Optional[Set[str]],
    min_confidence: Optional[int],
) -> tuple[pd.DataFrame | None, Set[str], Set[str], int, int]:
    if not path_str:
        return None, set(), set(), 0, 0

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    df = read_dataframe(path)
    original_count = len(df)
    filtered_df = df.copy()

    if allowed_labels:
        if "label" not in filtered_df.columns:
            raise ValueError("--allowed-labels requires a 'label' column in the labels file")
        filtered_df = filtered_df[filtered_df["label"].isin(allowed_labels)]

    if exclude_labels:
        if "label" not in filtered_df.columns:
            raise ValueError("--exclude-labels requires a 'label' column in the labels file")
        filtered_df = filtered_df[~filtered_df["label"].isin(exclude_labels)]

    if min_confidence is not None:
        if "confidence" not in filtered_df.columns:
            print("Warning: --min-confidence ignored because labels file lacks a 'confidence' column")
        else:
            confidences = pd.to_numeric(filtered_df["confidence"], errors="coerce").fillna(0)
            filtered_df = filtered_df[confidences >= min_confidence]

    filtered_count = len(filtered_df)
    if filtered_df.empty:
        print(f"Warning: labels filter produced an empty repository set from {path}")

    repo_ids, repo_names = extract_repo_identifiers(filtered_df)
    return filtered_df, repo_ids, repo_names, original_count, filtered_count


def combine_identifier_sets(
    current: Optional[Set[str]],
    new_values: Set[str],
    *,
    mode: str,
) -> Optional[Set[str]]:
    new_set = set(new_values)
    if current is None:
        return new_set
    if mode == "intersection":
        return current & new_set
    return current | new_set


def augment_with_pr_metadata(df: pd.DataFrame, pr_stats: pd.DataFrame) -> pd.DataFrame:
    pr_join_cols = ["pr_id", "pull_request_id", "pull_request_number"]
    join_col = next((col for col in pr_join_cols if col in df.columns and col in pr_stats.columns), None)
    if join_col is None:
        raise ValueError("Could not find a common PR identifier column to merge metadata")
    
    repo_columns = set(REPO_ID_COLUMNS + REPO_NAME_COLUMNS + ["repo_stars", "repo_forks", "repo_language", "stars", "forks", "full_name", "language"])
    available_repo_cols = [col for col in repo_columns if col in pr_stats.columns]
    if not available_repo_cols:
        raise ValueError("PR metadata does not contain repository columns to merge")
    
    merge_cols = [join_col] + available_repo_cols
    merge_frame = (
        pr_stats[merge_cols]
        .drop_duplicates(subset=[join_col])
    )
    augmented = df.merge(merge_frame, on=join_col, how="left")
    return augmented


def augment_with_commit_metadata(
    df: pd.DataFrame,
    commit_stats: pd.DataFrame,
    pr_stats: pd.DataFrame | None,
) -> pd.DataFrame:
    df_commit_col = next((col for col in COMMIT_ID_COLUMNS if col in df.columns), None)
    if df_commit_col is None:
        raise ValueError("Input data does not contain a commit identifier column")
    
    lookup, commit_col = build_commit_repo_lookup(commit_stats, pr_stats)
    augmented = df.merge(lookup, left_on=df_commit_col, right_on=commit_col, how="left")
    if commit_col != df_commit_col and commit_col in augmented.columns:
        augmented = augmented.drop(columns=[commit_col])
    return augmented


GITHUB_URL_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+)/")
URL_CANDIDATE_COLUMNS = [
    "html_url",
    "url",
    "commit_url",
    "compare_url",
    "repository_url",
]


def _extract_repo_from_url(url: object) -> str | None:
    if not isinstance(url, str):
        return None
    match = GITHUB_URL_RE.search(url)
    if not match:
        return None
    owner, repo = match.groups()
    if repo.endswith(".git"):
        repo = repo[:-4]
    return f"{owner}/{repo}"


def augment_with_inferred_repository(df: pd.DataFrame) -> pd.DataFrame | None:
    if "repo_name" in df.columns:
        return df
    for url_col in URL_CANDIDATE_COLUMNS:
        if url_col not in df.columns:
            continue
        extracted = df[url_col].map(_extract_repo_from_url)
        if extracted.notna().any():
            augmented = df.copy()
            augmented["repo_name"] = extracted
            return augmented
    return None


def determine_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    if args.in_place:
        return input_path
    if args.output_dir:
        return Path(args.output_dir) / input_path.name
    return input_path.with_name(f"{input_path.stem}{args.suffix}{input_path.suffix}")


def main():
    args = parse_args()

    combine_mode = args.combine_mode
    repo_ids: Optional[Set[str]] = None
    repo_names: Optional[Set[str]] = None

    repo_list_path = args.repo_list.strip() if args.repo_list else None

    whitelist_df: pd.DataFrame | None = None
    if repo_list_path:
        whitelist_df, list_repo_ids, list_repo_names = load_repo_whitelist(repo_list_path)
        print(f"Loaded whitelist with {len(whitelist_df)} repositories from {repo_list_path}")
        repo_ids = combine_identifier_sets(repo_ids, list_repo_ids, mode=combine_mode)
        repo_names = combine_identifier_sets(repo_names, list_repo_names, mode=combine_mode)

    labels_file = args.labels_file.strip() if args.labels_file else None

    allowed_labels = set(args.allowed_labels) if args.allowed_labels else None
    if labels_file and allowed_labels is None:
        allowed_labels = {"production_grade", "specialized_project"}
    exclude_labels = set(args.exclude_labels) if args.exclude_labels else None

    _labels_df, label_repo_ids, label_repo_names, original_label_count, filtered_label_count = load_repo_labels(
        labels_file,
        allowed_labels,
        exclude_labels,
        args.min_confidence,
    )
    if labels_file:
        print(f"Loaded {original_label_count} labeled repositories from {labels_file}")
        if allowed_labels or exclude_labels or args.min_confidence is not None:
            print(f"Label filters kept {filtered_label_count} repositories")
        repo_ids = combine_identifier_sets(repo_ids, label_repo_ids, mode=combine_mode)
        repo_names = combine_identifier_sets(repo_names, label_repo_names, mode=combine_mode)

    final_repo_ids = repo_ids if repo_ids is not None else set()
    final_repo_names = repo_names if repo_names is not None else set()

    if not final_repo_ids and not final_repo_names:
        raise ValueError("No repositories remain after applying whitelist/label criteria")

    print(
        f"Filtering with {len(final_repo_ids)} repo ids and {len(final_repo_names)} normalized repo names "
        f"(combine mode: {combine_mode})"
    )

    pr_stats_df = load_optional_dataframe(args.pr_stats)
    commit_stats_df = load_optional_dataframe(args.commit_stats)
    
    for input_file in args.inputs:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Skipping missing input: {input_path}")
            continue
        
        df = read_dataframe(input_path)
        augmented_df = None
        try:
            filtered_df, stats = filter_dataframe_by_repo_list(df, final_repo_ids, final_repo_names)
        except ValueError as err:
            if pr_stats_df is None or "matched the repository whitelist" not in str(err):
                augmented_df = None
            else:
                try:
                    augmented_df = augment_with_pr_metadata(df, pr_stats_df)
                except ValueError:
                    augmented_df = None
            if augmented_df is None and commit_stats_df is not None and "matched the repository whitelist" in str(err):
                try:
                    augmented_df = augment_with_commit_metadata(df, commit_stats_df, pr_stats_df)
                except ValueError:
                    augmented_df = None
            if augmented_df is None:
                augmented_df = augment_with_inferred_repository(df)
            if augmented_df is None:
                raise err
            filtered_df, stats = filter_dataframe_by_repo_list(augmented_df, final_repo_ids, final_repo_names)
            df = augmented_df
        
        print(
            f"{input_path}: kept {stats['filtered_rows']}/{stats['input_rows']} rows "
            f"using {stats['matched_columns']} matching columns"
        )
        
        if args.dry_run:
            continue
        
        output_path = determine_output_path(input_path, args)
        write_dataframe(filtered_df, output_path)
        print(f"Wrote filtered data to {output_path}")


if __name__ == "__main__":
    main()
