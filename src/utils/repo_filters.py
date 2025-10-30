"""
Utilities for filtering dataframes by repository whitelist files.
"""
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple
import numbers

import pandas as pd


REPO_ID_COLUMNS = [
    "repo_id",
    "repository_id",
    "repo",
    "repository",
    "id",
]

REPO_NAME_COLUMNS = [
    "repo_name",
    "repository_name",
    "full_name",
    "name",
]

COMMIT_ID_COLUMNS = [
    "commit_sha",
    "sha",
    "commit_id",
]

PR_ID_COLUMNS = [
    "pr_id",
    "pull_request_id",
    "pull_request_number",
    "number",
    "id",
]


def load_repo_whitelist(path: str | Path) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    """
    Load a repository whitelist file and return its dataframe plus identifier sets.
    """
    whitelist_path = Path(path)
    if not whitelist_path.exists():
        raise FileNotFoundError(f"Whitelist file not found: {whitelist_path}")

    if whitelist_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(whitelist_path)
    elif whitelist_path.suffix.lower() == ".csv":
        df = pd.read_csv(whitelist_path)
    else:
        raise ValueError(f"Unsupported whitelist format: {whitelist_path.suffix}")

    repo_ids, repo_names = extract_repo_identifiers(df)

    if not repo_ids and not repo_names:
        raise ValueError("Whitelist does not contain repository identifiers or names")

    return df, repo_ids, repo_names


def extract_repo_identifiers(df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """
    Return repository identifier and name sets from a dataframe.
    """
    repo_ids = _extract_values(df, REPO_ID_COLUMNS)
    repo_names = _extract_values(df, REPO_NAME_COLUMNS, normalize_case=True)
    return repo_ids, repo_names


def filter_dataframe_by_repo_list(
    df: pd.DataFrame,
    repo_ids: Iterable[str] | None = None,
    repo_names: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter a dataframe so only rows associated with whitelisted repositories remain.
    """
    repo_ids = set(str(v) for v in repo_ids or [])
    repo_names = set(_normalize_case(v) for v in (repo_names or []))

    if not repo_ids and not repo_names:
        raise ValueError("No repository identifiers provided for filtering")

    mask = pd.Series(False, index=df.index)
    matched_columns: Set[str] = set()

    if repo_ids:
        id_sets = _build_normalized_columns(df, REPO_ID_COLUMNS, as_strings=True)
        for col, series in id_sets.items():
            matches = series.isin(repo_ids)
            if matches.any():
                matched_columns.add(col)
            mask = mask | matches

    if repo_names:
        name_sets = _build_normalized_columns(df, REPO_NAME_COLUMNS, normalize_case=True)
        for col, series in name_sets.items():
            matches = series.isin(repo_names)
            if matches.any():
                matched_columns.add(col)
            mask = mask | matches

    if not matched_columns:
        raise ValueError("None of the dataframe columns matched the repository whitelist")

    filtered = df[mask].copy()
    stats = {
        "input_rows": len(df),
        "filtered_rows": len(filtered),
        "matched_columns": len(matched_columns),
    }
    return filtered, stats


def build_commit_repo_lookup(
    commit_df: pd.DataFrame,
    pr_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Build a commit -> repository mapping dataframe.
    """
    commit_col = next((col for col in COMMIT_ID_COLUMNS if col in commit_df.columns), None)
    if commit_col is None:
        raise ValueError("Commit dataframe does not contain a commit identifier column")

    repo_cols = [
        col for col in REPO_ID_COLUMNS + REPO_NAME_COLUMNS
        if col in commit_df.columns
    ]
    if repo_cols:
        lookup = (
            commit_df[[commit_col] + repo_cols]
            .dropna(subset=repo_cols, how="all")
            .drop_duplicates(subset=[commit_col])
        )
        return lookup, commit_col

    if pr_df is None:
        raise ValueError("Repository columns missing; provide PR metadata to enrich commit lookup")

    commit_pr_col = next((col for col in PR_ID_COLUMNS if col in commit_df.columns), None)
    pr_join_col = next((col for col in PR_ID_COLUMNS if col in pr_df.columns), None)
    if commit_pr_col is None or pr_join_col is None:
        raise ValueError("Could not align commit data with PR metadata for repository lookup")

    pr_repo_cols = [
        col for col in REPO_ID_COLUMNS + REPO_NAME_COLUMNS
        if col in pr_df.columns
    ]
    if not pr_repo_cols:
        raise ValueError("PR metadata does not contain repository columns")

    pr_subset = pr_df[[pr_join_col] + pr_repo_cols].drop_duplicates(subset=[pr_join_col])
    merged = commit_df[[commit_col, commit_pr_col]].merge(
        pr_subset,
        left_on=commit_pr_col,
        right_on=pr_join_col,
        how="left",
    )
    for col in {commit_pr_col, pr_join_col}:
        if col in merged.columns and col != commit_col:
            merged = merged.drop(columns=[col])
    lookup = merged.drop_duplicates(subset=[commit_col])
    if lookup.empty:
        raise ValueError("Commit lookup merge did not produce any repository matches")

    return lookup, commit_col


def _extract_values(
    df: pd.DataFrame,
    candidate_columns: Iterable[str],
    normalize_case: bool = False,
) -> Set[str]:
    result: Set[str] = set()
    for col in candidate_columns:
        if col not in df.columns:
            continue
        raw_values = df[col].dropna()
        if raw_values.empty:
            continue
        str_values = raw_values.apply(_stringify_identifier)
        str_values = str_values[str_values != ""]
        if normalize_case:
            str_values = str_values.map(_normalize_case)
        result.update(str_values.tolist())
    return result


def _build_normalized_columns(
    df: pd.DataFrame,
    candidate_columns: Iterable[str],
    *,
    as_strings: bool = False,
    normalize_case: bool = False,
) -> Dict[str, pd.Series]:
    normalized: Dict[str, pd.Series] = {}
    for col in candidate_columns:
        if col not in df.columns:
            continue
        series = df[col]
        if as_strings or normalize_case:
            series = series.map(lambda v: _stringify_identifier(v) if not pd.isna(v) else pd.NA)
        if normalize_case:
            series = series.map(lambda v: _normalize_case(v) if isinstance(v, str) else v)
        normalized[col] = series
    return normalized


def _normalize_case(value: str) -> str:
    return value.casefold()


def _stringify_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        return str(float_value)
    return str(value)
