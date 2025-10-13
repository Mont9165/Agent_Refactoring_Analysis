#!/usr/bin/env python3
"""Summarise commit/PR characteristics for the Java refactoring dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Set

import pandas as pd

DATA_PATH = Path("data/filtered/java_repositories/java_pr_commits_no_merges.parquet")
REFACTORING_COMMITS_PATH = Path("data/analysis/refactoring_instances/refactoring_commits.parquet")


def _load_commit_table() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing parquet at data/filtered/java_repositories/java_pr_commits_no_merges.parquet."
            " Run scripts/2_extract_commits.py first."
        )
    return pd.read_parquet(DATA_PATH)


def get_java_commit_dataframe() -> pd.DataFrame:
    df = _load_commit_table()
    commit_cols = [
        "sha",
        "commit_stats_total",
        "commit_stats_additions",
        "commit_stats_deletions",
        "pr_id",
        "state",
        "is_merged",
        "agent",
        "html_url",
    ]
    commit_level = df[commit_cols].drop_duplicates("sha").set_index("sha")
    total_files = df.groupby("sha").size()
    java_files = df[df.get("is_java_file", False)].groupby("sha").size()
    commit_level["file_changes_total"] = total_files
    commit_level["file_changes_java"] = java_files.fillna(0)
    java_commits = commit_level[commit_level["file_changes_java"] > 0]
    return java_commits.reset_index()


def _commit_level_stats(commit_df: pd.DataFrame) -> Dict[str, float]:
    java_commits = commit_df
    stats = {
        "total_commits": float(len(java_commits)),
        "files_changed_mean": float(java_commits[["file_changes_total"]].mean()),
        "files_changed_median": float(java_commits["file_changes_total"].median()),
        "java_files_changed_mean": float(java_commits["file_changes_java"].mean()),
        "java_files_changed_median": float(java_commits["file_changes_java"].median()),
        "additions_mean": java_commits[java_commits["commit_stats_additions"] > 0]["commit_stats_additions"].mean(),
        "additions_median": java_commits[java_commits["commit_stats_additions"] > 0]["commit_stats_additions"].median(),
        "deletions_mean": java_commits[java_commits["commit_stats_deletions"] > 0]["commit_stats_deletions"].mean(),
        "deletions_median": java_commits[java_commits["commit_stats_deletions"] > 0]["commit_stats_deletions"].median(),
        "changes_mean": java_commits[java_commits["commit_stats_total"] > 0]["commit_stats_total"].mean(),
        "changes_median": java_commits[java_commits["commit_stats_total"] > 0]["commit_stats_total"].median(),
        "agent_share_pct": float(java_commits["agent"].notna().mean() * 100.0),
    }

    state_counts = java_commits["state"].fillna("unknown").value_counts()
    stats["pr_state_closed_commits"] = int(state_counts.get("closed", 0))
    stats["pr_state_open_commits"] = int(state_counts.get("open", 0))

    closed_mask = java_commits["state"] == "closed"
    merged_counts = java_commits.loc[closed_mask, "is_merged"].fillna(False).value_counts()
    stats["merged_commits_in_closed_prs"] = int(merged_counts.get(True, 0))
    stats["unmerged_commits_in_closed_prs"] = int(merged_counts.get(False, 0))

    return stats


def _pr_level_stats(commit_df: pd.DataFrame) -> Dict[str, float]:
    pr_group = commit_df.groupby("pr_id")
    pr_state = pr_group["state"].first().fillna("unknown")
    pr_merged = pr_group["is_merged"].first().fillna(False)

    stats = {
        "total_prs": float(len(pr_group)),
        "closed_prs": float((pr_state == "closed").sum()),
        "open_prs": float((pr_state == "open").sum()),
        "merged_prs": float(pr_merged.sum()),
    }
    return stats


def _safe_read_parquet(path: Path, column: str) -> Set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=[column])
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not read {path}: {exc}")
        return set()
    return set(df[column].dropna().astype(str))


def _load_refactoring_commit_shas() -> Set[str]:
    shas = _safe_read_parquet(REFACTORING_COMMITS_PATH, "sha")
    if shas:
        return shas

    csv_path = REFACTORING_COMMITS_PATH.with_suffix(".csv")
    if not csv_path.exists():
        return set()

    try:
        df = pd.read_csv(csv_path, usecols=["sha"])
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not read {csv_path}: {exc}")
        return set()

    return set(df["sha"].dropna().astype(str))


def _coverage_stats(target_commit_shas: Set[str], sample_limit: int) -> Dict[str, Dict[str, object]]:
    base = Path("data/analysis")
    designite_dir = base / "designite" / "deltas"
    type_shas = _safe_read_parquet(designite_dir / "type_metric_deltas.parquet", "commit_sha")
    method_shas = _safe_read_parquet(designite_dir / "method_metric_deltas.parquet", "commit_sha")
    designite_shas = type_shas | method_shas

    readability_path = base / "readability" / "readability_deltas.parquet"
    readability_shas = _safe_read_parquet(readability_path, "commit_sha")

    def summarize(processed: Set[str]) -> Dict[str, object]:
        processed = processed & target_commit_shas
        missing = target_commit_shas - processed
        return {
            "processed_commits": len(processed),
            "missing_commits": len(missing),
            "coverage_pct": (len(processed) / len(target_commit_shas) * 100.0) if target_commit_shas else 0.0,
            "missing_samples": list(sorted(missing))[: max(sample_limit, 0)],
        }

    return {
        "designite": summarize(designite_shas),
        "readability": summarize(readability_shas),
    }


def _agent_stats(commit_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    commit_counts = commit_df["agent"].fillna("unknown").value_counts(sort=True)
    commit_pct = (commit_counts / commit_counts.sum() * 100).round(2)

    pr_group = commit_df.groupby("pr_id")
    pr_agents = pr_group["agent"].first().fillna("unknown")
    pr_counts = pr_agents.value_counts(sort=True)
    pr_pct = (pr_counts / pr_counts.sum() * 100).round(2)

    all_agents = sorted(set(commit_counts.index) | set(pr_counts.index))
    stats: Dict[str, Dict[str, float]] = {}
    for agent in all_agents:
        stats[agent] = {
            "commit_count": int(commit_counts.get(agent, 0)),
            "commit_pct": float(commit_pct.get(agent, 0.0)),
            "pr_count": int(pr_counts.get(agent, 0)),
            "pr_pct": float(pr_pct.get(agent, 0.0)),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Emit compact JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=5,
        help="List up to N missing commit SHAs per coverage category (default: 5).",
    )
    args = parser.parse_args()

    commit_df = get_java_commit_dataframe()
    commit_stats = _commit_level_stats(commit_df)
    pr_stats = _pr_level_stats(commit_df)
    refactoring_commit_shas = _load_refactoring_commit_shas()
    coverage_target_shas = refactoring_commit_shas or set(commit_df["sha"].astype(str))
    coverage = _coverage_stats(coverage_target_shas, args.show_missing)
    repo_count = (
        commit_df["html_url"].dropna()
        .apply(lambda url: "/".join(url.split("/")[3:5]) if isinstance(url, str) and url.count("/") >= 4 else None)
        .dropna()
        .nunique()
    )
    agent_stats = _agent_stats(commit_df)

    if args.as_json:
        import json

        payload = {
            "commit": commit_stats,
            "pr": pr_stats,
            "coverage": coverage,
            "repositories": int(repo_count),
            "agents": agent_stats,
        }
        print(json.dumps(payload, indent=2))
        return

    print("Commit-level summary (Java-touching commits)")
    for key, value in commit_stats.items():
        print(f"  {key}: {value}")
    print()
    print("Pull-request summary")
    for key, value in pr_stats.items():
        print(f"  {key}: {value}")
    print()
    print(f"Unique repositories: {repo_count}")
    print()
    print("Analysis coverage")
    for name, stats in coverage.items():
        print(f"  {name}:")
        print(f"    processed commits: {stats['processed_commits']}")
        print(f"    missing commits:  {stats['missing_commits']}")
        print(f"    coverage (%):     {stats['coverage_pct']:.2f}")
        if args.show_missing > 0 and stats['missing_commits'] > 0:
            sample = stats['missing_samples'][: args.show_missing]
            if sample:
                print(f"    sample missing:   {', '.join(sample)}")
    print()
    print("Agent distribution")
    for agent, stats in agent_stats.items():
        print(
            f"  {agent}: commits={stats['commit_count']} ({stats['commit_pct']:.2f}%),"
            f" PRs={stats['pr_count']} ({stats['pr_pct']:.2f}%)"
        )


if __name__ == "__main__":
    main()
