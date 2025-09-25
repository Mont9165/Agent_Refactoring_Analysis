#!/usr/bin/env python3
"""
Prepare tools/DesigniteRunner/projects.txt based on the dataset.

By default, uses commit-level results and includes only repositories that have
refactoring commits. You can customize with env vars:

- SOURCE:
  - commits_with_refactoring (default): data/analysis/refactoring_instances/commits_with_refactoring.parquet
  - java_commits: data/filtered/java_repositories/java_file_commits_for_refactoring.parquet

- ONLY_AGENTIC: 1 to include only agentic repos (requires 'agent' column)

Writes unique owner/repo lines in tools/DesigniteRunner/projects.txt
"""
import os
import sys
from pathlib import Path
import pandas as pd


def extract_owner_repo(url: str) -> str | None:
    try:
        # Accept PR URLs or repo URLs
        # Example PR: https://github.com/owner/repo/pull/123
        parts = url.split('/')
        if 'github.com' not in parts[2]:
            return None
        owner, repo = parts[3], parts[4]
        # strip trailing .git if any
        if repo.endswith('.git'):
            repo = repo[:-4]
        return f"{owner}/{repo}"
    except Exception:
        return None


def load_source() -> pd.DataFrame:
    preferred = Path('data/analysis/refactoring_instances/commits_with_refactoring.parquet')
    fallback = Path('data/filtered/java_repositories/java_file_commits_for_refactoring.parquet')
    source = os.environ.get('SOURCE', 'commits_with_refactoring')
    if source == 'java_commits' and fallback.exists():
        return pd.read_parquet(fallback)
    if preferred.exists():
        return pd.read_parquet(preferred)
    if fallback.exists():
        return pd.read_parquet(fallback)
    raise FileNotFoundError('No suitable source parquet found.')


def main():
    df = load_source()
    only_agentic = os.environ.get('ONLY_AGENTIC', '0') not in ('0', 'false', 'False')

    # Filter to refactoring commits if available
    # if 'has_refactoring' in df.columns:
    #     df = df[df['has_refactoring'] == True]
    # # Filter to agentic if requested
    # if only_agentic and 'agent' in df.columns:
    #     df = df[df['agent'].notna()]

    if 'html_url' not in df.columns:
        print('ERROR: html_url column not found; cannot infer owner/repo. Aborting.')
        sys.exit(1)

    urls = df['html_url'].dropna().astype(str)
    repos = set()
    for u in urls:
        rep = extract_owner_repo(u)
        if rep:
            repos.add(rep)

    if not repos:
        print('No repositories found to write to projects.txt')
        sys.exit(1)

    out_path = Path('tools/DesigniteRunner/projects.txt')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = sorted(repos)
    out_path.write_text('\n'.join(lines) + '\n')

    print(f"Wrote {len(lines)} repositories to {out_path}")


if __name__ == '__main__':
    main()

