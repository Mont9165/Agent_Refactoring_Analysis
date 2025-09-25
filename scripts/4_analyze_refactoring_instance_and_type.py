#!/usr/bin/env python3
"""
Phase 3.B: Analyze refactoring instances and types.

Prefers raw RefactoringMiner JSON files for source-of-truth, with fallback to
refminer_refactorings.parquet if raw files are unavailable.

Reads:
- data/analysis/refactoring_instances/refminer_raw/**/<commit>.json (if present)
- data/analysis/refactoring_instances/refminer_refactorings.parquet (fallback)
- data/filtered/java_repositories/java_file_commits_for_refactoring.parquet

Produces commit-level outputs:
- data/analysis/refactoring_instances/commits_with_refactoring.parquet
- data/analysis/refactoring_instances/refactoring_commits.parquet
- data/analysis/refactoring_instances/refactoring_commits.csv
"""
import os
import sys
import time
import json
from typing import List, Dict
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.phase3_refactoring_analysis.refminer_wrapper import RefactoringMinerWrapper


OUT_DIR = Path('data/analysis/refactoring_instances')
RAW_DIR = OUT_DIR / 'refminer_raw'


def load_inputs():
    commits_path_pq = 'data/filtered/java_repositories/java_file_commits_for_refactoring.parquet'
    commits_path_csv = 'data/filtered/java_repositories/java_file_commits_for_refactoring.csv'
    rm_path = OUT_DIR / 'refminer_refactorings.parquet'

    if os.path.exists(commits_path_pq):
        commits = pd.read_parquet(commits_path_pq)
    elif os.path.exists(commits_path_csv):
        commits = pd.read_csv(commits_path_csv)
    else:
        raise FileNotFoundError('Missing java commits parquet/csv under data/filtered/java_repositories')

    # Prefer raw JSON if available
    rm = None
    if RAW_DIR.exists():
        json_files = list(RAW_DIR.glob('**/*.json'))
        if len(json_files) > 0:
            print(f"Parsing {len(json_files)} raw RefactoringMiner JSON files…")
            rm = parse_raw_refminer(json_files)
            # persist parsed table for downstream steps
            if not rm.empty:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                rm.to_parquet(OUT_DIR / 'refminer_refactorings.parquet', index=False)
                rm.to_csv(OUT_DIR / 'refminer_refactorings.csv', index=False)
    if rm is None:
        if rm_path.exists():
            print("Using refminer_refactorings.parquet (raw JSON not found).")
            rm = pd.read_parquet(rm_path)
        else:
            print('Warning: No RefactoringMiner results found; analysis will mark no-refactoring.')
            rm = pd.DataFrame(columns=['commit_sha', 'refactoring_type', 'description'])
    return commits, rm


def parse_raw_refminer(json_files: List[Path]) -> pd.DataFrame:
    """Parse a list of raw RefactoringMiner JSON files into a flat DataFrame."""
    rows: List[Dict] = []
    for path in json_files:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        def add_rows(commit_sha: str, refs: List[Dict]):
            for ref in refs or []:
                rows.append({
                    'commit_sha': commit_sha,
                    'refactoring_type': ref.get('type'),
                    'description': ref.get('description'),
                    'left_side_locations': json.dumps(ref.get('leftSideLocations', [])),
                    'right_side_locations': json.dumps(ref.get('rightSideLocations', [])),
                })

        # Two possible shapes
        if isinstance(data, dict) and 'commits' in data and isinstance(data['commits'], list):
            for c in data['commits']:
                sha = c.get('sha1') or c.get('sha')
                refs = c.get('refactorings', [])
                if sha:
                    add_rows(sha, refs)
        elif isinstance(data, dict) and 'refactorings' in data:
            # File might not contain sha; derive from filename
            sha = path.stem
            add_rows(sha, data.get('refactorings', []))
        else:
            # unknown shape; skip
            continue

    return pd.DataFrame(rows)


def detect_self_affirmation(messages: pd.Series) -> pd.Series:
    pattern = r'\brefactor(?:ing|ed)?\b|\brestructur(?:e|ed|ing)\b|\bclean(?:ed|ing)?\s*up\b|\breorganiz(?:e|ed|ing)\b'
    return messages.str.contains(pattern, case=False, na=False, regex=True)


def analyze(commits_df: pd.DataFrame, rm_df: pd.DataFrame) -> pd.DataFrame:
    # Unique commit-level base with key metadata
    keep_pref = ['sha', 'agent', 'author', 'committer', 'message', 'state', 'html_url', 'title', 'is_merged', 'java_files_percentage']
    keep_cols = [c for c in keep_pref if c in commits_df.columns]
    base = commits_df.drop_duplicates('sha')[keep_cols].copy()

    # Aggregate RM results per commit
    if not rm_df.empty:
        per_commit = rm_df.groupby('commit_sha')['refactoring_type'].agg(list)
        per_commit = per_commit.rename_axis('sha').reset_index().rename(columns={'refactoring_type': 'refactoring_types'})
        base = base.merge(per_commit, on='sha', how='left')
        base['has_refactoring'] = base['refactoring_types'].apply(lambda x: isinstance(x, list) and len(x) > 0)
        base['refactoring_instance_count'] = base['refactoring_types'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        base['has_refactoring'] = False
        base['refactoring_instance_count'] = 0

    # Self-affirmation flag
    if 'message' in base.columns:
        base['is_self_affirmed'] = detect_self_affirmation(base['message'])
    else:
        base['is_self_affirmed'] = False

    # Ordering for convenience
    base = base.sort_values(['has_refactoring', 'refactoring_instance_count'], ascending=[False, False])
    return base


def determine_missing_commits(commits: pd.DataFrame) -> pd.DataFrame:
    """Return subset of commits for which raw JSON is missing."""
    # Determine candidate commits for RM based on environment
    local_repo = os.environ.get('REFMINER_LOCAL_REPO')
    if local_repo and os.path.exists(local_repo):
        candidates = commits.drop_duplicates('sha')
    else:
        if 'html_url' not in commits.columns:
            print("No 'html_url' column and no REFMINER_LOCAL_REPO; cannot determine repo for RM. Skipping auto-apply.")
            return commits.iloc[0:0]
        candidates = commits[commits['html_url'].notna()].drop_duplicates('sha')

    # Existing raw JSON SHAs from filenames
    existing = set([p.stem for p in RAW_DIR.glob('**/*.json')]) if RAW_DIR.exists() else set()
    missing_mask = ~candidates['sha'].astype(str).isin(existing)
    missing = candidates[missing_mask].copy()
    return missing


def main():
    print("=" * 60)
    print("       PHASE 3.B: ANALYZE REFACTORING INSTANCES & TYPES")
    print("=" * 60)

    start = time.time()
    try:
        commits, rm = load_inputs()
        print(f"Commits: {commits['sha'].nunique()} unique; RM rows: {len(rm)}")

        # Auto-apply RM for missing commits if enabled
        auto_apply = os.environ.get('REFMINER_AUTO_APPLY', '1') not in ('0', 'false', 'False')
        if auto_apply:
            missing = determine_missing_commits(commits)
            if len(missing) > 0:
                print(f"Missing raw JSON for {len(missing)} commits → applying RefactoringMiner now…")
                rmw = RefactoringMinerWrapper()
                if rmw.is_available():
                    max_missing = int(os.environ.get('REFMINER_MAX_COMMITS', len(missing)))
                    to_run = missing.head(max_missing)
                    # This call will also save raw JSON under refminer_raw/
                    _ = rmw.analyze_commits_batch(to_run, max_commits=max_missing)
                    # Re-parse raw after applying
                    json_files = list(RAW_DIR.glob('**/*.json'))
                    if len(json_files) > 0:
                        rm = parse_raw_refminer(json_files)
                        if not rm.empty:
                            OUT_DIR.mkdir(parents=True, exist_ok=True)
                            rm.to_parquet(OUT_DIR / 'refminer_refactorings.parquet', index=False)
                            rm.to_csv(OUT_DIR / 'refminer_refactorings.csv', index=False)
                else:
                    print("RefactoringMiner not available; cannot auto-apply for missing commits.")

        commit_level = analyze(commits, rm)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save commit-level datasets
        all_path = OUT_DIR / 'commits_with_refactoring.parquet'
        pos_path_pq = OUT_DIR / 'refactoring_commits.parquet'
        pos_path_csv = OUT_DIR / 'refactoring_commits.csv'

        commit_level.to_parquet(all_path, index=False)
        positive = commit_level[commit_level['has_refactoring']]
        positive.to_parquet(pos_path_pq, index=False)
        positive.to_csv(pos_path_csv, index=False)

        print(f"Saved {len(positive)} refactoring commits (commit-level) to {OUT_DIR}")

        # Refactoring-level analysis (instance counts)
        if not rm.empty:
            # Map commit meta onto instances
            meta_cols = [c for c in ['sha', 'agent', 'is_self_affirmed'] if c in commit_level.columns]
            instance_level = rm.merge(commit_level[meta_cols], left_on='commit_sha', right_on='sha', how='left')

            # Overall instance counts by type
            overall_counts = instance_level['refactoring_type'].value_counts()
            print("Refactoring-level counts (top 10):")
            for k, v in overall_counts.head(10).items():
                print(f"  {k}: {v}")

            # Save overall
            counts_overall_csv = OUT_DIR / 'refactoring_type_counts_overall.csv'
            counts_overall_json = OUT_DIR / 'refactoring_type_counts_overall.json'
            overall_counts.to_csv(counts_overall_csv, header=['count'])
            with open(counts_overall_json, 'w') as f:
                json.dump(overall_counts.to_dict(), f, indent=2)

            # Agentic-only counts (if available)
            if 'agent' in instance_level.columns:
                agentic = instance_level[instance_level['agent'].notna()]
                agent_counts = agentic['refactoring_type'].value_counts()
                (OUT_DIR / 'refactoring_type_counts_agentic.csv').write_text(agent_counts.to_csv(header=['count']))
                with open(OUT_DIR / 'refactoring_type_counts_agentic.json', 'w') as f:
                    json.dump(agent_counts.to_dict(), f, indent=2)

            # SAR-only counts (if available)
            if 'is_self_affirmed' in instance_level.columns:
                sar = instance_level[instance_level['is_self_affirmed'] == True]
                sar_counts = sar['refactoring_type'].value_counts()
                (OUT_DIR / 'refactoring_type_counts_sar.csv').write_text(sar_counts.to_csv(header=['count']))
                with open(OUT_DIR / 'refactoring_type_counts_sar.json', 'w') as f:
                    json.dump(sar_counts.to_dict(), f, indent=2)

            # Summary: overall vs SAR
            summary = {
                'instances_overall': int(len(instance_level)),
                'commits_overall_with_refactoring': int(instance_level['commit_sha'].nunique()),
            }
            if 'agent' in instance_level.columns:
                summary['instances_agentic'] = int(instance_level[instance_level['agent'].notna()].shape[0])
            if 'is_self_affirmed' in instance_level.columns:
                summary['instances_sar'] = int(instance_level[instance_level['is_self_affirmed'] == True].shape[0])
                if summary['instances_overall'] > 0:
                    summary['sar_instance_percentage'] = summary['instances_sar'] / summary['instances_overall'] * 100
                else:
                    summary['sar_instance_percentage'] = 0.0

            # Per-type SAR breakdown
            per_type = {}
            for t, c in overall_counts.items():
                entry = {'overall': int(c)}
                if 'is_self_affirmed' in instance_level.columns:
                    s = int(instance_level[(instance_level['refactoring_type'] == t) & (instance_level['is_self_affirmed'] == True)].shape[0])
                    entry['sar'] = s
                    entry['sar_pct'] = (s / c * 100) if c > 0 else 0.0
                per_type[t] = entry
            summary['per_type'] = per_type

            with open(OUT_DIR / 'refactoring_instances_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"Overall instances: {summary['instances_overall']}")
            if 'instances_sar' in summary:
                print(f"SAR instances: {summary['instances_sar']} ({summary['sar_instance_percentage']:.1f}%)")

        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print("REFACTORING ANALYSIS COMPLETED")
        print("=" * 60)
        print(f"Execution time: {elapsed:.1f}s")
        print(f"Outputs: {OUT_DIR}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
