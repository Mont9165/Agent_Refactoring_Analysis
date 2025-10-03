#!/usr/bin/env python3
"""
Phase 3.A: Apply RefactoringMiner to Java commits and persist results.

Outputs under data/analysis/refactoring_instances/:
- refminer_refactorings.parquet / .csv (parsed instances)
- refminer_analysis.json (summary)
- refminer_raw/<owner>/<repo>/<sha>.json (one per commit; if REFMINER_SAVE_JSON != 0)
"""
import os
import sys
import time
from pathlib import Path
from typing import Set

import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase3_refactoring_analysis.refminer_wrapper import RefactoringMinerWrapper


REFMINER_OUTPUT_DIR = Path("data/analysis/refactoring_instances")
REFMINER_PARQUET = REFMINER_OUTPUT_DIR / "refminer_refactorings.parquet"


def load_java_commits() -> pd.DataFrame:
    parquet_path = "data/filtered/java_repositories/java_file_commits_for_refactoring.parquet"
    csv_path = "data/filtered/java_repositories/java_file_commits_for_refactoring.csv"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} Java file changes from parquet")
        return df
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} Java file changes from CSV")
        return df
    raise FileNotFoundError("Missing java commits input: data/filtered/java_repositories/java_file_commits_for_refactoring.(parquet|csv)")


def _collect_cached_shas(rm: RefactoringMinerWrapper) -> Set[str]:
    """Return set of commit SHAs that already have RefactoringMiner output."""
    cached: set[str] = set()

    raw_dir = getattr(rm, "raw_json_dir", None)
    if raw_dir and raw_dir.exists():
        cached.update(path.stem for path in raw_dir.glob("**/*.json"))

    if REFMINER_PARQUET.exists():
        try:
            existing = pd.read_parquet(REFMINER_PARQUET, columns=["commit_sha"])
            cached.update(existing["commit_sha"].astype(str))
        except Exception:
            # If the parquet cannot be read we ignore it and rely on raw JSON cache
            pass

    return cached


def main():
    print("=" * 60)
    print("       PHASE 3.A: APPLY REFACTORINGMINER")
    print("=" * 60)

    start = time.time()
    try:
        rm = RefactoringMinerWrapper()
        if not rm.is_available():
            print("RefactoringMiner not available. Build in tools/RefactoringMiner with './gradlew shadowJar'.")
            sys.exit(1)

        commits = load_java_commits()
        print(f"Total: {len(commits)} file changes across {commits['sha'].nunique()} commits")

        # Select unique commits for analysis
        local_repo = os.environ.get('REFMINER_LOCAL_REPO')
        if local_repo and os.path.exists(local_repo):
            unique_commits = commits.drop_duplicates('sha')
        else:
            base = commits[commits['html_url'].notna()] if 'html_url' in commits.columns else pd.DataFrame()
            if base.empty:
                print("No html_url found and no REFMINER_LOCAL_REPO set. Cannot run RM.")
                sys.exit(1)
            unique_commits = base.drop_duplicates('sha')
        cached_shas = _collect_cached_shas(rm)
        if cached_shas:
            candidate_cached = set(unique_commits['sha'].astype(str)) & cached_shas
            if candidate_cached:
                print(f"Skipping {len(candidate_cached)} commits already cached by RefactoringMiner")

        unique_commits = unique_commits[~unique_commits['sha'].astype(str).isin(cached_shas)]
        if unique_commits.empty:
            print("All candidate commits already have RefactoringMiner results. Nothing to do.")
            return

        max_commits = int(os.environ.get('REFMINER_MAX_COMMITS', 100000))
        unique_commits = unique_commits.head(max_commits)
        print(f"Analyzing {len(unique_commits)} unique commits (REFMINER_MAX_COMMITS={max_commits})…")

        ref_df = rm.analyze_commits_batch(unique_commits, max_commits=max_commits)
        if ref_df.empty:
            print("No new refactorings detected by RefactoringMiner.")
            return

        existing_results = pd.read_parquet(REFMINER_PARQUET) if REFMINER_PARQUET.exists() else pd.DataFrame()
        combined_results = pd.concat([existing_results, ref_df], ignore_index=True)
        if not combined_results.empty:
            combined_results = combined_results.drop_duplicates(
                subset=["commit_sha", "refactoring_type", "description"], keep="last"
            )

        metadata = commits[commits['sha'].isin(combined_results['commit_sha'].unique())]
        metadata = metadata.drop_duplicates('sha') if not metadata.empty else metadata

        analysis = rm.analyze_refminer_results(combined_results, metadata)
        rm.save_refminer_results(combined_results, analysis)

        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print("REFACTORINGMINER APPLICATION COMPLETED")
        print("=" * 60)
        print(f"Commits scanned: {unique_commits['sha'].nunique()}")
        if not ref_df.empty:
            print(f"Refactorings found: {len(ref_df)} across {ref_df['commit_sha'].nunique()} commits")
        print(f"Execution time: {elapsed:.1f}s")
        print("Outputs: data/analysis/refactoring_instances")

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
