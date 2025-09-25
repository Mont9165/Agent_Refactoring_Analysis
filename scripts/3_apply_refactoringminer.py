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
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase3_refactoring_analysis.refminer_wrapper import RefactoringMinerWrapper


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

        max_commits = int(os.environ.get('REFMINER_MAX_COMMITS', 1000))
        unique_commits = unique_commits.head(max_commits)
        print(f"Analyzing {len(unique_commits)} unique commits (REFMINER_MAX_COMMITS={max_commits})…")

        ref_df = rm.analyze_commits_batch(unique_commits, max_commits=max_commits)
        if ref_df.empty:
            print("No refactorings detected by RefactoringMiner.")
        else:
            analysis = rm.analyze_refminer_results(ref_df, unique_commits)
            rm.save_refminer_results(ref_df, analysis)

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

