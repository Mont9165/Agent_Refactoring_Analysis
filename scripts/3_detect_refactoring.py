#!/usr/bin/env python3
"""
Detect refactoring in Java commits.

This script orchestrates Phase 3 by:
- Loading the Java commit dataset from Phase 2
- Using RefactoringMiner when available for AST-accurate detection
- Falling back to pattern-based detection otherwise
- Producing annotated outputs and analysis summaries under data/analysis/refactoring_instances/
"""
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.phase3_refactoring_analysis.refactoring_detector import RefactoringDetector


def _load_java_commits() -> pd.DataFrame:
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
    """Run RefactoringMiner on Java commits and save results"""

    print("=" * 60)
    print("       PHASE 3: REFACTORING DETECTION (DEPRECATED)")
    print("=" * 60)
    print("This entry point has been split into two steps:")
    print("  - scripts/3_apply_refactoringminer.py (run RefactoringMiner + save raw/parsed)")
    print("  - scripts/4_analyze_refactoring_instance_and_type.py (commit-level analysis)")
    print("Please use those scripts. Exiting.")
    return

    start_time = time.time()

    try:
        # Initialize detector (uses RefactoringMiner if available; falls back to patterns)
        detector = RefactoringDetector()

        # Load commits
        print("\n1) Loading Java commits…")
        commits_df = _load_java_commits()
        print(f"Total: {len(commits_df)} file changes across {commits_df['sha'].nunique()} commits")

        # Try RefactoringMiner first (with env-controlled cap), fallback internally to patterns
        max_commits = int(os.environ.get('REFMINER_MAX_COMMITS', 100000))
        print(f"\n2) Detecting refactoring (max {max_commits} unique commits via RefactoringMiner if available)…")
        commits_with_flags = detector.detect_refactoring_with_refminer(commits_df, max_commits=max_commits)

        # Analysis by agent and self-affirmation
        print("\n3) Running analysis…")
        analysis = detector.analyze_refactoring_by_agent(commits_with_flags)
        self_affirmation = detector.check_self_affirmation(commits_with_flags)

        # Persist annotated commits and analyses
        out_dir = detector.save_results(commits_with_flags, analysis, self_affirmation)

        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("REFACTORING DETECTION COMPLETED")
        print("=" * 60)
        print(f"Commits analyzed: {commits_with_flags['sha'].nunique()}")
        print(f"Commits with refactoring: {commits_with_flags[commits_with_flags['has_refactoring']]['sha'].nunique()}")
        print(f"Execution time: {elapsed:.1f}s")
        print(f"Outputs: {out_dir}")

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
