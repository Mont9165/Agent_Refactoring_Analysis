#!/usr/bin/env python3
"""
Phase 4: Quality impact analysis (DesigniteJava + Readability)

Usage:
- Set env vars to point to tools/local repos as needed:
  - REPOS_BASE=/path/to/local/repos/root (owner/repo expected under this)
  - or REFMINER_LOCAL_REPO=/path/to/single/repo (analyze only this repo)
  - DESIGNITE_JAVA_PATH=/path/to/DesigniteJava.jar
  - READABILITY_TOOL_CMD='python readability.py --input {input} --output {output}'
    or READABILITY_JAR=/path/to/readability.jar

Runs analysis over a sample of refactoring commits and writes results under data/analysis/quality/.
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.phase3_code_quality.quality_analysis import analyze_quality


def main():
    commits_path = Path('data/analysis/refactoring_instances/commits_with_refactoring.parquet')
    if not commits_path.exists():
        print('ERROR: commits_with_refactoring.parquet not found. Run scripts/4_analyze_refactoring_instance_and_type.py first.')
        sys.exit(1)

    max_commits = int(os.environ.get('QUALITY_MAX_COMMITS', 25))
    df, summary = analyze_quality(commits_path, max_commits=max_commits)

    print('Quality analysis complete:')
    print(f"  Analyzed commits: {len(df)}")
    if summary:
        for k, v in summary.items():
            print(f"  {k}: {v:.3f}")
    print('Outputs: data/analysis/quality/')


if __name__ == '__main__':
    main()

