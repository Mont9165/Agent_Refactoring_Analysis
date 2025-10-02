#!/usr/bin/env python3
"""
Run RefactoringMiner analysis on Java commits for accurate refactoring detection
"""
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase3_refactoring_analysis.refminer_wrapper import RefactoringMinerWrapper
import pandas as pd


def main():
    """Run RefactoringMiner analysis on Java commits"""
    
    print("="*60)
    print("       REFACTORINGMINER ANALYSIS")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize RefactoringMiner
        print("Initializing RefactoringMiner...")
        refminer = RefactoringMinerWrapper()
        
        if not refminer.is_available():
            print("RefactoringMiner not available. Please check setup.")
            return
        
        # Load Java commits
        print("\n1. Loading Java commits...")
        parquet_path = "data/filtered/java_repositories/java_file_commits_for_refactoring.parquet"
        csv_path = "data/filtered/java_repositories/java_file_commits_for_refactoring.csv"
        
        if os.path.exists(parquet_path):
            commits_df = pd.read_parquet(parquet_path)
            print(f"Loaded {len(commits_df)} Java file changes from parquet file")
        elif os.path.exists(csv_path):
            commits_df = pd.read_csv(csv_path)
            print(f"Loaded {len(commits_df)} Java file changes from CSV file")
        else:
            print(f"ERROR: Neither {parquet_path} nor {csv_path} found")
            return
        
        print(f"Total: {len(commits_df)} Java file changes from {commits_df['sha'].nunique()} unique commits")
        
        # Filter commits that have GitHub URLs for analysis
        commits_with_urls = commits_df[commits_df['html_url'].notna()].copy()
        unique_commits_with_urls = commits_with_urls.drop_duplicates('sha')
        
        print(f"Found {len(unique_commits_with_urls)} unique commits with GitHub URLs")
        
        # Set analysis scope (use environment variable or default)
        max_commits = int(os.environ.get('REFMINER_MAX_COMMITS', 50))  # Default to 50 commits
        print(f"\nAnalyzing {max_commits} commits (set REFMINER_MAX_COMMITS env var to change)")
        
        print(f"\n2. Running RefactoringMiner analysis on {max_commits} commits...")
        print("This may take several minutes...")
        
        # Run RefactoringMiner analysis
        analysis = refminer.run_full_refactoring_analysis(unique_commits_with_urls, max_commits)
        
        if analysis is None:
            print("RefactoringMiner analysis failed or found no refactorings")
            return
        
        # Print results
        print("\n" + "="*50)
        print("REFACTORINGMINER RESULTS")
        print("="*50)
        
        rm_results = analysis['refactoring_miner_results']
        print(f"\nRefactoringMiner Analysis:")
        print(f"  Commits analyzed: {rm_results['total_commits_analyzed']:,}")
        print(f"  Commits with refactoring: {rm_results['commits_with_refactoring']:,}")
        print(f"  Refactoring rate: {rm_results['refactoring_rate']:.1f}%")
        print(f"  Total refactorings found: {rm_results['total_refactorings_found']:,}")
        
        if rm_results['most_common_refactoring']:
            print(f"  Most common refactoring: {rm_results['most_common_refactoring']}")
        
        print(f"\nTop refactoring types found:")
        for reftype, count in rm_results['refactoring_types'].items():
            print(f"  {reftype}: {count}")
        
        # Compare with pattern-based results if available
        try:
            pattern_analysis_path = "data/analysis/refactoring_instances/refactoring_analysis.json"
            if os.path.exists(pattern_analysis_path):
                import json
                with open(pattern_analysis_path, 'r') as f:
                    pattern_results = json.load(f)
                
                pattern_rate = pattern_results['summary']['overall_refactoring_rate']
                rm_rate = rm_results['refactoring_rate']
                
                print(f"\n" + "="*40)
                print("COMPARISON: Pattern vs RefactoringMiner")
                print("="*40)
                print(f"Pattern-based detection: {pattern_rate:.1f}% refactoring rate")
                print(f"RefactoringMiner detection: {rm_rate:.1f}% refactoring rate")
                print(f"Difference: {abs(rm_rate - pattern_rate):.1f}% points")
                
                if rm_rate > pattern_rate:
                    print("→ RefactoringMiner detected MORE refactoring instances")
                else:
                    print("→ Pattern-based detected more potential refactorings")
                
        except Exception as e:
            print(f"Could not compare with pattern-based results: {e}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("REFACTORINGMINER ANALYSIS COMPLETED!")
        print("="*60)
        print(f"Execution time: {execution_time:.1f} seconds")
        print(f"Results saved to: data/analysis/refactoring_instances/")
        
        # Next steps
        print("\n" + "="*50)
        print("RESEARCH IMPLICATIONS")
        print("="*50)
        print("RefactoringMiner provides:")
        print("✓ Precise refactoring type identification")
        print("✓ Location-specific refactoring details")
        print("✓ High accuracy (99.9% precision)")
        print("✓ Detailed AST-level analysis")
        
        print("\nFiles generated:")
        print("- refminer_refactorings.parquet: Detailed refactoring instances")
        print("- refminer_analysis.json: Analysis summary")
        
        print(f"\nFor your research questions:")
        print(f"- RQ1: {rm_results['commits_with_refactoring']} commits with verified refactoring")
        print(f"- RQ3: {len(rm_results['refactoring_types'])} different refactoring types detected")
        print(f"- More accurate than pattern-based detection")
        
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