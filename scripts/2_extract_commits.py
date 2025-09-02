#!/usr/bin/env python3
"""
Extract commits for Java PRs - prepare data for refactoring analysis
"""
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_java_extraction.commit_extractor import CommitExtractor


def main():
    """Extract and analyze commits for Java PRs"""
    
    print("="*60)
    print("       COMMIT EXTRACTION FOR JAVA PRs")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize extractor
        print("Initializing commit extractor...")
        extractor = CommitExtractor()
        
        # Load Java PRs from previous step
        print("\n1. Loading Java PRs from previous analysis...")
        java_prs = extractor.load_java_prs()
        
        # Extract commits for these PRs
        print("\n2. Extracting commits for Java PRs...")
        java_commits = extractor.extract_commits_for_java_prs(java_prs)
        
        # Enhance with PR metadata
        print("\n3. Enhancing commits with PR metadata...")
        enhanced_commits = extractor.enhance_with_pr_metadata(java_commits, java_prs)
        
        # Analyze commit patterns
        print("\n4. Analyzing commit patterns...")
        analysis = extractor.analyze_commit_patterns(enhanced_commits)
        
        # Print analysis results
        print("\n" + "="*50)
        print("COMMIT ANALYSIS RESULTS")
        print("="*50)
        
        print(f"Total file changes: {analysis['total_file_changes']:,}")
        print(f"Unique commits: {analysis['unique_commits']:,}")
        print(f"Java file changes: {analysis['java_file_changes']:,}")
        print(f"Unique PRs: {analysis['unique_prs']:,}")
        print(f"Unique authors: {analysis['unique_authors']:,}")
        
        if 'unique_agentic_commits' in analysis:
            print(f"\nAgent involvement:")
            print(f"  Agentic file changes: {analysis['agentic_file_changes']:,}")
            print(f"  Unique agentic commits: {analysis['unique_agentic_commits']:,}")
            print(f"  Agentic percentage: {analysis['agentic_percentage']:.1f}%")
            
            if 'top_agents_in_commits' in analysis:
                print(f"  Top agents:")
                for agent, count in analysis['top_agents_in_commits'].items():
                    print(f"    {agent}: {count} commits")
        
        print(f"\nFile change statistics:")
        print(f"  Avg additions: {analysis['file_stats']['avg_additions']:.1f}")
        print(f"  Avg deletions: {analysis['file_stats']['avg_deletions']:.1f}")
        print(f"  Avg changes: {analysis['file_stats']['avg_changes']:.1f}")
        
        if 'java_file_stats' in analysis:
            print(f"\nJava file statistics:")
            print(f"  Java file changes: {analysis['java_file_stats']['java_file_changes']:,}")
            print(f"  Unique Java commits: {analysis['java_file_stats']['unique_java_commits']:,}")
            print(f"  Avg Java additions: {analysis['java_file_stats']['avg_java_additions']:.1f}")
            print(f"  Avg Java deletions: {analysis['java_file_stats']['avg_java_deletions']:.1f}")
            print(f"  Avg Java changes: {analysis['java_file_stats']['avg_java_changes']:.1f}")
        
        print(f"\nPotential refactoring indicators:")
        refactoring = analysis['potential_refactoring_commits']
        print(f"  'refactor' in message: {refactoring.get('refactor_in_message', 0)} commits")
        print(f"  'cleanup' in message: {refactoring.get('cleanup_in_message', 0)} commits")
        print(f"  'rename' in message: {refactoring.get('rename_in_message', 0)} commits")
        print(f"  'move' in message: {refactoring.get('move_in_message', 0)} commits")
        print(f"  'extract' in message: {refactoring.get('extract_in_message', 0)} commits")
        print(f"  Total potential refactoring: {refactoring.get('total_potential_refactoring', 0)} commits")
        print(f"  Percentage: {refactoring.get('percentage_potential_refactoring', 0):.1f}%")
        
        # Save results
        print("\n5. Saving commit data and analysis...")
        output_paths = extractor.save_commit_data(enhanced_commits, analysis)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("COMMIT EXTRACTION COMPLETED!")
        print("="*60)
        print(f"Execution time: {execution_time:.1f} seconds")
        print(f"\nOutput files:")
        print(f"  - PR commits (no merges): {output_paths['pr_commits_no_merges']}")
        print(f"  - Java commits for refactoring: {output_paths['java_commits']}")
        print(f"  - Analysis: {output_paths['analysis']}")
        
        print(f"\nNext steps:")
        print(f"1. Run RefactoringMiner on Java commits")
        print(f"2. Analyze refactoring patterns")
        print(f"3. Answer research questions")
        
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