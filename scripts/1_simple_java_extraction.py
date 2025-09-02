#!/usr/bin/env python3
"""
Simple Java project extraction - just filter for PRs with .java files
"""
import sys
import os
import time
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_java_extraction.simple_java_filter import SimpleJavaFilter


def main():
    """Run simple Java project extraction"""
    
    print("="*60)
    print("       SIMPLE JAVA PROJECT EXTRACTION")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize filter
        print("Initializing Java filter...")
        java_filter = SimpleJavaFilter()
        
        # Filter for Java PRs
        print("\nFiltering PRs with Java files...")
        pr_stats = java_filter.filter_java_prs()
        
        if pr_stats.empty:
            print("No Java PRs found!")
            return
        
        # Get summary statistics
        print("\nGenerating summary statistics...")
        summary = java_filter.get_summary_stats(pr_stats)
        
        # Print results
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        print(f"Total PRs in dataset: {summary['total_prs_in_dataset']:,}")
        print(f"PRs with Java files: {summary['java_prs_found']:,}")
        print(f"Java PR percentage: {summary['java_pr_percentage']:.2f}%")
        
        print(f"\nJava files statistics:")
        print(f"  Total Java files: {summary['java_files_stats']['total_java_files']:,}")
        print(f"  Average Java files per PR: {summary['java_files_stats']['avg_java_files_per_pr']:.1f}")
        print(f"  Median Java files per PR: {summary['java_files_stats']['median_java_files_per_pr']:.1f}")
        print(f"  Max Java files in single PR: {summary['java_files_stats']['max_java_files_per_pr']:,}")
        
        print(f"\nJava percentage statistics:")
        print(f"  Average Java percentage per PR: {summary['java_percentage_stats']['avg_java_percentage']:.1f}%")
        print(f"  Median Java percentage per PR: {summary['java_percentage_stats']['median_java_percentage']:.1f}%")
        print(f"  PRs with ≥60% Java files: {summary['java_percentage_stats']['prs_with_high_java_percentage']:,}")
        print(f"  PRs with 100% Java files: {summary['java_percentage_stats']['prs_with_100_percent_java']:,}")
        
        # Print PR status statistics if available
        if 'pr_status_stats' in summary:
            print(f"\nPR status statistics:")
            print(f"  Closed PRs: {summary['pr_status_stats']['closed_prs']:,}")
            print(f"  Open PRs: {summary['pr_status_stats']['open_prs']:,}")
            if 'merged_prs' in summary['pr_status_stats']:
                print(f"  Merged PRs: {summary['pr_status_stats']['merged_prs']:,}")
                print(f"  Closed but not merged: {summary['pr_status_stats']['closed_not_merged']:,}")
        
        # Print agent statistics if available
        if 'agent_stats' in summary:
            print(f"\nAgent statistics:")
            print(f"  Agentic PRs (with AI involvement): {summary['agent_stats']['total_agentic_prs']:,}")
            print(f"  Human PRs (no AI agent): {summary['agent_stats']['total_human_prs']:,}")
            agentic_percentage = summary['agent_stats']['total_agentic_prs'] / summary['java_prs_found'] * 100
            print(f"  Agentic percentage: {agentic_percentage:.1f}%")
            
            if summary['agent_stats']['agent_breakdown']:
                print(f"  Top agents:")
                for agent, count in list(summary['agent_stats']['agent_breakdown'].items())[:5]:
                    print(f"    {agent}: {count} PRs")
        
        # Print repository statistics if available
        if 'repository_stats' in summary:
            print(f"\nRepository statistics:")
            repo_stats = summary['repository_stats']
            print(f"  Unique repositories: {repo_stats['unique_repositories']:,}")
            print(f"  Average stars per repo: {repo_stats['avg_stars']:.1f}")
            print(f"  Median stars per repo: {repo_stats['median_stars']:.1f}")
            print(f"  Max stars: {repo_stats['max_stars']:,}")
            print(f"  Average forks per repo: {repo_stats['avg_forks']:.1f}")
            print(f"  Median forks per repo: {repo_stats['median_forks']:.1f}")
            print(f"  Max forks: {repo_stats['max_forks']:,}")
            
            if 'top_repositories_by_stars' in repo_stats:
                print(f"  Top repositories by stars:")
                for repo_name, info in list(repo_stats['top_repositories_by_stars'].items())[:3]:
                    print(f"    {repo_name}: {info['stars']:,} stars, {info['forks']:,} forks")
        
        # Save results
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        output_path = java_filter.save_java_prs(pr_stats, summary)
        
        # Show top Java PRs
        print(f"\nTop 10 PRs by Java file count:")
        
        # Get the correct PR ID column name (first column)
        pr_id_col = pr_stats.columns[0]
        
        # Select columns to display (including html_url, agent, and repo info if available)
        display_cols = [pr_id_col, 'java_files_count', 'java_files_percentage']
        if 'html_url' in pr_stats.columns:
            display_cols.append('html_url')
        if 'title' in pr_stats.columns:
            display_cols.append('title')
        if 'agent' in pr_stats.columns:
            display_cols.append('agent')
        if 'repo_name' in pr_stats.columns:
            display_cols.extend(['repo_name', 'repo_stars', 'repo_forks'])
        
        top_prs = pr_stats.nlargest(10, 'java_files_count')[display_cols]
        for _, row in top_prs.iterrows():
            url_info = f" - {row['html_url']}" if 'html_url' in row else ""
            title_info = f" ({row['title'][:50]}...)" if 'title' in row and pd.notna(row['title']) else ""
            agent_info = f" [Agent: {row['agent']}]" if 'agent' in row and pd.notna(row['agent']) else " [Human]"
            repo_info = ""
            if 'repo_name' in row and pd.notna(row['repo_name']):
                stars = int(row['repo_stars']) if 'repo_stars' in row and pd.notna(row['repo_stars']) else 0
                forks = int(row['repo_forks']) if 'repo_forks' in row and pd.notna(row['repo_forks']) else 0
                repo_info = f" [{row['repo_name']}: ⭐{stars:,} 🍴{forks:,}]"
            print(f"  PR {row[pr_id_col]}: {row['java_files_count']} Java files ({row['java_files_percentage']:.1f}%){agent_info}{repo_info}{title_info}{url_info}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("SIMPLE JAVA EXTRACTION COMPLETED!")
        print("="*60)
        print(f"Execution time: {execution_time:.1f} seconds")
        print(f"Results: {output_path}")
        
        print(f"\nNext steps:")
        print(f"1. Use these {summary['java_prs_found']:,} Java PRs for agentic detection (Phase 2)")
        print(f"2. Load results with: pd.read_parquet('{output_path}')")
        
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