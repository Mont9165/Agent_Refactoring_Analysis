"""
Simple Java project filter - just check for .java files in PR commits
"""
import pandas as pd
from typing import Dict, List
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader


class SimpleJavaFilter:
    """Simple and efficient Java project filtering based on .java file presence"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.loader = HFDatasetLoader(config_path)
        self.config = self.loader.config
        self.filtering_config = self.config.get("filtering", {})
        
        # Java file extensions from config
        self.java_extensions = self.config["java_detection"]["java_file_extensions"]
        print(f"Looking for files with extensions: {self.java_extensions}")
        
        # Repository filters
        self.min_repo_stars = int(self.filtering_config.get("min_repo_stars", 0) or 0)
        self._last_filter_stats = {}
    
    def filter_java_prs(self) -> pd.DataFrame:
        """
        Filter PRs that contain Java files
        
        Returns:
            DataFrame with Java PRs and their statistics
        """
        print("Loading commit details...")
        commit_details_df = self.loader.load_parquet_table("pr_commit_details")
        print(f"Loaded {len(commit_details_df)} commit details")
        
        # Check available columns
        print(f"Available columns: {list(commit_details_df.columns)}")
        
        # Find the correct column name for pull request ID
        pr_id_col = None
        possible_pr_cols = ['pull_request_id', 'pr_id', 'pull_request_number', 'pr_number']
        for col in possible_pr_cols:
            if col in commit_details_df.columns:
                pr_id_col = col
                break
        
        if pr_id_col is None:
            print("ERROR: Could not find pull request ID column!")
            print(f"Available columns: {list(commit_details_df.columns)}")
            return pd.DataFrame()
        
        print(f"Using PR ID column: {pr_id_col}")
        
        # Filter for Java files
        print("Filtering for Java files...")
        java_commits = commit_details_df[
            commit_details_df['filename'].str.endswith(tuple(self.java_extensions), na=False)
        ].copy()
        
        print(f"Found {len(java_commits)} commits with Java files")
        
        if java_commits.empty:
            print("No Java files found!")
            return pd.DataFrame()
        
        # Get PR statistics
        print("Calculating PR statistics...")
        pr_stats = self._calculate_pr_stats(java_commits, commit_details_df, pr_id_col)
        
        # Add PR status information
        print("Adding PR status information...")
        pr_stats_with_status = self._add_pr_status(pr_stats, pr_id_col)
        
        # Report repository filter summary
        if self._last_filter_stats.get("min_repo_stars") is not None:
            threshold = self._last_filter_stats["min_repo_stars"]
            kept = self._last_filter_stats.get("final_prs", len(pr_stats_with_status))
            if self._last_filter_stats.get("filter_skipped"):
                print(
                    f"Repository star filter (>= {threshold} stars) was skipped because repo_stars data was unavailable"
                )
            else:
                removed = self._last_filter_stats.get("removed_by_stars", 0) or 0
                if removed:
                    print(f"Repository star filter (>= {threshold} stars) removed {removed:,} PRs; {kept:,} remain")
                else:
                    print(f"Repository star filter (>= {threshold} stars) retained all {kept:,} PRs")
        elif self.min_repo_stars:
            print(
                f"Warning: Repository star filter configured (>= {self.min_repo_stars} stars) "
                "but could not be applied due to missing repository data"
            )
        
        return pr_stats_with_status
    
    def _calculate_pr_stats(self, java_commits: pd.DataFrame, all_commits: pd.DataFrame, pr_id_col: str) -> pd.DataFrame:
        """Calculate statistics for each PR with Java files"""
        
        # Find commit SHA column
        commit_col = None
        possible_commit_cols = ['commit_sha', 'sha', 'commit_id']
        for col in possible_commit_cols:
            if col in java_commits.columns:
                commit_col = col
                break
        
        if commit_col is None:
            print("Warning: Could not find commit SHA column, using filename for commit count")
            commit_col = 'filename'  # fallback
        
        # Get PR-level statistics
        java_pr_stats = java_commits.groupby(pr_id_col).agg({
            'filename': 'count',  # Java files count
            commit_col: 'nunique'  # Unique commits with Java files
        }).rename(columns={
            'filename': 'java_files_count',
            commit_col: 'java_commits_count'
        }).reset_index()
        
        # Get total files per PR for percentage calculation
        all_pr_stats = all_commits.groupby(pr_id_col).agg({
            'filename': 'count',
            commit_col: 'nunique'
        }).rename(columns={
            'filename': 'total_files_count',
            commit_col: 'total_commits_count'
        }).reset_index()
        
        # Merge to get complete stats
        pr_stats = java_pr_stats.merge(all_pr_stats, on=pr_id_col, how='left')
        
        # Calculate percentages
        pr_stats['java_files_percentage'] = (pr_stats['java_files_count'] / pr_stats['total_files_count'] * 100).round(2)
        pr_stats['java_commits_percentage'] = (pr_stats['java_commits_count'] / pr_stats['total_commits_count'] * 100).round(2)
        
        return pr_stats
    
    def _add_pr_status(self, pr_stats: pd.DataFrame, pr_id_col: str) -> pd.DataFrame:
        """Add PR status information (open/closed/merged)"""
        
        # Load PR data
        all_prs = self.loader.load_parquet_table("all_pull_request")
        
        # Find the matching ID column in all_prs
        pr_match_col = None
        possible_cols = ['id', 'number', 'pr_id']
        for col in possible_cols:
            if col in all_prs.columns:
                pr_match_col = col
                break
        
        if pr_match_col is None:
            print("Warning: Could not find matching PR ID column in all_pull_request")
            return pr_stats
        
        # Select status columns, html_url, and agent info
        status_cols = [pr_match_col, 'state', 'created_at', 'closed_at', 'merged_at', 'html_url', 'title', 'agent', 'user', 'repo_id']
        available_status_cols = [col for col in status_cols if col in all_prs.columns]
        
        pr_status_df = all_prs[available_status_cols].copy()
        
        # Merge with pr_stats
        merged_df = pr_stats.merge(
            pr_status_df,
            left_on=pr_id_col,
            right_on=pr_match_col,
            how='left'
        )
        
        # Add derived status columns
        if 'state' in merged_df.columns:
            merged_df['is_closed'] = merged_df['state'] == 'closed'
            merged_df['is_open'] = merged_df['state'] == 'open'
        
        if 'merged_at' in merged_df.columns:
            merged_df['is_merged'] = merged_df['merged_at'].notna()
        
        # Clean up duplicate ID columns
        if pr_match_col != pr_id_col and pr_match_col in merged_df.columns:
            merged_df = merged_df.drop(columns=[pr_match_col])
        
        # Add repository information (stars, forks)
        merged_df_with_repo = self._add_repo_info(merged_df)
        
        return merged_df_with_repo
    
    def _add_repo_info(self, pr_stats: pd.DataFrame) -> pd.DataFrame:
        """Add repository information (stars, forks) from all_repository table"""
        
        # Check if repo_id is available
        if 'repo_id' not in pr_stats.columns:
            print("Warning: No repo_id column found, cannot add star/fork information")
            return pr_stats
        
        try:
            # Load repository data
            repo_df = self.loader.load_parquet_table("all_repository")
            
            # Select relevant repository columns
            repo_cols = ['id', 'forks', 'stars', 'full_name', 'language']
            available_repo_cols = [col for col in repo_cols if col in repo_df.columns]
            
            if 'id' not in available_repo_cols:
                print("Warning: No id column in repository table")
                return pr_stats
                
            repo_info = repo_df[available_repo_cols].copy()
            
            # Merge repository information
            merged_df = pr_stats.merge(
                repo_info,
                left_on='repo_id',
                right_on='id',
                how='left',
                suffixes=('', '_repo')
            )
            
            # Clean up duplicate id column
            if 'id' in merged_df.columns and 'id' != 'repo_id':
                merged_df = merged_df.drop(columns=['id'])
            
            # Add derived columns
            if 'stars' in merged_df.columns:
                merged_df['repo_stars'] = merged_df['stars'].fillna(0).astype(int)
            if 'forks' in merged_df.columns:  
                merged_df['repo_forks'] = merged_df['forks'].fillna(0).astype(int)
            if 'full_name' in merged_df.columns:
                merged_df['repo_name'] = merged_df['full_name']
            if 'language' in merged_df.columns:
                merged_df['repo_language'] = merged_df['language']
            
            print(f"Added repository information for {len(merged_df)} PRs")
            return self._apply_repository_filters(merged_df)
            
        except Exception as e:
            print(f"Warning: Could not load repository information: {e}")
            self._last_filter_stats = {
                "initial_prs": len(pr_stats),
                "final_prs": len(pr_stats),
                "min_repo_stars": None,
                "removed_by_stars": 0
            }
            return pr_stats

    def _apply_repository_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply repository-level filters such as minimum star threshold"""
        filter_stats = {
            "initial_prs": len(df),
            "final_prs": len(df),
            "min_repo_stars": None,
            "removed_by_stars": 0,
            "filter_skipped": False
        }
        
        # Apply minimum star filter if configured
        if self.min_repo_stars > 0:
            filter_stats["min_repo_stars"] = self.min_repo_stars
            
            if 'repo_stars' not in df.columns:
                print("Warning: repo_stars column missing; skipping star-based repository filter")
                filter_stats["filter_skipped"] = True
                self._last_filter_stats = filter_stats
                return df
            
            before_count = len(df)
            filtered_df = df[df['repo_stars'] >= self.min_repo_stars].copy()
            removed = before_count - len(filtered_df)
            
            filter_stats["removed_by_stars"] = removed
            filter_stats["final_prs"] = len(filtered_df)
            self._last_filter_stats = filter_stats
            return filtered_df
        
        self._last_filter_stats = filter_stats
        return df
    
    def get_summary_stats(self, pr_stats: pd.DataFrame) -> Dict:
        """Get summary statistics"""
        if pr_stats.empty:
            return {"error": "No Java PRs found"}
        
        # Load all PRs for comparison
        all_prs = self.loader.load_parquet_table("all_pull_request")
        
        summary = {
            "total_prs_in_dataset": len(all_prs),
            "java_prs_found": len(pr_stats),
            "java_pr_percentage": len(pr_stats) / len(all_prs) * 100,
            
            "java_files_stats": {
                "total_java_files": pr_stats['java_files_count'].sum(),
                "avg_java_files_per_pr": pr_stats['java_files_count'].mean(),
                "median_java_files_per_pr": pr_stats['java_files_count'].median(),
                "max_java_files_per_pr": pr_stats['java_files_count'].max()
            },
            
            "java_percentage_stats": {
                "avg_java_percentage": pr_stats['java_files_percentage'].mean(),
                "median_java_percentage": pr_stats['java_files_percentage'].median(),
                "prs_with_high_java_percentage": (pr_stats['java_files_percentage'] >= 60).sum(),
                "prs_with_100_percent_java": (pr_stats['java_files_percentage'] == 100).sum()
            }
        }
        
        # Add PR status statistics if available
        if 'state' in pr_stats.columns:
            summary["pr_status_stats"] = {
                "closed_prs": (pr_stats['state'] == 'closed').sum(),
                "open_prs": (pr_stats['state'] == 'open').sum(),
            }
            
            if 'is_merged' in pr_stats.columns:
                summary["pr_status_stats"]["merged_prs"] = pr_stats['is_merged'].sum()
                summary["pr_status_stats"]["closed_not_merged"] = (
                    (pr_stats['state'] == 'closed') & (~pr_stats['is_merged'])
                ).sum()
        
        # Add agent statistics if available
        if 'agent' in pr_stats.columns:
            agent_counts = pr_stats['agent'].value_counts()
            summary["agent_stats"] = {
                "total_agentic_prs": pr_stats['agent'].notna().sum(),
                "total_human_prs": pr_stats['agent'].isna().sum(),
                "agent_breakdown": agent_counts.head(10).to_dict()  # Top 10 agents
            }
        
        # Add repository statistics if available
        if 'repo_stars' in pr_stats.columns and 'repo_forks' in pr_stats.columns:
            summary["repository_stats"] = {
                "avg_stars": pr_stats['repo_stars'].mean(),
                "median_stars": pr_stats['repo_stars'].median(),
                "max_stars": pr_stats['repo_stars'].max(),
                "avg_forks": pr_stats['repo_forks'].mean(),
                "median_forks": pr_stats['repo_forks'].median(),
                "max_forks": pr_stats['repo_forks'].max(),
                "unique_repositories": pr_stats['repo_id'].nunique()
            }
            
            # Popular repositories (top 5 by stars)
            if 'repo_name' in pr_stats.columns:
                top_repos = pr_stats.nlargest(5, 'repo_stars')[['repo_name', 'repo_stars', 'repo_forks']].drop_duplicates('repo_name')
                summary["repository_stats"]["top_repositories_by_stars"] = {
                    row['repo_name']: {"stars": int(row['repo_stars']), "forks": int(row['repo_forks'])}
                    for _, row in top_repos.iterrows()
                }
        
        if self._last_filter_stats:
            summary["filtering_stats"] = self._last_filter_stats
        
        return summary
    
    def save_java_prs(self, pr_stats: pd.DataFrame, summary: Dict):
        """Save Java PR results"""
        output_dir = Path("data/filtered/java_repositories")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        whitelist_path = self._save_repo_whitelist(pr_stats, output_dir)
        if whitelist_path:
            summary.setdefault("filtering_stats", self._last_filter_stats)
            summary["filtering_stats"]["repo_whitelist_path"] = str(whitelist_path)
        
        # Save main results
        pr_stats.to_parquet(output_dir / "simple_java_prs.parquet", index=False)
        pr_stats.to_csv(output_dir / "simple_java_prs.csv", index=False)
        
        # Save summary (convert numpy types to Python types for JSON serialization)
        import json
        def convert_numpy_types(obj):
            """Convert numpy types to Python types for JSON serialization"""
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        summary_json = convert_numpy_types(summary)
        with open(output_dir / "simple_java_summary.json", 'w') as f:
            json.dump(summary_json, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")
        print(f"- simple_java_prs.parquet: {len(pr_stats)} Java PRs")
        print(f"- simple_java_summary.json: Summary statistics")
        
        return output_dir / "simple_java_prs.parquet"

    def _save_repo_whitelist(self, pr_stats: pd.DataFrame, output_dir: Path) -> Path | None:
        """Persist unique repositories that passed filtering criteria"""
        if 'repo_id' not in pr_stats.columns:
            print("Warning: Cannot create repository whitelist because repo_id column is missing")
            return None
        
        repo_columns = ['repo_id']
        optional_columns = ['repo_name', 'repo_stars', 'repo_forks', 'repo_language']
        repo_columns.extend([col for col in optional_columns if col in pr_stats.columns])
        
        repo_df = pr_stats[repo_columns].drop_duplicates('repo_id').sort_values('repo_id').reset_index(drop=True)
        if repo_df.empty:
            print("Warning: Repository whitelist would be empty; skipping save")
            return None
        
        whitelist_path = output_dir / "high_star_repositories.csv"
        repo_df.to_csv(whitelist_path, index=False)
        repo_df.to_parquet(whitelist_path.with_suffix(".parquet"), index=False)
        whitelist_count = len(repo_df)
        print(f"Saved repository whitelist with {whitelist_count} entries to {whitelist_path}")
        self._last_filter_stats["whitelist_entries"] = whitelist_count
        return whitelist_path
    
    def get_java_pr_ids(self) -> List[str]:
        """Get list of Java PR IDs for further analysis"""
        pr_stats = self.filter_java_prs()
        if 'pull_request_id' in pr_stats.columns:
            return pr_stats['pull_request_id'].tolist()
        else:
            # Use the first column (should be the PR ID column)
            return pr_stats.iloc[:, 0].tolist()
