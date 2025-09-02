"""
Data merger for joining repository, PR, and commit data
"""
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader


class DataMerger:
    """Handles merging of different dataset tables for Java project analysis"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.loader = HFDatasetLoader(config_path)
        self.config = self.loader.config
    
    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load all relevant tables for analysis
        
        Returns:
            Dictionary with all loaded tables
        """
        print("Loading all dataset tables...")
        
        tables = {}
        table_names = [
            "all_repository", "all_pull_request", "all_user",
            "pr_commits", "pr_commit_details", "pull_request",
            "pr_comments", "pr_reviews"
        ]
        
        for table_name in table_names:
            try:
                df = self.loader.load_parquet_table(table_name)
                tables[table_name] = df
                print(f"Loaded {table_name}: {len(df)} rows")
            except Exception as e:
                print(f"Warning: Could not load {table_name}: {e}")
                tables[table_name] = pd.DataFrame()
        
        return tables
    
    def merge_pr_with_commits(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge pull request data with commit information
        
        Args:
            tables: Dictionary with all loaded tables
            
        Returns:
            Merged DataFrame with PR and commit data
        """
        print("\nMerging PR data with commits...")
        
        # Start with pull request data (smaller subset)
        if not tables["pull_request"].empty:
            pr_df = tables["pull_request"].copy()
        else:
            print("Using all_pull_request as fallback")
            pr_df = tables["all_pull_request"].copy()
        
        print(f"Starting with {len(pr_df)} pull requests")
        
        # Merge with commit data
        if not tables["pr_commits"].empty:
            commits_df = tables["pr_commits"].copy()
            
            # Identify merge keys
            pr_id_col = self._find_common_column(pr_df, commits_df, 
                                               ['pull_request_id', 'pr_id', 'id'])
            
            if pr_id_col:
                merged_df = pr_df.merge(
                    commits_df,
                    on=pr_id_col,
                    how='inner',
                    suffixes=('_pr', '_commit')
                )
                print(f"After merging with commits: {len(merged_df)} records")
            else:
                print("Warning: Could not find common column for PR-commit merge")
                merged_df = pr_df
        else:
            merged_df = pr_df
        
        return merged_df
    
    def merge_with_commit_details(self, pr_commits_df: pd.DataFrame, 
                                tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge PR-commit data with detailed commit information
        
        Args:
            pr_commits_df: DataFrame with PR and commit data
            tables: Dictionary with all loaded tables
            
        Returns:
            DataFrame with detailed commit information
        """
        print("\nMerging with commit details...")
        
        if tables["pr_commit_details"].empty:
            print("No commit details available")
            return pr_commits_df
        
        commit_details_df = tables["pr_commit_details"].copy()
        
        # Find common columns for merging
        commit_col = self._find_common_column(pr_commits_df, commit_details_df,
                                            ['commit_sha', 'sha', 'commit_id'])
        
        if commit_col:
            # Merge on commit identifier
            merged_df = pr_commits_df.merge(
                commit_details_df,
                on=commit_col,
                how='inner',
                suffixes=('', '_detail')
            )
            print(f"After merging with commit details: {len(merged_df)} records")
        else:
            print("Warning: Could not find common column for commit details merge")
            merged_df = pr_commits_df
        
        return merged_df
    
    def merge_with_repository_data(self, pr_data_df: pd.DataFrame,
                                 tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge PR data with repository information
        
        Args:
            pr_data_df: DataFrame with PR data
            tables: Dictionary with all loaded tables
            
        Returns:
            DataFrame with repository information added
        """
        print("\nMerging with repository data...")
        
        # Use all_repository for comprehensive repository info
        if tables["all_repository"].empty:
            print("No repository data available")
            return pr_data_df
        
        repo_df = tables["all_repository"].copy()
        
        # Find common columns for repository merge
        repo_col = self._find_common_column(pr_data_df, repo_df,
                                          ['repository_id', 'repo_id', 'repository_full_name', 'full_name'])
        
        if repo_col:
            merged_df = pr_data_df.merge(
                repo_df,
                on=repo_col,
                how='left',  # Keep all PRs even if repo info missing
                suffixes=('', '_repo')
            )
            print(f"After merging with repositories: {len(merged_df)} records")
        else:
            print("Warning: Could not find common column for repository merge")
            merged_df = pr_data_df
        
        return merged_df
    
    def merge_with_user_data(self, data_df: pd.DataFrame,
                           tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge data with user information
        
        Args:
            data_df: DataFrame with existing data
            tables: Dictionary with all loaded tables
            
        Returns:
            DataFrame with user information added
        """
        print("\nMerging with user data...")
        
        if tables["all_user"].empty:
            print("No user data available")
            return data_df
        
        user_df = tables["all_user"].copy()
        
        # Find common columns for user merge
        user_col = self._find_common_column(data_df, user_df,
                                          ['user_id', 'author_id', 'creator_id'])
        
        if user_col:
            merged_df = data_df.merge(
                user_df,
                on=user_col,
                how='left',
                suffixes=('', '_user')
            )
            print(f"After merging with users: {len(merged_df)} records")
        else:
            print("Warning: Could not find common column for user merge")
            merged_df = data_df
        
        return merged_df
    
    def create_comprehensive_dataset(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create comprehensive dataset by merging all relevant tables
        
        Args:
            tables: Dictionary with all loaded tables
            
        Returns:
            Comprehensive merged DataFrame
        """
        print("=== Creating Comprehensive Dataset ===")
        
        # Step 1: Merge PRs with commits
        pr_commits_df = self.merge_pr_with_commits(tables)
        
        # Step 2: Add commit details
        pr_commit_details_df = self.merge_with_commit_details(pr_commits_df, tables)
        
        # Step 3: Add repository information
        with_repo_df = self.merge_with_repository_data(pr_commit_details_df, tables)
        
        # Step 4: Add user information
        final_df = self.merge_with_user_data(with_repo_df, tables)
        
        print(f"\n=== Final dataset: {len(final_df)} records ===")
        print(f"Columns: {len(final_df.columns)}")
        
        return final_df
    
    def _find_common_column(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                          candidate_cols: List[str]) -> Optional[str]:
        """
        Find common column between two DataFrames
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            candidate_cols: List of candidate column names
            
        Returns:
            Name of common column or None if not found
        """
        df1_cols = set(df1.columns)
        df2_cols = set(df2.columns)
        
        for col in candidate_cols:
            if col in df1_cols and col in df2_cols:
                return col
        
        # If no exact match, try to find similar columns
        common_cols = df1_cols.intersection(df2_cols)
        if common_cols:
            return list(common_cols)[0]
        
        return None
    
    def save_merged_dataset(self, merged_df: pd.DataFrame, output_path: str = None):
        """
        Save merged dataset
        
        Args:
            merged_df: Merged DataFrame
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = "data/filtered/java_repositories/merged_dataset.parquet"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        merged_df.to_parquet(output_path, index=False)
        print(f"Saved merged dataset to {output_path}")
        
        # Save column information
        col_info_path = output_path.replace('.parquet', '_columns.txt')
        with open(col_info_path, 'w') as f:
            f.write("Dataset Columns:\n")
            f.write("================\n\n")
            for i, col in enumerate(merged_df.columns):
                f.write(f"{i+1:3d}. {col}\n")
            
            f.write(f"\nTotal columns: {len(merged_df.columns)}\n")
            f.write(f"Total rows: {len(merged_df)}\n")
        
        print(f"Column information saved to {col_info_path}")
        
        return output_path
    
    def get_merge_statistics(self, tables: Dict[str, pd.DataFrame], 
                           final_df: pd.DataFrame) -> Dict:
        """
        Generate statistics about the merge process
        
        Args:
            tables: Original tables
            final_df: Final merged DataFrame
            
        Returns:
            Dictionary with merge statistics
        """
        stats = {
            'original_tables': {name: len(df) for name, df in tables.items()},
            'final_dataset_size': len(final_df),
            'final_dataset_columns': len(final_df.columns),
            'merge_efficiency': {}
        }
        
        # Calculate merge efficiency
        for table_name, df in tables.items():
            if not df.empty:
                stats['merge_efficiency'][table_name] = len(final_df) / len(df)
        
        return stats