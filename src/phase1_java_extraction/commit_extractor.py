"""
Commit extractor for Java PRs - extracts detailed commit information for analysis
"""
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader


class CommitExtractor:
    """Extracts detailed commit information for Java PRs"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.loader = HFDatasetLoader(config_path)
        self.config = self.loader.config
    
    def load_java_prs(self, java_prs_path: str = None) -> pd.DataFrame:
        """Load the identified Java PRs"""
        if java_prs_path is None:
            java_prs_path = "data/filtered/java_repositories/simple_java_prs.parquet"
        
        java_prs = pd.read_parquet(java_prs_path)
        print(f"Loaded {len(java_prs)} Java PRs")
        return java_prs
    
    def extract_commits_for_java_prs(self, java_prs: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all commit details for Java PRs
        
        Args:
            java_prs: DataFrame with Java PR information
            
        Returns:
            DataFrame with detailed commit information
        """
        print("Loading commit details...")
        commit_details = self.loader.load_parquet_table("pr_commit_details")
        
        # Get PR ID column name from java_prs
        pr_id_col = java_prs.columns[0]  # First column should be PR ID
        java_pr_ids = set(java_prs[pr_id_col].tolist())
        
        print(f"Filtering commits for {len(java_pr_ids)} Java PRs...")
        
        # Filter commit details for Java PRs only
        java_commits = commit_details[
            commit_details['pr_id'].isin(java_pr_ids)
        ].copy()
        
        print(f"Found {len(java_commits)} commits for Java PRs")
        
        # Remove merge commits
        print("Filtering out merge commits...")
        
        # Method 1: Check commit message for merge patterns
        merge_patterns = r'(?i)(^merge\s|^merged?\s|merge\spull\srequest|merge\sbranch)'
        is_merge = java_commits['message'].str.contains(merge_patterns, na=False, regex=True)
        
        # Method 2: Also check for commits with 0 changes (often merge commits)
        no_changes = (java_commits['additions'] == 0) & (java_commits['deletions'] == 0)
        
        # Combine both conditions
        java_commits = java_commits[~(is_merge | no_changes)].copy()
        
        print(f"After removing merge commits: {len(java_commits)} commits")
        
        # Add Java file indicator
        java_commits['is_java_file'] = java_commits['filename'].str.endswith('.java', na=False)
        
        return java_commits
    
    def enhance_with_pr_metadata(self, java_commits: pd.DataFrame, java_prs: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance commit data with PR metadata (agent, status, etc.)
        
        Args:
            java_commits: DataFrame with commit details
            java_prs: DataFrame with PR metadata
            
        Returns:
            Enhanced commit DataFrame
        """
        print("Enhancing commits with PR metadata...")
        
        # Get PR ID column name
        pr_id_col = java_prs.columns[0]
        
        # Select metadata columns to merge
        metadata_cols = [pr_id_col]
        optional_cols = ['state', 'agent', 'html_url', 'title', 'is_merged', 'java_files_percentage']
        
        for col in optional_cols:
            if col in java_prs.columns:
                metadata_cols.append(col)
        
        pr_metadata = java_prs[metadata_cols].copy()
        
        # Merge with commits
        enhanced_commits = java_commits.merge(
            pr_metadata,
            left_on='pr_id',
            right_on=pr_id_col,
            how='left'
        )
        
        # Clean up duplicate ID column if needed
        if pr_id_col != 'pr_id' and pr_id_col in enhanced_commits.columns:
            enhanced_commits = enhanced_commits.drop(columns=[pr_id_col])
        
        print(f"Enhanced {len(enhanced_commits)} commits with PR metadata")
        return enhanced_commits
    
    def analyze_commit_patterns(self, enhanced_commits: pd.DataFrame) -> Dict:
        """
        Analyze commit patterns for refactoring detection
        
        Args:
            enhanced_commits: DataFrame with enhanced commit data
            
        Returns:
            Dictionary with commit analysis
        """
        print("Analyzing commit patterns...")
        
        analysis = {
            "total_file_changes": len(enhanced_commits),
            "unique_commits": enhanced_commits['sha'].nunique(),
            "java_file_changes": (enhanced_commits['is_java_file'] == True).sum(),
            "unique_prs": enhanced_commits['pr_id'].nunique(),
            "unique_authors": enhanced_commits['author'].nunique(),
        }
        
        # Agent analysis
        if 'agent' in enhanced_commits.columns:
            agentic_commits = enhanced_commits[enhanced_commits['agent'].notna()]
            analysis["agentic_file_changes"] = len(agentic_commits)
            analysis["unique_agentic_commits"] = agentic_commits['sha'].nunique()
            analysis["agentic_percentage"] = agentic_commits['sha'].nunique() / enhanced_commits['sha'].nunique() * 100
            
            if len(agentic_commits) > 0:
                analysis["top_agents_in_commits"] = agentic_commits['agent'].value_counts().head().to_dict()
        
        # File change analysis
        analysis["file_stats"] = {
            "avg_additions": enhanced_commits['additions'].mean(),
            "avg_deletions": enhanced_commits['deletions'].mean(),
            "avg_changes": enhanced_commits['changes'].mean(),
            "max_changes": enhanced_commits['changes'].max(),
        }
        
        # Java-specific analysis
        java_only_commits = enhanced_commits[enhanced_commits['is_java_file'] == True]
        if len(java_only_commits) > 0:
            analysis["java_file_stats"] = {
                "java_file_changes": len(java_only_commits),
                "unique_java_commits": java_only_commits['sha'].nunique(),
                "avg_java_additions": java_only_commits['additions'].mean(),
                "avg_java_deletions": java_only_commits['deletions'].mean(),
                "avg_java_changes": java_only_commits['changes'].mean(),
                "largest_java_change": java_only_commits['changes'].max(),
            }
        
        # Potential refactoring indicators
        analysis["potential_refactoring_commits"] = self._identify_potential_refactoring(enhanced_commits)
        
        return analysis
    
    def _identify_potential_refactoring(self, commits: pd.DataFrame) -> Dict:
        """Identify commits that might contain refactoring based on patterns"""
        
        # Filter for Java files only
        java_commits = commits[commits['is_java_file'] == True].copy()
        
        if java_commits.empty:
            return {"error": "No Java file commits found"}
        
        # Pattern-based detection
        refactoring_patterns = {
            "refactor_in_message": java_commits['message'].str.contains(
                r'\brefactor\b|\brefactoring\b', case=False, na=False
            ),
            "cleanup_in_message": java_commits['message'].str.contains(
                r'\bclean\s?up\b|\bcleanup\b', case=False, na=False  
            ),
            "rename_in_message": java_commits['message'].str.contains(
                r'\brename\b|\brenamed\b', case=False, na=False
            ),
            "move_in_message": java_commits['message'].str.contains(
                r'\bmove\b|\bmoved\b|\brelocate\b', case=False, na=False
            ),
            "extract_in_message": java_commits['message'].str.contains(
                r'\bextract\b|\bextracted\b', case=False, na=False
            ),
        }
        
        results = {}
        total_potential = None
        
        for pattern_name, pattern_match in refactoring_patterns.items():
            count = pattern_match.sum()
            results[pattern_name] = count
            
            if total_potential is None:
                total_potential = pattern_match
            else:
                total_potential = total_potential | pattern_match
        
        results["total_potential_refactoring"] = total_potential.sum()
        results["percentage_potential_refactoring"] = (total_potential.sum() / len(java_commits) * 100) if len(java_commits) > 0 else 0
        
        return results
    
    def save_commit_data(self, enhanced_commits: pd.DataFrame, analysis: Dict):
        """Save commit data and analysis results"""
        output_dir = Path("data/filtered/java_repositories")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save non-merge commit data (all files from Java PRs)
        enhanced_commits.to_parquet(output_dir / "java_pr_commits_no_merges.parquet", index=False)
        print(f"Saved non-merge commits to: {output_dir}/java_pr_commits_no_merges.parquet")
        
        # Save Java-only commits for refactoring analysis
        java_only = enhanced_commits[enhanced_commits['is_java_file'] == True]
        java_only.to_parquet(output_dir / "java_file_commits_for_refactoring.parquet", index=False)
        java_only.to_csv(output_dir / "java_file_commits_for_refactoring.csv", index=False)
        print(f"Saved Java-only commits to: {output_dir}/java_file_commits_for_refactoring.parquet ({len(java_only)} commits)")
        
        # Save analysis results
        import json
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        analysis_json = convert_numpy_types(analysis)
        with open(output_dir / "commit_analysis.json", 'w') as f:
            json.dump(analysis_json, f, indent=2)
        print(f"Saved commit analysis to: {output_dir}/commit_analysis.json")
        
        return {
            "pr_commits_no_merges": output_dir / "java_pr_commits_no_merges.parquet",
            "java_commits": output_dir / "java_file_commits_for_refactoring.parquet", 
            "analysis": output_dir / "commit_analysis.json"
        }