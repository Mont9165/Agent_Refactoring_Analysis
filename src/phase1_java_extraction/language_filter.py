"""
Repository language filtering for Java project detection
"""
import pandas as pd
from typing import List, Dict, Set
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader


class LanguageFilter:
    """Filters repositories based on programming language indicators"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.loader = HFDatasetLoader(config_path)
        self.config = self.loader.config
        self.filtering_config = self.config.get("filtering", {})
        
        # Java detection settings
        self.java_extensions = set(self.config["java_detection"]["java_file_extensions"])
        self.build_files = set(self.config["java_detection"]["build_file_patterns"])
        self.min_java_percentage = self.config["java_detection"]["min_java_percentage"]
        self.check_build_files = self.config["java_detection"]["check_build_files"]
        
        # Repository filters
        self.min_repo_stars = int(self.filtering_config.get("min_repo_stars", 0) or 0)
    
    def load_repository_data(self) -> pd.DataFrame:
        """Load repository metadata from parquet file"""
        return self.loader.load_parquet_table("all_repository")
    
    def filter_by_repository_language(self, repos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter repositories by language field
        
        Args:
            repos_df: Repository dataframe
            
        Returns:
            Filtered dataframe with Java repositories
        """
        # Check if 'language' column exists
        if 'language' not in repos_df.columns:
            print("Warning: 'language' column not found in repository data")
            return repos_df
        
        # Filter for Java repositories
        java_repos = repos_df[
            repos_df['language'].str.lower().str.contains('java', na=False) |
            repos_df['language'].str.lower().str.contains('kotlin', na=False)
        ].copy()
        
        pre_filter_count = len(java_repos)
        
        if self.min_repo_stars > 0:
            if 'stars' in java_repos.columns:
                stars_series = pd.to_numeric(java_repos['stars'], errors='coerce').fillna(0)
                mask = stars_series >= self.min_repo_stars
                removed = (~mask).sum()
                java_repos = java_repos[mask].copy()
                print(
                    f"Applied min_repo_stars filter (>= {self.min_repo_stars} stars): "
                    f"removed {removed:,} repositories"
                )
            else:
                print("Warning: 'stars' column not found; skipping min_repo_stars filter")
        
        print(f"Found {len(java_repos)} repositories with Java/Kotlin language tag (from {pre_filter_count:,})")
        return java_repos
    
    def analyze_repository_languages(self, repos_df: pd.DataFrame) -> Dict:
        """
        Analyze language distribution in repository dataset
        
        Args:
            repos_df: Repository dataframe
            
        Returns:
            Dictionary with language statistics
        """
        if 'language' not in repos_df.columns:
            return {"error": "No language column found"}
        
        # Get language counts
        lang_counts = repos_df['language'].value_counts()
        
        # Find Java-related languages
        java_related = lang_counts[
            lang_counts.index.str.lower().str.contains('java|kotlin', na=False)
        ]
        
        stats = {
            "total_repos": len(repos_df),
            "repos_with_language": repos_df['language'].notna().sum(),
            "top_10_languages": lang_counts.head(10).to_dict(),
            "java_related": java_related.to_dict(),
            "java_related_count": java_related.sum()
        }
        
        return stats
    
    def get_repository_names(self, repos_df: pd.DataFrame) -> List[str]:
        """
        Extract repository names/identifiers
        
        Args:
            repos_df: Repository dataframe
            
        Returns:
            List of repository identifiers
        """
        # Try common column names for repository identification
        possible_name_cols = ['full_name', 'name', 'repo_name', 'repository_name', 'url']
        
        for col in possible_name_cols:
            if col in repos_df.columns:
                return repos_df[col].tolist()
        
        # If no name column found, use index
        print("Warning: No repository name column found, using index")
        return repos_df.index.tolist()
    
    def save_filtered_repositories(self, repos_df: pd.DataFrame, output_path: str = None):
        """
        Save filtered Java repositories to parquet file
        
        Args:
            repos_df: Filtered repository dataframe
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = "data/filtered/java_repositories/repos_metadata.parquet"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        repos_df.to_parquet(output_path, index=False)
        print(f"Saved {len(repos_df)} Java repositories to {output_path}")
        
        # Also save as CSV for human readability
        csv_path = output_path.replace('.parquet', '.csv')
        repos_df.to_csv(csv_path, index=False)
        print(f"Also saved as CSV: {csv_path}")
        
        return output_path
