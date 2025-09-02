"""
File extension analysis from commit details to identify Java projects
"""
import pandas as pd
import re
from typing import Dict, Set, List, Tuple
from collections import Counter
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader


class FileExtensionChecker:
    """Analyzes commit details to determine Java project likelihood based on file extensions"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.loader = HFDatasetLoader(config_path)
        self.config = self.loader.config
        
        # Java detection settings
        self.java_extensions = set(self.config["java_detection"]["java_file_extensions"])
        self.build_files = set(self.config["java_detection"]["build_file_patterns"])
        self.min_java_percentage = self.config["java_detection"]["min_java_percentage"]
    
    def load_commit_details(self) -> pd.DataFrame:
        """Load commit details from parquet file"""
        return self.loader.load_parquet_table("pr_commit_details")
    
    def extract_file_paths_from_commit(self, commit_details_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract file paths from commit details
        
        Args:
            commit_details_df: Commit details dataframe
            
        Returns:
            DataFrame with extracted file information
        """
        results = []
        
        for idx, row in commit_details_df.iterrows():
            # Check if filename column exists
            if 'filename' in row and pd.notna(row['filename']):
                filename = row['filename']
                file_ext = Path(filename).suffix.lower()
                
                results.append({
                    'commit_sha': row.get('commit_sha', row.get('sha', 'unknown')),
                    'pull_request_id': row.get('pull_request_id', row.get('pr_id', 'unknown')),
                    'filename': filename,
                    'file_extension': file_ext,
                    'is_java_file': file_ext in self.java_extensions,
                    'is_build_file': any(build_file in filename.lower() for build_file in self.build_files)
                })
            
            # Also try to extract from patch/diff if available
            elif 'patch' in row and pd.notna(row['patch']):
                filenames = self._extract_filenames_from_patch(row['patch'])
                
                for filename in filenames:
                    file_ext = Path(filename).suffix.lower()
                    
                    results.append({
                        'commit_sha': row.get('commit_sha', row.get('sha', 'unknown')),
                        'pull_request_id': row.get('pull_request_id', row.get('pr_id', 'unknown')),
                        'filename': filename,
                        'file_extension': file_ext,
                        'is_java_file': file_ext in self.java_extensions,
                        'is_build_file': any(build_file in filename.lower() for build_file in self.build_files)
                    })
        
        return pd.DataFrame(results)
    
    def _extract_filenames_from_patch(self, patch_content: str) -> List[str]:
        """
        Extract filenames from git patch/diff content
        
        Args:
            patch_content: Raw patch/diff string
            
        Returns:
            List of extracted filenames
        """
        filenames = []
        
        if not isinstance(patch_content, str):
            return filenames
        
        # Common git diff patterns
        patterns = [
            r'^diff --git a/(.*?) b/.*?$',  # diff --git a/file b/file
            r'^\+\+\+ b/(.*?)$',            # +++ b/filename
            r'^--- a/(.*?)$',               # --- a/filename
            r'^index.*?\.\. .* (\d+)? (.*?)$'  # index line
        ]
        
        for line in patch_content.split('\n'):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    filename = match.group(1).strip()
                    if filename and filename != '/dev/null':
                        filenames.append(filename)
                        break
        
        return list(set(filenames))  # Remove duplicates
    
    def calculate_java_percentage_by_pr(self, file_details_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Java file percentage for each pull request
        
        Args:
            file_details_df: DataFrame with file details
            
        Returns:
            DataFrame with PR-level Java statistics
        """
        pr_stats = []
        
        for pr_id, pr_files in file_details_df.groupby('pull_request_id'):
            total_files = len(pr_files)
            java_files = pr_files['is_java_file'].sum()
            build_files = pr_files['is_build_file'].sum()
            
            java_percentage = (java_files / total_files * 100) if total_files > 0 else 0
            
            # Get unique file extensions
            extensions = pr_files['file_extension'].value_counts().to_dict()
            
            pr_stats.append({
                'pull_request_id': pr_id,
                'total_files': total_files,
                'java_files': java_files,
                'build_files': build_files,
                'java_percentage': java_percentage,
                'is_java_project': java_percentage >= self.min_java_percentage or build_files > 0,
                'file_extensions': extensions,
                'unique_extensions': len(extensions)
            })
        
        return pd.DataFrame(pr_stats)
    
    def get_java_projects_summary(self, pr_stats_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of Java project detection
        
        Args:
            pr_stats_df: DataFrame with PR statistics
            
        Returns:
            Summary dictionary
        """
        java_projects = pr_stats_df[pr_stats_df['is_java_project']]
        
        summary = {
            'total_prs': len(pr_stats_df),
            'java_prs': len(java_projects),
            'java_pr_percentage': len(java_projects) / len(pr_stats_df) * 100,
            'avg_java_percentage': java_projects['java_percentage'].mean(),
            'median_java_percentage': java_projects['java_percentage'].median(),
            'prs_with_build_files': (pr_stats_df['build_files'] > 0).sum(),
            'most_common_extensions': {}
        }
        
        # Get most common extensions in Java projects
        all_extensions = Counter()
        for extensions_dict in java_projects['file_extensions']:
            if isinstance(extensions_dict, dict):
                all_extensions.update(extensions_dict)
        
        summary['most_common_extensions'] = dict(all_extensions.most_common(10))
        
        return summary
    
    def save_java_pr_analysis(self, pr_stats_df: pd.DataFrame, output_path: str = None):
        """
        Save Java PR analysis results
        
        Args:
            pr_stats_df: DataFrame with PR statistics
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = "data/filtered/java_repositories/java_prs.parquet"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Filter for Java projects only
        java_prs = pr_stats_df[pr_stats_df['is_java_project']].copy()
        
        # Save to parquet
        java_prs.to_parquet(output_path, index=False)
        print(f"Saved {len(java_prs)} Java PRs to {output_path}")
        
        # Also save summary CSV
        csv_path = output_path.replace('.parquet', '_summary.csv')
        summary_df = java_prs[['pull_request_id', 'java_percentage', 'total_files', 'java_files', 'build_files']].copy()
        summary_df.to_csv(csv_path, index=False)
        print(f"Also saved summary as CSV: {csv_path}")
        
        return output_path