"""
Java project validator that combines all detection methods
"""
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader
from src.phase1_java_extraction.language_filter import LanguageFilter
from src.phase1_java_extraction.file_extension_checker import FileExtensionChecker
from src.phase1_java_extraction.maven_gradle_detector import MavenGradleDetector


class JavaProjectValidator:
    """Combines all Java detection methods to validate Java projects"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.loader = HFDatasetLoader(config_path)
        self.config = self.loader.config
        
        # Initialize component validators
        self.language_filter = LanguageFilter(config_path)
        self.extension_checker = FileExtensionChecker(config_path)
        self.build_detector = MavenGradleDetector(config_path)
        
        # Validation thresholds
        self.min_java_percentage = self.config["java_detection"]["min_java_percentage"]
        self.check_build_files = self.config["java_detection"]["check_build_files"]
    
    def run_comprehensive_java_detection(self) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive Java project detection using all methods
        
        Returns:
            Dictionary containing all analysis results
        """
        print("=== Starting Comprehensive Java Project Detection ===")
        
        results = {}
        
        # Step 1: Repository language filtering
        print("\n1. Loading and filtering repositories by language...")
        repos_df = self.language_filter.load_repository_data()
        java_repos_by_lang = self.language_filter.filter_by_repository_language(repos_df)
        
        # Get language statistics
        lang_stats = self.language_filter.analyze_repository_languages(repos_df)
        print(f"Language analysis: {lang_stats.get('java_related_count', 0)} Java-related repos")
        
        results['repositories'] = repos_df
        results['java_repos_by_language'] = java_repos_by_lang
        results['language_stats'] = lang_stats
        
        # Step 2: File extension analysis
        print("\n2. Analyzing commit files for Java extensions...")
        commit_details_df = self.extension_checker.load_commit_details()
        print(f"Loaded {len(commit_details_df)} commit details")
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        all_file_details = []
        
        for i in range(0, len(commit_details_df), chunk_size):
            chunk = commit_details_df.iloc[i:i+chunk_size]
            file_details = self.extension_checker.extract_file_paths_from_commit(chunk)
            all_file_details.append(file_details)
            print(f"Processed chunk {i//chunk_size + 1}/{(len(commit_details_df)-1)//chunk_size + 1}")
        
        # Combine all file details
        if all_file_details:
            file_details_df = pd.concat(all_file_details, ignore_index=True)
            print(f"Extracted {len(file_details_df)} file records")
            
            # Calculate Java percentages by PR
            pr_java_stats = self.extension_checker.calculate_java_percentage_by_pr(file_details_df)
            java_summary = self.extension_checker.get_java_projects_summary(pr_java_stats)
            
            results['file_details'] = file_details_df
            results['pr_java_stats'] = pr_java_stats
            results['java_file_summary'] = java_summary
            
            print(f"Found {java_summary.get('java_prs', 0)} Java PRs ({java_summary.get('java_pr_percentage', 0):.1f}%)")
        
        # Step 3: Build file detection
        print("\n3. Detecting Maven/Gradle build files...")
        build_files_df = self.build_detector.detect_build_files_in_commits(commit_details_df)
        
        if not build_files_df.empty:
            build_summary_df = self.build_detector.summarize_build_systems_by_pr(build_files_df)
            print(f"Found {len(build_summary_df)} PRs with build files")
            
            results['build_files'] = build_files_df
            results['build_summary'] = build_summary_df
        else:
            print("No build files detected")
            results['build_files'] = pd.DataFrame()
            results['build_summary'] = pd.DataFrame()
        
        return results
    
    def combine_validation_results(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all validation results into final Java project classification
        
        Args:
            results: Dictionary with all analysis results
            
        Returns:
            DataFrame with final Java project validation
        """
        print("\n4. Combining validation results...")
        
        # Start with PR-level Java statistics
        if 'pr_java_stats' not in results or results['pr_java_stats'].empty:
            print("No PR Java statistics available")
            return pd.DataFrame()
        
        final_results = results['pr_java_stats'].copy()
        
        # Merge with build system information
        if 'build_summary' in results and not results['build_summary'].empty:
            build_df = results['build_summary']
            final_results = final_results.merge(
                build_df[['pull_request_id', 'has_maven', 'has_gradle', 'primary_build_system', 'max_confidence']],
                on='pull_request_id',
                how='left'
            )
            
            # Fill NaN values for PRs without build files
            final_results['has_maven'] = final_results['has_maven'].fillna(False)
            final_results['has_gradle'] = final_results['has_gradle'].fillna(False)
            final_results['primary_build_system'] = final_results['primary_build_system'].fillna('none')
            final_results['max_confidence'] = final_results['max_confidence'].fillna(0.0)
        else:
            # Add empty build system columns
            final_results['has_maven'] = False
            final_results['has_gradle'] = False
            final_results['primary_build_system'] = 'none'
            final_results['max_confidence'] = 0.0
        
        # Apply comprehensive Java project validation
        final_results['is_java_project_final'] = self._apply_final_validation_rules(final_results)
        
        # Add confidence scoring
        final_results['java_confidence_score'] = self._calculate_confidence_score(final_results)
        
        return final_results
    
    def _apply_final_validation_rules(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply final validation rules to determine Java projects
        
        Args:
            df: DataFrame with all validation metrics
            
        Returns:
            Boolean series indicating Java projects
        """
        conditions = []
        
        # Rule 1: High Java file percentage
        conditions.append(df['java_percentage'] >= self.min_java_percentage)
        
        # Rule 2: Has build files (Maven/Gradle)
        if self.check_build_files:
            conditions.append(df['has_maven'] | df['has_gradle'])
        
        # Rule 3: Has Java files and some indication of Java project
        conditions.append(
            (df['java_files'] > 0) & 
            ((df['java_percentage'] >= 30) | (df['has_maven']) | (df['has_gradle']))
        )
        
        # Combine conditions (OR logic - any condition can qualify as Java project)
        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition = final_condition | condition
        
        return final_condition
    
    def _calculate_confidence_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence score for Java project classification
        
        Args:
            df: DataFrame with validation metrics
            
        Returns:
            Series with confidence scores (0-1)
        """
        confidence = pd.Series(0.0, index=df.index)
        
        # Java file percentage contribution (max 0.6)
        confidence += (df['java_percentage'] / 100) * 0.6
        
        # Build file presence contribution (max 0.3)
        build_confidence = df['max_confidence'].fillna(0.0)
        confidence += build_confidence * 0.3
        
        # Java file count contribution (max 0.1)
        normalized_java_files = (df['java_files'] / df['java_files'].max()).fillna(0)
        confidence += normalized_java_files * 0.1
        
        # Cap at 1.0
        confidence = confidence.clip(upper=1.0)
        
        return confidence
    
    def generate_validation_report(self, final_results: pd.DataFrame) -> Dict:
        """
        Generate comprehensive validation report
        
        Args:
            final_results: DataFrame with final validation results
            
        Returns:
            Dictionary with validation report
        """
        java_projects = final_results[final_results['is_java_project_final']]
        
        report = {
            'total_prs_analyzed': len(final_results),
            'java_projects_identified': len(java_projects),
            'java_project_percentage': len(java_projects) / len(final_results) * 100,
            
            'java_file_stats': {
                'avg_java_percentage': java_projects['java_percentage'].mean(),
                'median_java_percentage': java_projects['java_percentage'].median(),
                'min_java_percentage': java_projects['java_percentage'].min(),
                'max_java_percentage': java_projects['java_percentage'].max()
            },
            
            'build_system_stats': {
                'projects_with_maven': (java_projects['has_maven'] == True).sum(),
                'projects_with_gradle': (java_projects['has_gradle'] == True).sum(),
                'projects_with_both': ((java_projects['has_maven'] == True) & 
                                     (java_projects['has_gradle'] == True)).sum(),
                'projects_without_build_files': ((java_projects['has_maven'] == False) & 
                                               (java_projects['has_gradle'] == False)).sum()
            },
            
            'confidence_stats': {
                'avg_confidence': java_projects['java_confidence_score'].mean(),
                'high_confidence_projects': (java_projects['java_confidence_score'] >= 0.8).sum(),
                'medium_confidence_projects': ((java_projects['java_confidence_score'] >= 0.5) & 
                                             (java_projects['java_confidence_score'] < 0.8)).sum(),
                'low_confidence_projects': (java_projects['java_confidence_score'] < 0.5).sum()
            }
        }
        
        return report
    
    def save_validation_results(self, final_results: pd.DataFrame, report: Dict):
        """
        Save all validation results
        
        Args:
            final_results: DataFrame with final validation results
            report: Validation report dictionary
        """
        output_dir = Path("data/filtered/java_repositories")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Java projects
        java_projects = final_results[final_results['is_java_project_final']]
        java_projects.to_parquet(output_dir / "validated_java_projects.parquet", index=False)
        java_projects.to_csv(output_dir / "validated_java_projects.csv", index=False)
        
        # Save full results
        final_results.to_parquet(output_dir / "full_validation_results.parquet", index=False)
        
        # Save report
        import json
        with open(output_dir / "validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nValidation complete! Results saved to {output_dir}")
        print(f"Java projects identified: {len(java_projects)}")
        print(f"Overall Java project rate: {report['java_project_percentage']:.1f}%")