"""
Maven/Gradle build file detector for Java project identification
"""
import pandas as pd
import re
from typing import Dict, Set, List, Tuple
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader.hf_dataset_loader import HFDatasetLoader


class MavenGradleDetector:
    """Detects Java projects by analyzing Maven and Gradle build files in commits"""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.loader = HFDatasetLoader(config_path)
        self.config = self.loader.config
        
        # Build file patterns from config
        self.build_file_patterns = set(self.config["java_detection"]["build_file_patterns"])
        
        # Additional patterns for build file detection
        self.maven_indicators = {
            'pom.xml', 'pom.xml.template', 'parent-pom.xml'
        }
        
        self.gradle_indicators = {
            'build.gradle', 'build.gradle.kts', 'settings.gradle', 
            'settings.gradle.kts', 'gradle.properties', 'gradlew', 
            'gradlew.bat', 'gradle/wrapper/gradle-wrapper.properties'
        }
        
        # Content patterns to look for in patches
        self.maven_content_patterns = [
            r'<groupId>.*</groupId>',
            r'<artifactId>.*</artifactId>',
            r'<version>.*</version>',
            r'<dependencies>',
            r'<dependency>',
            r'<maven\.',
            r'maven-compiler-plugin'
        ]
        
        self.gradle_content_patterns = [
            r'dependencies\s*\{',
            r'implementation\s*[\'"]',
            r'compile\s*[\'"]',
            r'testImplementation\s*[\'"]',
            r'apply plugin:',
            r'plugins\s*\{',
            r'repositories\s*\{',
            r'buildscript\s*\{'
        ]
    
    def detect_build_files_in_commits(self, commit_details_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect build files in commit details
        
        Args:
            commit_details_df: Commit details dataframe
            
        Returns:
            DataFrame with build file detection results
        """
        results = []
        
        for idx, row in commit_details_df.iterrows():
            commit_sha = row.get('commit_sha', row.get('sha', 'unknown'))
            pr_id = row.get('pull_request_id', row.get('pr_id', 'unknown'))
            
            # Check filename if available
            filename = row.get('filename', '')
            patch_content = row.get('patch', '')
            
            build_file_info = self._analyze_build_files(filename, patch_content)
            
            if build_file_info['has_build_files']:
                results.append({
                    'commit_sha': commit_sha,
                    'pull_request_id': pr_id,
                    'filename': filename,
                    'has_maven': build_file_info['has_maven'],
                    'has_gradle': build_file_info['has_gradle'],
                    'maven_files': build_file_info['maven_files'],
                    'gradle_files': build_file_info['gradle_files'],
                    'build_system_confidence': build_file_info['confidence']
                })
        
        return pd.DataFrame(results)
    
    def _analyze_build_files(self, filename: str, patch_content: str) -> Dict:
        """
        Analyze individual commit for build file presence
        
        Args:
            filename: File name from commit
            patch_content: Patch/diff content
            
        Returns:
            Dictionary with build file analysis results
        """
        result = {
            'has_build_files': False,
            'has_maven': False,
            'has_gradle': False,
            'maven_files': [],
            'gradle_files': [],
            'confidence': 0.0
        }
        
        files_to_check = []
        
        # Add filename if provided
        if filename and isinstance(filename, str):
            files_to_check.append(filename)
        
        # Extract filenames from patch content
        if patch_content and isinstance(patch_content, str):
            extracted_files = self._extract_filenames_from_patch(patch_content)
            files_to_check.extend(extracted_files)
        
        # Check each file
        for file_path in files_to_check:
            file_lower = file_path.lower()
            
            # Maven detection
            if any(maven_file in file_lower for maven_file in self.maven_indicators):
                result['has_maven'] = True
                result['maven_files'].append(file_path)
                result['confidence'] += 0.8
            
            # Gradle detection
            if any(gradle_file in file_lower for gradle_file in self.gradle_indicators):
                result['has_gradle'] = True
                result['gradle_files'].append(file_path)
                result['confidence'] += 0.8
        
        # Content analysis for additional confidence
        if patch_content and isinstance(patch_content, str):
            content_confidence = self._analyze_patch_content(patch_content)
            result['confidence'] += content_confidence
            
            # If we found content patterns but no explicit files, still mark as having build system
            if content_confidence > 0.3 and not result['has_maven'] and not result['has_gradle']:
                # Try to determine which build system based on content
                maven_score = sum(1 for pattern in self.maven_content_patterns 
                                if re.search(pattern, patch_content, re.IGNORECASE))
                gradle_score = sum(1 for pattern in self.gradle_content_patterns 
                                 if re.search(pattern, patch_content, re.IGNORECASE))
                
                if maven_score > gradle_score:
                    result['has_maven'] = True
                elif gradle_score > 0:
                    result['has_gradle'] = True
        
        # Normalize confidence to 0-1 range
        result['confidence'] = min(result['confidence'], 1.0)
        result['has_build_files'] = result['has_maven'] or result['has_gradle']
        
        return result
    
    def _extract_filenames_from_patch(self, patch_content: str) -> List[str]:
        """Extract filenames from git patch content"""
        filenames = []
        
        patterns = [
            r'^diff --git a/(.*?) b/',
            r'^\+\+\+ b/(.*?)$',
            r'^--- a/(.*?)$'
        ]
        
        for line in patch_content.split('\n'):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    filename = match.group(1).strip()
                    if filename and filename != '/dev/null':
                        filenames.append(filename)
        
        return list(set(filenames))
    
    def _analyze_patch_content(self, patch_content: str) -> float:
        """
        Analyze patch content for build system indicators
        
        Args:
            patch_content: Git patch/diff content
            
        Returns:
            Confidence score (0-1) for build system presence
        """
        confidence = 0.0
        
        # Check Maven patterns
        maven_matches = sum(1 for pattern in self.maven_content_patterns 
                           if re.search(pattern, patch_content, re.IGNORECASE))
        
        # Check Gradle patterns
        gradle_matches = sum(1 for pattern in self.gradle_content_patterns 
                           if re.search(pattern, patch_content, re.IGNORECASE))
        
        # Calculate confidence based on matches
        total_patterns = len(self.maven_content_patterns) + len(self.gradle_content_patterns)
        total_matches = maven_matches + gradle_matches
        
        if total_matches > 0:
            confidence = min(total_matches / total_patterns * 2, 0.5)  # Max 0.5 from content
        
        return confidence
    
    def summarize_build_systems_by_pr(self, build_files_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize build system usage by pull request
        
        Args:
            build_files_df: DataFrame with build file detections
            
        Returns:
            DataFrame with PR-level build system summary
        """
        if build_files_df.empty:
            return pd.DataFrame()
        
        pr_summary = []
        
        for pr_id, pr_data in build_files_df.groupby('pull_request_id'):
            summary = {
                'pull_request_id': pr_id,
                'has_maven': pr_data['has_maven'].any(),
                'has_gradle': pr_data['has_gradle'].any(),
                'maven_files_count': pr_data['maven_files'].apply(len).sum(),
                'gradle_files_count': pr_data['gradle_files'].apply(len).sum(),
                'max_confidence': pr_data['build_system_confidence'].max(),
                'avg_confidence': pr_data['build_system_confidence'].mean(),
                'commits_with_build_files': len(pr_data)
            }
            
            # Determine primary build system
            if summary['has_maven'] and summary['has_gradle']:
                summary['primary_build_system'] = 'both'
            elif summary['has_maven']:
                summary['primary_build_system'] = 'maven'
            elif summary['has_gradle']:
                summary['primary_build_system'] = 'gradle'
            else:
                summary['primary_build_system'] = 'unknown'
            
            pr_summary.append(summary)
        
        return pd.DataFrame(pr_summary)
    
    def save_build_system_analysis(self, pr_summary_df: pd.DataFrame, output_path: str = None):
        """
        Save build system analysis results
        
        Args:
            pr_summary_df: DataFrame with PR build system summary
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = "data/filtered/java_repositories/build_systems.parquet"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        pr_summary_df.to_parquet(output_path, index=False)
        print(f"Saved build system analysis for {len(pr_summary_df)} PRs to {output_path}")
        
        # Also save as CSV
        csv_path = output_path.replace('.parquet', '.csv')
        pr_summary_df.to_csv(csv_path, index=False)
        print(f"Also saved as CSV: {csv_path}")
        
        return output_path