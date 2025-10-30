"""
Refactoring detection using RefactoringMiner or pattern analysis
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.phase3_refactoring_analysis.refminer_wrapper import RefactoringMinerWrapper
from src.phase3_refactoring_analysis.self_affirmation import SELF_AFFIRMATION_PATTERN

class RefactoringDetector:
    """Detects refactoring in Java commits"""
    
    def __init__(self):
        self.output_dir = Path("data/analysis/refactoring_instances")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for last run
        self._last_refminer_results = None

    def detect_refactoring_with_refminer(self, commits: pd.DataFrame, max_commits: int = 200) -> pd.DataFrame:
        """
        Detect refactoring using RefactoringMiner and annotate the commits DataFrame.

        Args:
            commits: DataFrame of Java file changes; must include 'sha' and ideally 'html_url'.
            max_commits: Max unique commits to analyze with RefactoringMiner.

        Returns:
            DataFrame with columns: 'has_refactoring', 'refactoring_type_count', 'primary_refactoring_type'.
        """
        print("Using RefactoringMiner for refactoring detection (with fallback)...")

        # Initialize wrapper and verify availability
        wrapper = RefactoringMinerWrapper()
        if not wrapper.is_available():
            print("RefactoringMiner not available at runtime; falling back to pattern-based detection.")
            return self.detect_refactoring_patterns(commits)

        # Select commits that have a GitHub URL for -gc mode, de-duplicate by sha
        if 'sha' not in commits.columns:
            raise ValueError("Expected 'sha' column in commits DataFrame")

        local_repo_override = os.environ.get('REFMINER_LOCAL_REPO')
        if local_repo_override and Path(local_repo_override).exists():
            unique_commits = commits.drop_duplicates('sha')
        else:
            commits_with_urls = commits[commits.get('html_url').notna()] if 'html_url' in commits.columns else pd.DataFrame()
            if commits_with_urls.empty:
                print("No 'html_url' found and no REFMINER_LOCAL_REPO set; falling back to pattern-based detection.")
                return self.detect_refactoring_patterns(commits)
            unique_commits = commits_with_urls.drop_duplicates('sha')
        if len(unique_commits) > max_commits:
            unique_commits = unique_commits.head(max_commits)

        # Run analysis
        ref_df = wrapper.analyze_commits_batch(unique_commits, max_commits=max_commits)
        self._last_refminer_results = ref_df

        # Build per-commit aggregates
        if ref_df.empty:
            print("RefactoringMiner found no refactorings; annotating with no-refactor results.")
            commits = commits.copy()
            commits['has_refactoring'] = False
            commits['refactoring_type_count'] = 0
            commits['primary_refactoring_type'] = 'none'
            return commits

        # Count refactorings and primary type per commit
        per_commit_counts = ref_df.groupby('commit_sha')['refactoring_type'].agg(list)

        commit_has_refactor = per_commit_counts.index.to_series().to_frame(name='sha')
        commit_has_refactor['has_refactoring'] = True
        commit_has_refactor['refactoring_type_count'] = per_commit_counts.apply(len).values
        # Choose the first type in the list as primary (RefactoringMiner has no severity ordering)
        commit_has_refactor['primary_refactoring_type'] = per_commit_counts.apply(lambda lst: lst[0] if lst else 'unknown').values

        # Map back to rows by 'sha'
        commits = commits.copy()
        commits = commits.merge(commit_has_refactor, on='sha', how='left')
        commits['has_refactoring'] = commits['has_refactoring'].fillna(False).astype(bool)
        commits['refactoring_type_count'] = commits['refactoring_type_count'].fillna(0).astype(int)
        commits['primary_refactoring_type'] = commits['primary_refactoring_type'].fillna('none')

        return commits
    
    def load_java_commits(self) -> pd.DataFrame:
        """Load Java commits from previous extraction"""
        # Check for both parquet and csv files
        parquet_path = Path("data/filtered/java_repositories/java_file_commits_for_refactoring.parquet")
        csv_path = Path("data/filtered/java_repositories/java_file_commits_for_refactoring.csv")
        
        if parquet_path.exists():
            commits = pd.read_parquet(parquet_path)
            print(f"Loaded {len(commits)} Java file changes from parquet file")
        elif csv_path.exists():
            commits = pd.read_csv(csv_path)
            print(f"Loaded {len(commits)} Java file changes from CSV file")
        else:
            raise FileNotFoundError(f"Java commits file not found at {parquet_path} or {csv_path}")
        
        print(f"Total: {len(commits)} Java file changes from {commits['sha'].nunique()} unique commits")
        return commits
    
    def detect_refactoring_patterns(self, commits: pd.DataFrame) -> pd.DataFrame:
        """
        Pattern-based refactoring detection from commit messages and patches
        
        Args:
            commits: DataFrame with Java commits
            
        Returns:
            DataFrame with refactoring detection results
        """
        print("Detecting refactoring patterns...")
        
        # Define refactoring patterns
        refactoring_keywords = {
            'extract_method': [r'\bextract(?:ed)?\s+method\b', r'\bmethod\s+extract(?:ed|ion)?\b'],
            'rename': [r'\brename(?:d)?\b', r'\brenam(?:ing|ed)\b'],
            'move': [r'\bmove(?:d)?\b', r'\breloca(?:te|ted)\b', r'\bmigrat(?:e|ed)\b'],
            'inline': [r'\binlin(?:e|ed|ing)\b'],
            'extract_class': [r'\bextract(?:ed)?\s+class\b', r'\bclass\s+extract(?:ed|ion)?\b'],
            'pull_up': [r'\bpull(?:ed)?\s*up\b', r'\bpull\s+up\b'],
            'push_down': [r'\bpush(?:ed)?\s*down\b', r'\bpush\s+down\b'],
            'refactor_general': [r'\brefactor(?:ing|ed)?\b', r'\brestructur(?:e|ed|ing)\b'],
            'cleanup': [r'\bclean\s*up\b', r'\bclean(?:ed|ing)\b', r'\bcode\s+clean\b'],
            'simplify': [r'\bsimplif(?:y|ied|ying)\b'],
            'optimize': [r'\boptimiz(?:e|ed|ing|ation)\b'],
            'reorganize': [r'\breorganiz(?:e|ed|ing)\b', r'\breorg\b']
        }
        
        # Create detection columns
        for pattern_name, patterns in refactoring_keywords.items():
            pattern_regex = '|'.join(patterns)
            commits[f'has_{pattern_name}'] = commits['message'].str.contains(
                pattern_regex, case=False, na=False, regex=True
            )
        
        # Overall refactoring indicator
        refactoring_cols = [col for col in commits.columns if col.startswith('has_')]
        commits['has_refactoring'] = commits[refactoring_cols].any(axis=1)
        
        # Count refactoring types per commit
        commits['refactoring_type_count'] = commits[refactoring_cols].sum(axis=1)
        
        # Get primary refactoring type
        def get_primary_type(row):
            for col in refactoring_cols:
                if row[col]:
                    return col.replace('has_', '')
            return 'none'
        
        commits['primary_refactoring_type'] = commits.apply(get_primary_type, axis=1)
        
        return commits
    
    def analyze_refactoring_by_agent(self, commits_with_refactoring: pd.DataFrame) -> Dict:
        """
        Analyze refactoring patterns by agent (agentic vs human)
        
        Args:
            commits_with_refactoring: DataFrame with refactoring detection
            
        Returns:
            Dictionary with comparative analysis
        """
        print("Analyzing refactoring by agent type...")
        print(f"Available columns: {list(commits_with_refactoring.columns)}")
        
        # Since we're using AI agents dataset, determine how to identify agents
        if 'agent' in commits_with_refactoring.columns:
            # Use existing agent column
            agentic_commits = commits_with_refactoring[commits_with_refactoring['agent'].notna()]
            human_commits = commits_with_refactoring[commits_with_refactoring['agent'].isna()]
            print(f"Using 'agent' column for agent identification")
        elif 'author' in commits_with_refactoring.columns:
            # In AI agents dataset, all commits are from AI agents, so author column contains agent info
            agentic_commits = commits_with_refactoring.copy()
            human_commits = pd.DataFrame()  # Empty since all are AI agents
            print(f"Treating all commits as AI agent commits based on author column")
        else:
            # Fallback: treat all as agentic since it's an AI agents dataset
            agentic_commits = commits_with_refactoring.copy()
            human_commits = pd.DataFrame()
            print(f"Treating all commits as AI agent commits (fallback)")
        
        print(f"AI agent commits: {len(agentic_commits)}, Human commits: {len(human_commits)}")
        
        analysis = {
            'overall': {
                'total_file_changes': len(commits_with_refactoring),
                'unique_commits': commits_with_refactoring['sha'].nunique(),
                'refactoring_file_changes': commits_with_refactoring['has_refactoring'].sum(),
                'unique_refactoring_commits': commits_with_refactoring[commits_with_refactoring['has_refactoring']]['sha'].nunique(),
                'refactoring_percentage': (
                    commits_with_refactoring[commits_with_refactoring['has_refactoring']]['sha'].nunique() / 
                    commits_with_refactoring['sha'].nunique() * 100
                )
            },
            'agentic': self._analyze_subset(agentic_commits, 'AI_Agent') if len(agentic_commits) > 0 else None,
            'human': self._analyze_subset(human_commits, 'Human') if len(human_commits) > 0 else None
        }
        
        # Compare refactoring types (only if we have both types)
        if len(agentic_commits) > 0 and len(human_commits) > 0:
            analysis['refactoring_types_comparison'] = self._compare_refactoring_types(
                agentic_commits, human_commits
            )
        
        # Agent-specific analysis (by individual agent if available)
        if len(agentic_commits) > 0:
            if 'agent' in agentic_commits.columns:
                agent_groups = agentic_commits.groupby('agent')
                analysis['by_agent'] = {}
                for agent_name, agent_data in agent_groups:
                    if len(agent_data) > 10:  # Only analyze agents with sufficient data
                        analysis['by_agent'][agent_name] = self._analyze_subset(agent_data, agent_name)
            elif 'author' in agentic_commits.columns:
                # Group by author as they represent different AI agents
                author_groups = agentic_commits.groupby('author')
                analysis['by_agent'] = {}
                for author_name, author_data in author_groups:
                    if len(author_data) > 10:  # Only analyze agents with sufficient data
                        analysis['by_agent'][author_name] = self._analyze_subset(author_data, author_name)
        
        return analysis
    
    def _analyze_subset(self, subset: pd.DataFrame, label: str) -> Dict:
        """Analyze a subset of commits"""
        if len(subset) == 0:
            return {'error': f'No {label} commits found'}
        
        refactoring_subset = subset[subset['has_refactoring']]
        unique_commits = subset['sha'].nunique()
        unique_refactoring = refactoring_subset['sha'].nunique() if len(refactoring_subset) > 0 else 0
        
        result = {
            'label': label,
            'total_file_changes': len(subset),
            'unique_commits': unique_commits,
            'refactoring_file_changes': len(refactoring_subset),
            'unique_refactoring_commits': unique_refactoring,
            'refactoring_percentage': (unique_refactoring / unique_commits * 100) if unique_commits > 0 else 0,
            'avg_refactoring_types_per_commit': refactoring_subset['refactoring_type_count'].mean() if len(refactoring_subset) > 0 else 0
        }
        
        # Top refactoring types
        if len(refactoring_subset) > 0:
            type_counts = refactoring_subset['primary_refactoring_type'].value_counts()
            result['top_refactoring_types'] = type_counts.head(5).to_dict()
        
        return result
    
    def _compare_refactoring_types(self, agentic: pd.DataFrame, human: pd.DataFrame) -> Dict:
        """Compare refactoring types between agentic and human commits"""
        comparison = {}
        
        refactoring_cols = [col for col in agentic.columns if col.startswith('has_') and col != 'has_refactoring']
        
        for col in refactoring_cols:
            refactoring_type = col.replace('has_', '')
            agentic_count = agentic[col].sum()
            human_count = human[col].sum()
            
            agentic_pct = (agentic_count / len(agentic) * 100) if len(agentic) > 0 else 0
            human_pct = (human_count / len(human) * 100) if len(human) > 0 else 0
            
            comparison[refactoring_type] = {
                'agentic_count': int(agentic_count),
                'agentic_percentage': float(agentic_pct),
                'human_count': int(human_count),
                'human_percentage': float(human_pct),
                'difference': float(agentic_pct - human_pct)
            }
        
        return comparison
    
    def check_self_affirmation(self, commits_with_refactoring: pd.DataFrame) -> Dict:
        """
        Check for self-affirmed refactoring (explicit mention in commit message)
        
        Args:
            commits_with_refactoring: DataFrame with refactoring detection
            
        Returns:
            Dictionary with self-affirmation analysis (RQ2)
        """
        print("Checking self-affirmation...")
        
        commits_with_refactoring = commits_with_refactoring.copy()
        commits_with_refactoring["is_self_affirmed"] = (
            commits_with_refactoring["message"]
            .astype(str)
            .str.contains(SELF_AFFIRMATION_PATTERN, regex=True, na=False)
        )
        
        # Determine how to split by agent type (similar to analyze_refactoring_by_agent)
        if 'agent' in commits_with_refactoring.columns:
            agentic = commits_with_refactoring[commits_with_refactoring['agent'].notna()]
            human = commits_with_refactoring[commits_with_refactoring['agent'].isna()]
        else:
            # All are AI agents in this dataset
            agentic = commits_with_refactoring.copy()
            human = pd.DataFrame()
        
        result = {
            'overall': {
                'total_refactoring': commits_with_refactoring['has_refactoring'].sum(),
                'self_affirmed': commits_with_refactoring['is_self_affirmed'].sum(),
                'self_affirmation_rate': (
                    commits_with_refactoring['is_self_affirmed'].sum() / 
                    commits_with_refactoring['has_refactoring'].sum() * 100
                ) if commits_with_refactoring['has_refactoring'].sum() > 0 else 0
            }
        }
        
        if len(agentic) > 0:
            agentic_refactoring = agentic[agentic['has_refactoring']]
            result['agentic'] = {
                'total_refactoring': len(agentic_refactoring),
                'self_affirmed': agentic_refactoring['is_self_affirmed'].sum() if len(agentic_refactoring) > 0 else 0,
                'self_affirmation_rate': (
                    agentic_refactoring['is_self_affirmed'].sum() / len(agentic_refactoring) * 100
                ) if len(agentic_refactoring) > 0 else 0
            }
        
        if len(human) > 0:
            human_refactoring = human[human['has_refactoring']]
            result['human'] = {
                'total_refactoring': len(human_refactoring),
                'self_affirmed': human_refactoring['is_self_affirmed'].sum() if len(human_refactoring) > 0 else 0,
                'self_affirmation_rate': (
                    human_refactoring['is_self_affirmed'].sum() / len(human_refactoring) * 100
                ) if len(human_refactoring) > 0 else 0
            }
        
        return result
    
    def save_results(self, commits_with_refactoring: pd.DataFrame, analysis: Dict, self_affirmation: Dict):
        """Save refactoring detection results"""
        
        # Save enhanced commits with refactoring detection
        commits_with_refactoring.to_parquet(self.output_dir / "commits_with_refactoring.parquet", index=False)
        
        # Save refactoring-only commits
        refactoring_only = commits_with_refactoring[commits_with_refactoring['has_refactoring']]
        refactoring_only.to_parquet(self.output_dir / "refactoring_commits.parquet", index=False)
        refactoring_only.to_csv(self.output_dir / "refactoring_commits.csv", index=False)
        
        print(f"Saved {len(refactoring_only)} refactoring commits to {self.output_dir}")
        
        # Save analysis results
        combined_analysis = {
            'refactoring_analysis': analysis,
            'self_affirmation_analysis': self_affirmation,
            'summary': {
                'total_commits_analyzed': analysis['overall']['unique_commits'],
                'refactoring_commits_found': analysis['overall']['unique_refactoring_commits'],
                'overall_refactoring_rate': analysis['overall']['refactoring_percentage'],
                'agentic_refactoring_rate': analysis['agentic']['refactoring_percentage'] if analysis['agentic'] else 0,
                'human_refactoring_rate': analysis['human']['refactoring_percentage'] if analysis['human'] else 0,
                'self_affirmation_rate': self_affirmation['overall']['self_affirmation_rate']
            }
        }
        
        sar_outputs = self._generate_sar_summary_outputs(commits_with_refactoring)
        if sar_outputs:
            sar_summary = combined_analysis.setdefault("sar_summary", {})
            sar_summary["summary_csv"] = str(sar_outputs["summary_csv"])
            sar_summary["summary_parquet"] = str(sar_outputs["summary_parquet"])
            if sar_outputs["plot_path"]:
                sar_summary["plot_path"] = str(sar_outputs["plot_path"])
        
        with open(self.output_dir / "refactoring_analysis.json", 'w') as f:
            json.dump(combined_analysis, f, indent=2, default=str)
        
        print(f"Saved analysis to {self.output_dir}/refactoring_analysis.json")
        
        return self.output_dir

    def _generate_sar_summary_outputs(self, commits_with_refactoring: pd.DataFrame) -> Optional[Dict[str, Optional[Path]]]:
        """
        Create SAR vs Non-SAR summary data and plot for RQ1 analysis.
        """
        if 'sha' not in commits_with_refactoring.columns or 'is_self_affirmed' not in commits_with_refactoring.columns:
            print("Skipping SAR summary: required columns missing")
            return None
        
        commit_level_cols = ['sha', 'is_self_affirmed', 'has_refactoring']
        missing_cols = [col for col in commit_level_cols if col not in commits_with_refactoring.columns]
        if missing_cols:
            print(f"Skipping SAR summary: missing columns {missing_cols}")
            return None
        
        commit_level = commits_with_refactoring[commit_level_cols].drop_duplicates('sha')
        if commit_level.empty:
            print("Skipping SAR summary: no commit-level data available")
            return None
        
        commit_level['is_self_affirmed'] = commit_level['is_self_affirmed'].fillna(False)
        commit_level['sar_category'] = commit_level['is_self_affirmed'].map({True: 'SAR', False: 'Non-SAR'})
        
        categories = ['Non-SAR', 'SAR']
        total_commits = commit_level.groupby('sar_category')['sha'].nunique().reindex(categories, fill_value=0)
        refactoring_commits = (
            commit_level[commit_level['has_refactoring']]
            .groupby('sar_category')['sha']
            .nunique()
            .reindex(categories, fill_value=0)
        )
        
        summary_df = pd.DataFrame({
            'category': categories,
            'total_commits': total_commits.values.astype(int),
            'refactoring_commits': refactoring_commits.values.astype(int),
        })
        summary_df['refactoring_rate'] = summary_df.apply(
            lambda row: row['refactoring_commits'] / row['total_commits'] if row['total_commits'] else 0.0,
            axis=1
        )
        summary_df['refactoring_rate_pct'] = (summary_df['refactoring_rate'] * 100).round(2)
        
        if self._last_refminer_results is not None and not self._last_refminer_results.empty:
            commit_lookup = commit_level[['sha', 'sar_category']]
            refminer_counts = (
                self._last_refminer_results.merge(commit_lookup, left_on='commit_sha', right_on='sha', how='left')
                .groupby('sar_category')
                .size()
                .reindex(categories, fill_value=0)
            )
            summary_df['refactoring_instances'] = refminer_counts.values.astype(int)
        else:
            summary_df['refactoring_instances'] = 0
        
        summary_dir = self.output_dir
        summary_csv = summary_dir / "sar_commit_summary.csv"
        summary_parquet = summary_dir / "sar_commit_summary.parquet"
        summary_df.to_csv(summary_csv, index=False)
        summary_df.to_parquet(summary_parquet, index=False)
        print(f"Saved SAR vs Non-SAR summary to {summary_csv}")
        
        plot_dir = Path("outputs/research_questions/rq1")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path: Optional[Path] = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not generate SAR plot (matplotlib unavailable: {exc})")
        else:
            plot_path = plot_dir / "sar_vs_non_sar_refactoring_rate.png"
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                summary_df['category'],
                summary_df['refactoring_rate_pct'],
                color=['#6baed6', '#fd8d3c']
            )
            ax.set_ylabel("Refactoring commits per total commits (%)")
            ax.set_xlabel("Commit category")
            ax.set_title("SAR vs Non-SAR Refactoring Rate\n(all Java commits)", pad=12)
            ax.set_ylim(0, max(5, summary_df['refactoring_rate_pct'].max() * 1.2 if not summary_df.empty else 1))
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            for bar, (_, row) in zip(bars, summary_df.iterrows()):
                height = bar.get_height()
                label = f"{row['refactoring_rate_pct']:.1f}%\n({row['refactoring_commits']}/{row['total_commits']})"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(0.3, height * 0.05),
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )
            
            fig.tight_layout()
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved SAR vs Non-SAR plot to {plot_path}")
        
        return {"summary_csv": summary_csv, "summary_parquet": summary_parquet, "plot_path": plot_path}
