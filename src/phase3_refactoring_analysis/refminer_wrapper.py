"""
RefactoringMiner wrapper for accurate refactoring detection

Supports two modes:
- Local repo mode (-c): use a locally cloned repository path
- GitHub mode (-gc): use repo URL and commit id (requires OAuth in github-oauth.properties)

Env vars:
- REFMINER_LOCAL_REPO: absolute or relative path to a local repo to use with -c
- REFMINER_MAX_COMMITS: cap number of unique commits analyzed (handled by caller)
- REFMINER_TIMEOUT: per-commit timeout in seconds (default: 120)
"""
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests
import os


class RefactoringMinerWrapper:
    """Wrapper for RefactoringMiner tool"""
    
    def __init__(self):
        self.refminer_jar = None
        self.refminer_root = Path("tools/RefactoringMiner")
        self.output_dir = Path("data/analysis/refactoring_instances")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_json_dir = self.output_dir / "refminer_raw"
        self.raw_json_dir.mkdir(parents=True, exist_ok=True)

        # timeouts and modes
        self.timeout = int(os.environ.get("REFMINER_TIMEOUT", 10000))
        self.local_repo_override = os.environ.get("REFMINER_LOCAL_REPO")
        self.save_json = os.environ.get("REFMINER_SAVE_JSON", "1") not in ("0", "false", "False")

        # Check if RefactoringMiner is available
        self.setup_refminer()
    
    def setup_refminer(self):
        """Setup RefactoringMiner by detecting existing installation"""
        refminer_dir = self.refminer_root

        # Check for existing RefactoringMiner installation (prioritize fat/shadow jars)
        if refminer_dir.exists():
            # First try to find fat/shadow jar (executable) - this is what we built
            fat_jars = list(refminer_dir.glob("build/libs/RM-fat.jar")) + list(refminer_dir.glob("build/libs/*-all.jar"))
            if fat_jars:
                self.refminer_jar = fat_jars[0].resolve()
                print(f"Found RefactoringMiner fat jar at: {self.refminer_jar}")
                return
            
            # Fallback to regular jar
            possible_jars = list(refminer_dir.glob("build/libs/RefactoringMiner*.jar"))
            if possible_jars:
                self.refminer_jar = possible_jars[0].resolve()
                print(f"Found RefactoringMiner at: {self.refminer_jar}")
                print("Warning: Using regular jar, may not be executable. Consider running './gradlew shadowJar' in tools/RefactoringMiner/")
                return
        
        # RefactoringMiner not found
        print("RefactoringMiner not found at tools/RefactoringMiner/")
        print("\nTo set up RefactoringMiner:")
        print("1. git clone https://github.com/tsantalis/RefactoringMiner.git tools/RefactoringMiner")
        print("2. cd tools/RefactoringMiner")
        print("3. ./gradlew shadowJar")
        print("4. The executable jar will be at build/libs/RM-fat.jar")
        self.refminer_jar = None
    
    def is_available(self) -> bool:
        """Check if RefactoringMiner is available"""
        return self.refminer_jar is not None and self.refminer_jar.exists()
    
    def _run_rm(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run RefactoringMiner with cwd set so github-oauth.properties is discoverable."""
        if not self.is_available():
            raise RuntimeError("RefactoringMiner not available")
        cmd = ['java', '-jar', str(self.refminer_jar)] + args
        return subprocess.run(
            cmd,
            cwd=str(self.refminer_root),
            capture_output=True,
            text=True,
            timeout=self.timeout
        )

    def _raw_json_path(self, commit_sha: str, repo_url: Optional[str] = None, repo_path: Optional[str] = None) -> Path:
        """Compute storage path for raw JSON per commit."""
        if repo_url and "github.com" in repo_url:
            try:
                parts = repo_url.rstrip('/').split('/')
                owner, repo = parts[-2], parts[-1]
                return (self.raw_json_dir / owner / repo / f"{commit_sha}.json").resolve()
            except Exception:
                pass
        if repo_path:
            safe = Path(repo_path).name
            return (self.raw_json_dir / safe / f"{commit_sha}.json").resolve()
        return (self.raw_json_dir / f"{commit_sha}.json").resolve()

    def analyze_commit_local(self, repo_path: str, commit_sha: str) -> Optional[Dict]:
        """Analyze a single commit using a locally cloned repository (-c)."""
        if not self.is_available():
            return None

        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            args = ['-c', repo_path, commit_sha, '-json', tmp_path]
            result = self._run_rm(args)

            if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path, 'r') as f:
                    data = json.load(f)
                # persist raw JSON if requested
                if self.save_json:
                    out_path = self._raw_json_path(commit_sha, repo_path=repo_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        # move/copy the temp file
                        shutil.move(tmp_path, out_path)
                        tmp_path = None  # prevent cleanup
                    except Exception:
                        # fallback: write content
                        with open(out_path, 'w') as wf:
                            json.dump(data, wf, indent=2)
                return data
            else:
                err = result.stderr.strip()
                if err:
                    print(f"RefactoringMiner (-c) failed for {commit_sha}: {err}")
                if result.stdout:
                    print(f"  stdout: {result.stdout}")
                return None
        except subprocess.TimeoutExpired:
            print(f"RefactoringMiner (-c) timed out for commit {commit_sha}")
            return None
        except Exception as e:
            print(f"Error (-c) analyzing commit {commit_sha}: {e}")
            return None
        finally:
            if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def analyze_commit_from_github(self, repo_url: str, commit_sha: str) -> Optional[Dict]:
        """
        Analyze a single commit from GitHub using RefactoringMiner
        
        Args:
            repo_url: GitHub repository URL
            commit_sha: Commit SHA to analyze
            
        Returns:
            Dictionary with refactoring results or None if failed
        """
        if not self.is_available():
            return None
        
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Run RefactoringMiner with GitHub URL (-gc option)
            args = ['-gc', repo_url, commit_sha, str(self.timeout), '-json', tmp_path]
            result = self._run_rm(args)
            
            if result.returncode == 0:
                # Check if output file exists and has content
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    # Read results
                    with open(tmp_path, 'r') as f:
                        refactoring_data = json.load(f)

                    # Debug output
                    if refactoring_data.get('commits') and len(refactoring_data['commits']) > 0:
                        refactorings_found = sum(len(commit.get('refactorings', [])) for commit in refactoring_data['commits'])
                        if refactorings_found > 0:
                            print(f"  → Found {refactorings_found} refactorings")

                    # Persist raw JSON if requested
                    if self.save_json:
                        out_path = self._raw_json_path(commit_sha, repo_url=repo_url)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.move(tmp_path, out_path)
                            tmp_path = None
                        except Exception:
                            with open(out_path, 'w') as wf:
                                json.dump(refactoring_data, wf, indent=2)

                    return refactoring_data
                else:
                    print(f"  → No output file generated (empty result)")
                    return None
            else:
                print(f"RefactoringMiner (-gc) failed for {commit_sha}: {result.stderr}")
                if result.stdout:
                    print(f"  stdout: {result.stdout}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"RefactoringMiner (-gc) timed out for commit {commit_sha}")
            return None
        except Exception as e:
            print(f"Error (-gc) analyzing commit {commit_sha}: {e}")
            return None
        finally:
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def analyze_commits_batch(self, commits_df: pd.DataFrame, max_commits: int = 100) -> pd.DataFrame:
        """
        Analyze multiple commits using RefactoringMiner
        
        Args:
            commits_df: DataFrame with commit information (must have 'sha' and 'html_url' columns)
            max_commits: Maximum number of commits to analyze
            
        Returns:
            DataFrame with refactoring detection results
        """
        if not self.is_available():
            raise RuntimeError("RefactoringMiner not available")
        
        # Extract repository URLs from commits
        commits_to_analyze = commits_df.drop_duplicates('sha').head(max_commits).copy()
        results = []
        
        print(f"Analyzing {len(commits_to_analyze)} commits with RefactoringMiner...")
        
        for idx, row in commits_to_analyze.iterrows():
            commit_sha = row['sha']

            refactoring_result = None

            # Prefer local repo override if provided
            if self.local_repo_override and Path(self.local_repo_override).exists():
                print(f"Analyzing commit {commit_sha[:8]} with local repo (-c): {self.local_repo_override}")
                refactoring_result = self.analyze_commit_local(self.local_repo_override, commit_sha)
            else:
                # Extract repo URL from html_url if available
                if 'html_url' in row and pd.notna(row['html_url']) and 'github.com' in str(row['html_url']):
                    html_url = row['html_url']
                    parts = html_url.split('/')
                    if len(parts) >= 5:
                        repo_url = f"https://github.com/{parts[3]}/{parts[4]}"
                        print(f"Analyzing commit {commit_sha[:8]} from {repo_url} (-gc)")
                        refactoring_result = self.analyze_commit_from_github(repo_url, commit_sha)
                    else:
                        print(f"Skipping {commit_sha[:8]}: could not parse repo from html_url")
                        continue
                else:
                    print(f"Skipping {commit_sha[:8]}: missing html_url and no REFMINER_LOCAL_REPO set")
                    continue
            
            if refactoring_result:
                # Parse RefactoringMiner results
                commit_refactorings = self.parse_refactoring_result(refactoring_result, commit_sha)
                results.extend(commit_refactorings)
        
        # Convert to DataFrame
        refactoring_df = pd.DataFrame(results)
        return refactoring_df
    
    def parse_refactoring_result(self, refactoring_result: Dict, commit_sha: str) -> List[Dict]:
        """
        Parse RefactoringMiner JSON result
        
        Args:
            refactoring_result: Raw JSON result from RefactoringMiner
            commit_sha: Commit SHA being analyzed
            
        Returns:
            List of refactoring records
        """
        refactorings = []
        
        # RefactoringMiner output structure varies, but typically:
        # {"commits": [{"sha1": "...", "refactorings": [...]}]}
        
        if 'commits' in refactoring_result:
            for commit_data in refactoring_result['commits']:
                if commit_data.get('sha1') == commit_sha:
                    for refactoring in commit_data.get('refactorings', []):
                        refactorings.append({
                            'commit_sha': commit_sha,
                            'refactoring_type': refactoring.get('type'),
                            'description': refactoring.get('description'),
                            'left_side_locations': refactoring.get('leftSideLocations', []),
                            'right_side_locations': refactoring.get('rightSideLocations', []),
                            'refactoring_detail': json.dumps(refactoring)
                        })
        
        # Alternative structure: direct refactorings list
        elif 'refactorings' in refactoring_result:
            for refactoring in refactoring_result['refactorings']:
                refactorings.append({
                    'commit_sha': commit_sha,
                    'refactoring_type': refactoring.get('type'),
                    'description': refactoring.get('description'),
                    'left_side_locations': refactoring.get('leftSideLocations', []),
                    'right_side_locations': refactoring.get('rightSideLocations', []),
                    'refactoring_detail': json.dumps(refactoring)
                })
        
        return refactorings
    
    def run_full_refactoring_analysis(self, commits_df: pd.DataFrame, max_commits: int = 200) -> Dict:
        """
        Run complete refactoring analysis using RefactoringMiner
        
        Args:
            commits_df: DataFrame with Java commits
            max_commits: Maximum commits to analyze (for performance)
            
        Returns:
            Complete analysis results
        """
        print("=== Running RefactoringMiner Analysis ===")
        
        if not self.is_available():
            print("RefactoringMiner not available - falling back to pattern analysis")
            return None
        
        # Select commits for analysis (prioritize recent commits with agent info)
        analysis_commits = commits_df.copy()
        
        # Sort by agent presence and take a sample
        if 'agent' in analysis_commits.columns:
            analysis_commits = analysis_commits.sort_values(['agent'], na_position='last')
        
        analysis_commits = analysis_commits.head(max_commits)
        
        # Run RefactoringMiner
        refactoring_df = self.analyze_commits_batch(analysis_commits, max_commits)
        
        if refactoring_df.empty:
            print("No refactorings detected by RefactoringMiner")
            return None
        
        # Analyze results
        analysis = self.analyze_refminer_results(refactoring_df, analysis_commits)
        
        # Save results
        self.save_refminer_results(refactoring_df, analysis)
        
        return analysis
    
    def analyze_refminer_results(self, refactoring_df: pd.DataFrame, commits_df: pd.DataFrame) -> Dict:
        """Analyze RefactoringMiner results"""
        
        # Get refactoring types
        type_counts = refactoring_df['refactoring_type'].value_counts()
        
        # Get commits with refactoring
        refactoring_commits = refactoring_df['commit_sha'].nunique()
        total_commits = commits_df['sha'].nunique()
        
        analysis = {
            'refactoring_miner_results': {
                'total_commits_analyzed': total_commits,
                'commits_with_refactoring': refactoring_commits,
                'refactoring_rate': (refactoring_commits / total_commits * 100) if total_commits > 0 else 0,
                'total_refactorings_found': len(refactoring_df),
                'refactoring_types': type_counts.head(10).to_dict(),
                'most_common_refactoring': type_counts.index[0] if len(type_counts) > 0 else None
            }
        }
        
        return analysis
    
    def save_refminer_results(self, refactoring_df: pd.DataFrame, analysis: Dict):
        """Save RefactoringMiner results"""
        
        # Save detailed refactorings
        refactoring_df.to_parquet(self.output_dir / "refminer_refactorings.parquet", index=False)
        refactoring_df.to_csv(self.output_dir / "refminer_refactorings.csv", index=False)
        
        # Save analysis
        with open(self.output_dir / "refminer_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"RefactoringMiner results saved to {self.output_dir}")
        print(f"Found {len(refactoring_df)} refactorings in {refactoring_df['commit_sha'].nunique()} commits")
