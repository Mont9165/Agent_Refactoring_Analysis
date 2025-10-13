"""Compute before/after Designite metric deltas for refactoring entities."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
import logging
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TYPE_METRIC_COLUMNS = [
    "LOC",
    "WMC",
    "NOM",
    "NOF",
    "NOPM",
    "NOPF",
    "NC",
    "DIT",
    "LCOM",
    "FANIN",
    "FANOUT",
]
METHOD_METRIC_COLUMNS = ["LOC", "CC", "PC"]
DESIGN_SMELL_NAMES = [
    "Imperative Abstraction",
    "Multifaceted Abstraction",
    "Unnecessary Abstraction",
    "Unutilized Abstraction",
    "Deficient Encapsulation",
    "Unexploited Encapsulation",
    "Broken Modularization",
    "Cyclic-Dependent Modularization",
    "Insufficient Modularization",
    "Hub-like Modularization",
    "Broken Hierarchy",
    "Cyclic Hierarchy",
    "Deep Hierarchy",
    "Missing Hierarchy",
    "Multipath Hierarchy",
    "Rebellious Hierarchy",
    "Wide Hierarchy",
]
IMPLEMENTATION_SMELL_NAMES = [
    "Abstract Function Call From Constructor",
    "Complex Conditional",
    "Complex Method",
    "Empty catch clause",
    "Long Identifier",
    "Long Method",
    "Long Parameter List",
    "Long Statement",
    "Magic Number",
    "Missing default",
]
DESIGN_SMELL_FILE_CANDIDATES = (
    "designCodeSmells.csv",
    "DesignSmells.csv",
    "designSmells.csv",
)
IMPLEMENTATION_SMELL_FILE_CANDIDATES = (
    "implementationCodeSmells.csv",
    "ImplementationSmells.csv",
    "implementationSmells.csv",
)
SMELL_COLUMN_CANDIDATES = ("Code Smell", "Smell", "Smell Name")
SMELL_REF_TYPE = "__commit__"

logger = logging.getLogger(__name__)

def _resolve_designite_root() -> Path:
    env_root = os.environ.get("DESIGNITE_OUTPUT_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if env_path.exists():
            return env_path

    candidates = [
        Path("tools/DesigniteRunner/outputs"),
        Path("data/designite/outputs"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DESIGNITE_OUTPUT_ROOT = _resolve_designite_root()
DELTA_Output_DIR = Path("data/analysis/designite/deltas")


@dataclass
class ToolConfig:
    designite_path: Optional[str]
    repos_base: Optional[Path]
    local_repo: Optional[Path]
    auto_generate_outputs: bool
    auto_clone_repos: bool
    git_remote_template: str
    max_retries: int
    timeout_seconds: int


@dataclass
class RepoCommit:
    owner: str
    repo: str
    sha: str
    parent_sha: str

    @property
    def child_output_dir(self) -> Path:
        return DESIGNITE_OUTPUT_ROOT / self.owner / self.repo / self.sha

    @property
    def parent_output_dir(self) -> Path:
        return DESIGNITE_OUTPUT_ROOT / self.owner / self.repo / self.parent_sha


@dataclass
class EntityRecord:
    commit_sha: str
    refactoring_type: str
    entity_kind: str  # "type" or "method"
    before_key: Optional[str]
    after_key: Optional[str]
    before_file_path: Optional[str]
    after_file_path: Optional[str]


class DesigniteDeltaCalculator:
    def __init__(
        self,
        config: ToolConfig,
        max_commits: Optional[int] = None,
        persist_outputs: bool = True,
    ) -> None:
        self.config = config
        self.max_commits = max_commits
        DELTA_Output_DIR.mkdir(parents=True, exist_ok=True)
        self.persist_outputs = persist_outputs

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------
    def _run(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 600) -> Tuple[int, str, str]:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def _infer_repo_path(self, owner: str, repo: str) -> Optional[Path]:
        if self.config.local_repo and self.config.local_repo.exists():
            return self.config.local_repo
        if not self.config.repos_base:
            return None
        candidate = self.config.repos_base / owner / repo
        if candidate.exists():
            return candidate
        if not self.config.auto_clone_repos:
            return None
        return self._clone_repo(owner, repo)

    def _clone_repo(self, owner: str, repo: str) -> Optional[Path]:
        if not self.config.repos_base:
            return None
        destination = self.config.repos_base / owner / repo
        destination.parent.mkdir(parents=True, exist_ok=True)
        remote = self.config.git_remote_template.format(owner=owner, repo=repo)
        logger.info("Cloning %s → %s", remote, destination)
        code, out, err = self._run(["git", "clone", remote, str(destination)])
        if code != 0:
            logger.error("git clone failed for %s/%s: %s", owner, repo, err.strip() or out.strip())
            return None
        return destination if destination.exists() else None

    def _ensure_commit_available(self, repo_path: Path, sha: str) -> bool:
        code, _, _ = self._run(["git", "rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=repo_path)
        if code == 0:
            return True
        logger.info("Fetching commit %s for %s", sha[:10], repo_path.name)
        fetch_cmd = ["git", "fetch", "origin", sha]
        code, out, err = self._run(fetch_cmd, cwd=repo_path)
        if code != 0:
            logger.error(
                "git fetch failed for %s (%s): %s",
                repo_path.name,
                sha[:10],
                err.strip() or out.strip(),
            )
            return False
        return True

    def _get_parent_sha(self, repo_path: Path, sha: str) -> Optional[str]:
        if not self._ensure_commit_available(repo_path, sha):
            return None
        code, out, _ = self._run(["git", "rev-parse", f"{sha}^"] , cwd=repo_path)
        if code == 0:
            return out.strip()
        return None

    def _checkout_worktree(self, repo_path: Path, sha: str) -> Optional[Path]:
        tmp_root = Path(tempfile.mkdtemp(prefix="designite_delta_"))
        worktree = tmp_root / sha[:10]
        code, _, _ = self._run(["git", "worktree", "add", str(worktree), sha], cwd=repo_path)
        if code != 0:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None
        return worktree

    def _cleanup_worktree(self, repo_path: Path, worktree: Path) -> None:
        if not worktree:
            return
        self._run(["git", "worktree", "remove", str(worktree), "--force"], cwd=repo_path)
        shutil.rmtree(worktree.parent, ignore_errors=True)

    # ------------------------------------------------------------------
    # Designite helpers
    # ------------------------------------------------------------------
    def _run_designite(self, input_dir: Path, output_dir: Path) -> bool:
        if not self.config.designite_path:
            return False
        designite_path = Path(self.config.designite_path)
        if not designite_path.exists():
            logger.error("Designite jar not found at %s", designite_path)
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["java", "-jar", str(designite_path), "-i", str(input_dir), "-o", str(output_dir)]

        for attempt in range(1, self.config.max_retries + 1):
            code, out, err = self._run(cmd, timeout=self.config.timeout_seconds)
            if code == 0:
                return True
            logger.warning(
                "Designite run failed (attempt %d/%d) for %s: %s",
                attempt,
                self.config.max_retries,
                input_dir,
                err.strip() or out.strip(),
            )
        return False

    def ensure_designite_outputs(self, repo_commit: RepoCommit) -> Tuple[Optional[Path], Optional[Path]]:
        child_dir = repo_commit.child_output_dir
        parent_dir = repo_commit.parent_output_dir

        logger.debug(
            "Checking Designite outputs for %s/%s@%s",
            repo_commit.owner,
            repo_commit.repo,
            repo_commit.sha[:10],
        )
        logger.debug("Child output dir: %s", child_dir)
        logger.debug("Parent output dir: %s", parent_dir)
        logger.debug("Child CSV exists? %s", (child_dir / "typeMetrics.csv").exists())
        logger.debug("Parent CSV exists? %s", (parent_dir / "typeMetrics.csv").exists())

        child_ready = (child_dir / "typeMetrics.csv").exists()
        parent_ready = (parent_dir / "typeMetrics.csv").exists()

        repo_path = None
        if not (child_ready and parent_ready):
            repo_path = self._infer_repo_path(repo_commit.owner, repo_commit.repo)
            if not repo_path:
                return (parent_dir if parent_ready else None, child_dir if child_ready else None)

            if not child_ready and not self._ensure_commit_available(repo_path, repo_commit.sha):
                return parent_dir, child_dir
            if not parent_ready and repo_commit.parent_sha and not self._ensure_commit_available(repo_path, repo_commit.parent_sha):
                return parent_dir, child_dir

        if self.config.auto_generate_outputs:
            if not child_ready:
                child_worktree = self._checkout_worktree(repo_path, repo_commit.sha)
                if child_worktree:
                    try:
                        success = self._run_designite(child_worktree, child_dir)
                        if success:
                            child_ready = True
                            child_dir = repo_commit.child_output_dir
                        else:
                            logger.warning(
                                "Designite failed for child commit %s in %s/%s",
                                repo_commit.sha[:10],
                                repo_commit.owner,
                                repo_commit.repo,
                            )
                            child_dir = None
                    finally:
                        self._cleanup_worktree(repo_path, child_worktree)
                else:
                    logger.error(
                        "Could not create worktree for child commit %s in %s/%s",
                        repo_commit.sha[:10],
                        repo_commit.owner,
                        repo_commit.repo,
                    )
                    child_dir = None

            if not parent_ready and repo_commit.parent_sha:
                parent_worktree = self._checkout_worktree(repo_path, repo_commit.parent_sha)
                if parent_worktree:
                    try:
                        success = self._run_designite(parent_worktree, parent_dir)
                        if success:
                            parent_ready = True
                            parent_dir = repo_commit.parent_output_dir
                        else:
                            logger.warning(
                                "Designite failed for parent commit %s in %s/%s",
                                repo_commit.parent_sha[:10],
                                repo_commit.owner,
                                repo_commit.repo,
                            )
                            parent_dir = None
                    finally:
                        self._cleanup_worktree(repo_path, parent_worktree)
                else:
                    logger.error(
                        "Could not create worktree for parent commit %s in %s/%s",
                        repo_commit.parent_sha[:10],
                        repo_commit.owner,
                        repo_commit.repo,
                    )
                    parent_dir = None

        return parent_dir, child_dir

    # ------------------------------------------------------------------
    # Designite parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _read_csv(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception:
            return None

    @staticmethod
    def _normalize_package(path: str) -> Optional[str]:
        if not path:
            return None
        normalized = path.replace("\\", "/")
        if normalized.endswith(".java"):
            normalized = normalized[:-5]
        markers = [
            "/src/main/java/",
            "/src/test/java/",
            "/src/java/",
            "/app/src/main/java/",
            "/app/src/test/java/",
            "/java/",
        ]
        pkg_path = None
        for marker in markers:
            if marker in normalized:
                pkg_path = normalized.split(marker, 1)[1]
                break
        if pkg_path is None:
            parts = normalized.split("/")
            if len(parts) > 1:
                pkg_path = "/".join(parts[:-1])
            else:
                pkg_path = ""
        else:
            parts = pkg_path.split("/")
            pkg_path = "/".join(parts[:-1])
        if not pkg_path:
            return None
        return pkg_path.replace("/", ".")

    @staticmethod
    def _extract_type_name(code_element: Optional[str], file_path: str) -> Optional[str]:
        if code_element:
            clean = code_element.strip()
            if " " not in clean and "(" not in clean:
                if "." in clean:
                    return clean.split(".")[-1]
                return clean.split("<")[0]
            # handle tokens like "public class Foo"
            tokens = re.split(r"\s+", clean)
            if tokens:
                candidate = tokens[-1]
                candidate = candidate.split("(")[0]
                candidate = candidate.split("<")[0]
                if candidate:
                    return candidate
        # fallback to filename
        filename = Path(file_path).name
        if filename.endswith(".java"):
            return filename[:-5]
        return Path(file_path).stem or None

    @staticmethod
    def _extract_method_name(code_element: Optional[str]) -> Optional[str]:
        if not code_element:
            return None
        before_paren = code_element.split("(")[0]
        tokens = re.split(r"\s+", before_paren.strip())
        if not tokens:
            return None
        candidate = tokens[-1]
        candidate = candidate.replace("<", "").replace(">", "")
        if not candidate:
            return None
        if candidate in {"public", "private", "protected", "static", "final", "native", "synchronized"}:
            return None
        return candidate

    @classmethod
    def load_type_metrics(cls, output_dir: Path) -> Optional[pd.DataFrame]:
        df = cls._read_csv(output_dir / "typeMetrics.csv")
        if df is None or df.empty:
            return None
        df = df.copy()
        df["Package Name"] = df["Package Name"].astype(str).fillna("")
        df["Type Name"] = df["Type Name"].astype(str)
        df["type_key"] = df.apply(
            lambda r: f"{r['Package Name']}.{r['Type Name']}" if r['Package Name'] else r['Type Name'],
            axis=1,
        )
        return df

    @classmethod
    def load_method_metrics(cls, output_dir: Path) -> Optional[pd.DataFrame]:
        df = cls._read_csv(output_dir / "methodMetrics.csv")
        if df is None or df.empty:
            return None
        df = df.copy()
        df["Package Name"] = df.get("Package Name", "").astype(str)
        df["Type Name"] = df["Type Name"].astype(str)
        df["MethodName"] = df["MethodName"].astype(str)
        df["method_key"] = df.apply(
            lambda r: "".join(filter(None, [
                f"{r['Package Name']}." if r['Package Name'] else "",
                f"{r['Type Name']}.",
                r['MethodName'],
            ])),
            axis=1,
        )
        return df

    @classmethod
    def _load_smell_frame(cls, output_dir: Optional[Path], candidates: Tuple[str, ...], fallback_glob: str) -> Optional[pd.DataFrame]:
        if output_dir is None:
            return None
        for candidate in candidates:
            df = cls._read_csv(output_dir / candidate)
            if df is not None:
                return df
        for path in output_dir.glob(fallback_glob):
            df = cls._read_csv(path)
            if df is not None:
                return df
        return None

    @classmethod
    def load_design_smells(cls, output_dir: Optional[Path]) -> Optional[pd.DataFrame]:
        return cls._load_smell_frame(output_dir, DESIGN_SMELL_FILE_CANDIDATES, "**/*DesignSmells*.csv")

    @classmethod
    def load_implementation_smells(cls, output_dir: Optional[Path]) -> Optional[pd.DataFrame]:
        return cls._load_smell_frame(output_dir, IMPLEMENTATION_SMELL_FILE_CANDIDATES, "**/*ImplementationSmells*.csv")

    @staticmethod
    def _smell_counts(smell_df: Optional[pd.DataFrame], smell_names: Iterable[str]) -> pd.Series:
        base = pd.Series({name: 0.0 for name in smell_names}, dtype=float)
        if smell_df is None or smell_df.empty:
            return base
        smell_column = next((col for col in SMELL_COLUMN_CANDIDATES if col in smell_df.columns), None)
        if not smell_column:
            return base
        counts = (
            smell_df[smell_column]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
        )
        for name in base.index:
            base.at[name] = float(counts.get(name, 0))
        return base

    # ------------------------------------------------------------------
    # Refactoring entity extraction
    # ------------------------------------------------------------------
    @classmethod
    def _build_entity_key(cls, file_path: Optional[str], code_element: Optional[str], code_type: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not file_path:
            return None, None, None
        package = cls._normalize_package(file_path)
        entity_kind = None
        simple_type = None
        method_name = None

        if code_type in {"TYPE_DECLARATION", "ENUM_DECLARATION", "TYPE"}:
            entity_kind = "type"
            simple_type = cls._extract_type_name(code_element, file_path)
        elif code_type in {"METHOD_DECLARATION", "CONSTRUCTOR_DECLARATION"}:
            entity_kind = "method"
            simple_type = cls._extract_type_name(None, file_path)
            method_name = cls._extract_method_name(code_element)
        else:
            return None, None, None

        if entity_kind == "type":
            if not simple_type:
                return None, None, None
            key = f"{package}.{simple_type}" if package else simple_type
            return entity_kind, key, None
        if entity_kind == "method":
            if not simple_type or not method_name:
                return None, None, None
            base = f"{package}.{simple_type}" if package else simple_type
            key = f"{base}.{method_name}"
            return entity_kind, key, base
        return None, None, None

    def extract_entities_for_commit(self, refs: pd.DataFrame) -> List[EntityRecord]:
        records: List[EntityRecord] = []
        def _decode_locations(raw):
            if isinstance(raw, str):
                try:
                    return json.loads(raw or "[]")
                except json.JSONDecodeError:
                    return []
            if isinstance(raw, (list, tuple)):
                return raw
            if isinstance(raw, np.ndarray):
                return raw.tolist()
            return []
        for _, row in refs.iterrows():
            left_locs = _decode_locations(row.get("left_side_locations", []))
            right_locs = _decode_locations(row.get("right_side_locations", []))
            for left, right in zip_longest(left_locs, right_locs, fillvalue=None):
                loc = right or left
                if not loc:
                    continue
                entity_kind, after_key, _ = self._build_entity_key(
                    right.get("filePath") if right else None,
                    right.get("codeElement") if right else None,
                    right.get("codeElementType") if right else None,
                ) if right else (None, None, None)
                before_kind, before_key, _ = self._build_entity_key(
                    left.get("filePath") if left else None,
                    left.get("codeElement") if left else None,
                    left.get("codeElementType") if left else None,
                ) if left else (None, None, None)

                target_kind = entity_kind or before_kind
                if not target_kind:
                    continue

                records.append(
                    EntityRecord(
                        commit_sha=row["commit_sha"],
                        refactoring_type=row["refactoring_type"],
                        entity_kind=target_kind,
                        before_key=before_key,
                        after_key=after_key,
                        before_file_path=left.get("filePath") if left else None,
                        after_file_path=right.get("filePath") if right else None,
                    )
                )
        return records

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metric_delta(
        before_row: pd.Series,
        after_row: pd.Series,
        metrics: Iterable[str],
        commit_sha: str,
        ref_type: str,
        entity_kind: str,
        before_key: Optional[str],
        after_key: Optional[str],
        parent_sha: str,
        child_sha: str,
    ) -> List[Dict[str, object]]:
        deltas: List[Dict[str, object]] = []
        for metric in metrics:
            b_val = float(before_row.get(metric)) if metric in before_row else None
            a_val = float(after_row.get(metric)) if metric in after_row else None
            if b_val is None or a_val is None:
                continue
            deltas.append(
                {
                    "commit_sha": commit_sha,
                    "child_sha": child_sha,
                    "parent_sha": parent_sha,
                    "refactoring_type": ref_type,
                    "entity_kind": entity_kind,
                    "metric": metric,
                    "before_key": before_key,
                    "after_key": after_key,
                    "before_value": b_val,
                    "after_value": a_val,
                    "delta": a_val - b_val,
                }
            )
        return deltas

    def compute_commit_deltas(
        self,
        repo_commit: RepoCommit,
        type_entities: List[EntityRecord],
        method_entities: List[EntityRecord],
    ) -> Tuple[
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
    ]:
        parent_dir, child_dir = repo_commit.parent_output_dir, repo_commit.child_output_dir
        parent_types = self.load_type_metrics(parent_dir)
        child_types = self.load_type_metrics(child_dir)
        parent_methods = self.load_method_metrics(parent_dir)
        child_methods = self.load_method_metrics(child_dir)
        parent_design_smells = self.load_design_smells(parent_dir)
        child_design_smells = self.load_design_smells(child_dir)
        parent_impl_smells = self.load_implementation_smells(parent_dir)
        child_impl_smells = self.load_implementation_smells(child_dir)

        type_deltas: List[Dict[str, object]] = []
        method_deltas: List[Dict[str, object]] = []
        design_smell_deltas: List[Dict[str, object]] = []
        implementation_smell_deltas: List[Dict[str, object]] = []

        type_map_parent = {row["type_key"]: row for _, row in parent_types.iterrows()} if parent_types is not None else {}
        type_map_child = {row["type_key"]: row for _, row in child_types.iterrows()} if child_types is not None else {}

        method_map_parent = {row["method_key"]: row for _, row in parent_methods.iterrows()} if parent_methods is not None else {}
        method_map_child = {row["method_key"]: row for _, row in child_methods.iterrows()} if child_methods is not None else {}

        for record in type_entities:
            after_key = record.after_key or record.before_key
            before_key = record.before_key or record.after_key
            if not after_key or not before_key:
                continue
            before_row = type_map_parent.get(before_key)
            after_row = type_map_child.get(after_key)
            if before_row is None or after_row is None:
                continue
            type_deltas.extend(
                self._compute_metric_delta(
                    before_row,
                    after_row,
                    TYPE_METRIC_COLUMNS,
                    record.commit_sha,
                    record.refactoring_type,
                    record.entity_kind,
                    before_key,
                    after_key,
                    repo_commit.parent_sha,
                    repo_commit.sha,
                )
            )

        for record in method_entities:
            after_key = record.after_key or record.before_key
            before_key = record.before_key or record.after_key
            if not after_key or not before_key:
                continue
            before_row = method_map_parent.get(before_key)
            after_row = method_map_child.get(after_key)
            if before_row is None or after_row is None:
                continue
            method_deltas.extend(
                self._compute_metric_delta(
                    before_row,
                    after_row,
                    METHOD_METRIC_COLUMNS,
                    record.commit_sha,
                    record.refactoring_type,
                    record.entity_kind,
                    before_key,
                    after_key,
                    repo_commit.parent_sha,
                    repo_commit.sha,
                )
            )

        parent_design_counts = self._smell_counts(parent_design_smells, DESIGN_SMELL_NAMES)
        child_design_counts = self._smell_counts(child_design_smells, DESIGN_SMELL_NAMES)
        design_smell_deltas.extend(
            self._compute_metric_delta(
                parent_design_counts,
                child_design_counts,
                DESIGN_SMELL_NAMES,
                repo_commit.sha,
                SMELL_REF_TYPE,
                "design_smell",
                None,
                None,
                repo_commit.parent_sha,
                repo_commit.sha,
            )
        )

        parent_impl_counts = self._smell_counts(parent_impl_smells, IMPLEMENTATION_SMELL_NAMES)
        child_impl_counts = self._smell_counts(child_impl_smells, IMPLEMENTATION_SMELL_NAMES)
        implementation_smell_deltas.extend(
            self._compute_metric_delta(
                parent_impl_counts,
                child_impl_counts,
                IMPLEMENTATION_SMELL_NAMES,
                repo_commit.sha,
                SMELL_REF_TYPE,
                "implementation_smell",
                None,
                None,
                repo_commit.parent_sha,
                repo_commit.sha,
            )
        )

        return type_deltas, method_deltas, design_smell_deltas, implementation_smell_deltas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, commits_df: pd.DataFrame, refminer_df: pd.DataFrame) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        commits = commits_df[commits_df.get("has_refactoring", False) == True].copy()
        if self.max_commits is not None:
            commits = commits.head(self.max_commits)

        all_type_deltas: List[Dict[str, object]] = []
        all_method_deltas: List[Dict[str, object]] = []
        all_design_smell_deltas: List[Dict[str, object]] = []
        all_impl_smell_deltas: List[Dict[str, object]] = []
        missing_designite: List[str] = []
        missing_repos: List[str] = []
        missing_parent_git: List[str] = []

        for _, commit in commits.iterrows():
            html_url = commit.get("html_url")
            if not isinstance(html_url, str):
                continue
            parts = html_url.split("/")
            if len(parts) < 5:
                continue
            owner, repo = parts[3], parts[4]
            repo_path = self._infer_repo_path(owner, repo)
            if not repo_path:
                logger.warning(
                    "Repository clone not found for %s/%s; skipping commit %s",
                    owner,
                    repo,
                    str(commit["sha"])[:10],
                )
                missing_repos.append(f"{owner}/{repo}:{commit['sha']}")
                continue
            parent_sha = self._get_parent_sha(repo_path, commit["sha"])
            if not parent_sha:
                missing_parent_git.append(f"{owner}/{repo}:{commit['sha']}")
                continue

            repo_commit = RepoCommit(owner=owner, repo=repo, sha=commit["sha"], parent_sha=parent_sha)
            parent_dir, child_dir = self.ensure_designite_outputs(repo_commit)
            if not parent_dir or not child_dir:
                missing_designite.append(f"{owner}/{repo}:{commit['sha']} (parent: {parent_sha})")
                continue

            commit_refs = refminer_df[refminer_df["commit_sha"] == commit["sha"]]
            if commit_refs.empty:
                continue
            entities = self.extract_entities_for_commit(commit_refs)
            type_entities = [e for e in entities if e.entity_kind == "type"]
            method_entities = [e for e in entities if e.entity_kind == "method"]
            type_deltas, method_deltas, design_smell_deltas, impl_smell_deltas = self.compute_commit_deltas(
                repo_commit,
                type_entities,
                method_entities,
            )
            all_type_deltas.extend(type_deltas)
            all_method_deltas.extend(method_deltas)
            all_design_smell_deltas.extend(design_smell_deltas)
            all_impl_smell_deltas.extend(impl_smell_deltas)

        if missing_designite:
            logger.warning("Designite outputs missing for commits:")
            for entry in missing_designite[:20]:
                logger.warning("  - %s", entry)
            if len(missing_designite) > 20:
                logger.warning("  ... and %d more", len(missing_designite) - 20)
        if missing_repos:
            logger.warning("Repository clones not found for commits:")
            for entry in missing_repos[:20]:
                logger.warning("  - %s", entry)
            if len(missing_repos) > 20:
                logger.warning("  ... and %d more", len(missing_repos) - 20)
        if missing_parent_git:
            logger.warning("Parent commit not reachable via git for commits:")
            for entry in missing_parent_git[:20]:
                logger.warning("  - %s", entry)
            if len(missing_parent_git) > 20:
                logger.warning("  ... and %d more", len(missing_parent_git) - 20)

        type_df = pd.DataFrame(all_type_deltas)
        method_df = pd.DataFrame(all_method_deltas)
        design_smell_df = pd.DataFrame(all_design_smell_deltas)
        implementation_smell_df = pd.DataFrame(all_impl_smell_deltas)

        if self.persist_outputs:
            if not type_df.empty:
                type_df.to_parquet(DELTA_Output_DIR / "type_metric_deltas.parquet", index=False)
                type_df.to_csv(DELTA_Output_DIR / "type_metric_deltas.csv", index=False)
            if not method_df.empty:
                method_df.to_parquet(DELTA_Output_DIR / "method_metric_deltas.parquet", index=False)
                method_df.to_csv(DELTA_Output_DIR / "method_metric_deltas.csv", index=False)
            if not design_smell_df.empty:
                design_smell_df.to_parquet(DELTA_Output_DIR / "design_smell_deltas.parquet", index=False)
                design_smell_df.to_csv(DELTA_Output_DIR / "design_smell_deltas.csv", index=False)
            if not implementation_smell_df.empty:
                implementation_smell_df.to_parquet(DELTA_Output_DIR / "implementation_smell_deltas.parquet", index=False)
                implementation_smell_df.to_csv(DELTA_Output_DIR / "implementation_smell_deltas.csv", index=False)

        return type_df, method_df, design_smell_df, implementation_smell_df


def load_tool_config() -> ToolConfig:
    designite_default = Path("tools/DesigniteRunner/DesigniteJava.jar")
    designite_path = os.environ.get("DESIGNITE_JAVA_PATH")
    if not designite_path and designite_default.exists():
        designite_path = str(designite_default.resolve())
    default_repos_base = Path("tools/DesigniteRunner/cloned_repos")
    env_repos_base = os.environ.get("REPOS_BASE")
    if env_repos_base:
        repos_base = Path(env_repos_base).expanduser().resolve()
    elif default_repos_base.exists():
        repos_base = default_repos_base.resolve()
    else:
        repos_base = None
    local_repo = Path(os.environ["REFMINER_LOCAL_REPO"]) if os.environ.get("REFMINER_LOCAL_REPO") else None
    auto_generate = os.environ.get("DESIGNITE_AUTO", "1") not in {"0", "false", "False"}
    auto_clone = os.environ.get("DESIGNITE_AUTO_CLONE", "1") not in {"0", "false", "False"}
    git_remote_template = os.environ.get("DESIGNITE_GIT_REMOTE_TMPL", "https://github.com/{owner}/{repo}.git")
    max_retries = int(os.environ.get("DESIGNITE_MAX_RETRIES", "3"))
    return ToolConfig(
        designite_path=designite_path,
        repos_base=repos_base,
        local_repo=local_repo,
        auto_generate_outputs=auto_generate,
        auto_clone_repos=auto_clone,
        git_remote_template=git_remote_template,
        max_retries=max(1, max_retries),
        timeout_seconds=max(60, int(os.environ.get("DESIGNITE_TIMEOUT", "600"))),
    )


def aggregate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby(["entity_kind", "refactoring_type", "metric"])
    summary = grouped["delta"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    summary.to_parquet(DELTA_Output_DIR / "delta_summary.parquet", index=False)
    summary.to_csv(DELTA_Output_DIR / "delta_summary.csv", index=False)
    return summary
