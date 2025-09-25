"""Compute before/after Designite metric deltas for refactoring entities."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


TYPE_METRIC_COLUMNS = ["LOC", "WMC", "NOM", "NOF", "DIT", "LCOM", "FANIN", "FANOUT"]
METHOD_METRIC_COLUMNS = ["LOC", "CC", "PC"]

DESIGNITE_OUTPUT_ROOT = Path("data/designite/outputs")
DELTA_Output_DIR = Path("data/analysis/designite/deltas")


@dataclass
class ToolConfig:
    designite_path: Optional[str]
    repos_base: Optional[Path]
    local_repo: Optional[Path]


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
    def __init__(self, config: ToolConfig, max_commits: Optional[int] = None) -> None:
        self.config = config
        self.max_commits = max_commits
        DELTA_Output_DIR.mkdir(parents=True, exist_ok=True)

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
        return candidate if candidate.exists() else None

    def _get_parent_sha(self, repo_path: Path, sha: str) -> Optional[str]:
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
        if not self.config.designite_path or not Path(self.config.designite_path).exists():
            return False
        output_dir.mkdir(parents=True, exist_ok=True)
        code, _, _ = self._run(
            ["java", "-jar", self.config.designite_path, "-i", str(input_dir), "-o", str(output_dir)]
        )
        return code == 0

    def ensure_designite_outputs(self, repo_commit: RepoCommit) -> Tuple[Optional[Path], Optional[Path]]:
        repo_path = self._infer_repo_path(repo_commit.owner, repo_commit.repo)
        if not repo_path:
            return None, None

        child_dir = repo_commit.child_output_dir
        parent_dir = repo_commit.parent_output_dir

        if not (child_dir / "typeMetrics.csv").exists():
            child_worktree = self._checkout_worktree(repo_path, repo_commit.sha)
            if child_worktree:
                try:
                    success = self._run_designite(child_worktree, child_dir)
                    if not success:
                        child_dir = None
                finally:
                    self._cleanup_worktree(repo_path, child_worktree)
            else:
                child_dir = None

        if not (parent_dir / "typeMetrics.csv").exists():
            parent_worktree = self._checkout_worktree(repo_path, repo_commit.parent_sha)
            if parent_worktree:
                try:
                    success = self._run_designite(parent_worktree, parent_dir)
                    if not success:
                        parent_dir = None
                finally:
                    self._cleanup_worktree(repo_path, parent_worktree)
            else:
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
        for _, row in refs.iterrows():
            left_locs = json.loads(row.get("left_side_locations", "[]") or "[]")
            right_locs = json.loads(row.get("right_side_locations", "[]") or "[]")
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
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        parent_dir, child_dir = repo_commit.parent_output_dir, repo_commit.child_output_dir
        parent_types = self.load_type_metrics(parent_dir)
        child_types = self.load_type_metrics(child_dir)
        parent_methods = self.load_method_metrics(parent_dir)
        child_methods = self.load_method_metrics(child_dir)

        type_deltas: List[Dict[str, object]] = []
        method_deltas: List[Dict[str, object]] = []

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
                )
            )

        return type_deltas, method_deltas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, commits_df: pd.DataFrame, refminer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        commits = commits_df[commits_df.get("has_refactoring", False) == True].copy()
        if self.max_commits is not None:
            commits = commits.head(self.max_commits)

        all_type_deltas: List[Dict[str, object]] = []
        all_method_deltas: List[Dict[str, object]] = []

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
                continue
            parent_sha = self._get_parent_sha(repo_path, commit["sha"])
            if not parent_sha:
                continue

            repo_commit = RepoCommit(owner=owner, repo=repo, sha=commit["sha"], parent_sha=parent_sha)
            parent_dir, child_dir = self.ensure_designite_outputs(repo_commit)
            if not parent_dir or not child_dir:
                continue

            commit_refs = refminer_df[refminer_df["commit_sha"] == commit["sha"]]
            if commit_refs.empty:
                continue
            entities = self.extract_entities_for_commit(commit_refs)
            type_entities = [e for e in entities if e.entity_kind == "type"]
            method_entities = [e for e in entities if e.entity_kind == "method"]
            if not type_entities and not method_entities:
                continue

            type_deltas, method_deltas = self.compute_commit_deltas(repo_commit, type_entities, method_entities)
            all_type_deltas.extend(type_deltas)
            all_method_deltas.extend(method_deltas)

        type_df = pd.DataFrame(all_type_deltas)
        method_df = pd.DataFrame(all_method_deltas)

        if not type_df.empty:
            type_df.to_parquet(DELTA_Output_DIR / "type_metric_deltas.parquet", index=False)
            type_df.to_csv(DELTA_Output_DIR / "type_metric_deltas.csv", index=False)
        if not method_df.empty:
            method_df.to_parquet(DELTA_Output_DIR / "method_metric_deltas.parquet", index=False)
            method_df.to_csv(DELTA_Output_DIR / "method_metric_deltas.csv", index=False)

        return type_df, method_df


def load_tool_config() -> ToolConfig:
    designite_default = Path("tools/DesigniteRunner/DesigniteJava.jar")
    designite_path = os.environ.get("DESIGNITE_JAVA_PATH")
    if not designite_path and designite_default.exists():
        designite_path = str(designite_default.resolve())
    repos_base = Path(os.environ["REPOS_BASE"]) if os.environ.get("REPOS_BASE") else None
    local_repo = Path(os.environ["REFMINER_LOCAL_REPO"]) if os.environ.get("REFMINER_LOCAL_REPO") else None
    return ToolConfig(
        designite_path=designite_path,
        repos_base=repos_base,
        local_repo=local_repo,
    )


def aggregate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby(["entity_kind", "refactoring_type", "metric"])
    summary = grouped["delta"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    summary.to_parquet(DELTA_Output_DIR / "delta_summary.parquet", index=False)
    summary.to_csv(DELTA_Output_DIR / "delta_summary.csv", index=False)
    return summary
