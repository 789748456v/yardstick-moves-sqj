#!/usr/bin/env python3
"""
Exact New-Code line interval extractor and N09-compatible code churn calculator
between consecutive release tags (r_{t-1} -> r_t) using `git diff -U0`.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def extract_new_code_lines_and_churn(
    repo_dir: Path, prev_commit: str, curr_commit: str
) -> Tuple[Dict[str, Set[int]], Dict[str, int]]:
    """
    Returns:
      - new_lines_by_file: mapping rel_path -> set of 1-based line numbers in curr_commit
        that were added or modified relative to prev_commit (matching `diff-quality` / CI PR gate semantics).
      - churn_stats: {
          'modified_py_files': int,
          'py_lines_added': int,
          'py_lines_deleted': int,
          'py_lines_changed': int,
          'all_modified_files': int,
          'all_lines_changed': int,
        }
    """
    # 1. Overall churn across repository (matching N09 RQ3 commit/release activity)
    numstat_cmd = ["git", "-C", str(repo_dir), "diff", "--numstat", prev_commit, curr_commit]
    res_num = subprocess.run(numstat_cmd, capture_output=True, text=True, errors="replace", check=True)
    all_files = 0
    all_changed = 0
    py_files = 0
    py_added = 0
    py_deleted = 0

    for raw in res_num.stdout.splitlines():
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        add_s, del_s, path_s = parts[0], parts[1], parts[2]
        all_files += 1
        a = int(add_s) if add_s.isdigit() else 0
        d = int(del_s) if del_s.isdigit() else 0
        all_changed += a + d
        if path_s.endswith(".py"):
            py_files += 1
            py_added += a
            py_deleted += d

    # 2. Exact line numbers modified/added in Python files (`git diff -U0`)
    diff_cmd = [
        "git",
        "-C",
        str(repo_dir),
        "diff",
        "-U0",
        "--diff-algorithm=histogram",
        prev_commit,
        curr_commit,
        "--",
        "*.py",
    ]
    res_diff = subprocess.run(diff_cmd, capture_output=True, text=True, errors="replace", check=True)

    new_lines_by_file: Dict[str, Set[int]] = {}
    current_file: str | None = None

    for line in res_diff.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            new_lines_by_file.setdefault(current_file, set())
        elif line.startswith("+++ /dev/null"):
            current_file = None
        elif line.startswith("@@ ") and current_file is not None:
            m = HUNK_RE.match(line)
            if m:
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                if new_count > 0:
                    new_lines_by_file[current_file].update(
                        range(new_start, new_start + new_count)
                    )

    churn_stats = {
        "modified_py_files": py_files,
        "py_lines_added": py_added,
        "py_lines_deleted": py_deleted,
        "py_lines_changed": py_added + py_deleted,
        "all_modified_files": all_files,
        "all_lines_changed": all_changed,
    }
    return new_lines_by_file, churn_stats
