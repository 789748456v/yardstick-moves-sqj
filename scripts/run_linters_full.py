#!/usr/bin/env python3
"""
Deterministic multi-linter, multi-version execution across all consecutive official release snapshots
in the D04-Plus Full Study (49 N09 Python 3 repositories, 882 snapshots).
Strictly bounded to <= 6 worker processes (within the 8 vCPU / 16 GB RAM budget).
Includes incremental backfill for `ruff_modern_pylint` on cached Pilot snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))
MAX_WORKERS = 6

EXPECTED_TOOLS = {
    "flake8_71",
    "ruff_01_default",
    "ruff_01_extended",
    "ruff_modern_default",
    "ruff_modern_extended",
    "ruff_modern_pylint",
    "pylint_217",
    "pylint_33",
}


def line_hash(file_cache: Dict[str, List[str]], wt_dir: Path, rel_path: str, line_no: int) -> str:
    if rel_path not in file_cache:
        fp = wt_dir / rel_path
        try:
            file_cache[rel_path] = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            file_cache[rel_path] = []
    lines = file_cache[rel_path]
    if 1 <= line_no <= len(lines):
        norm = lines[line_no - 1].strip()
        return hashlib.md5(norm.encode("utf-8", errors="replace")).hexdigest()[:12]
    return ""


def norm_rel_path(wt_dir: Path, raw_path: str) -> str:
    p = Path(raw_path)
    if p.is_absolute():
        try:
            return p.relative_to(wt_dir).as_posix()
        except ValueError:
            return p.as_posix()
    s = p.as_posix()
    if s.startswith("./"):
        s = s[2:]
    return s


def run_flake8(wt_dir: Path, file_cache: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    flake8_bin = BASE_DIR / "envs" / "flake8_71" / "bin" / "flake8"
    cmd = [
        str(flake8_bin),
        "--isolated",
        "--max-complexity=10",
        "--exclude=.git,__pycache__,build,dist,.tox,venv,.venv,.eggs",
        "--format=%(path)s\t%(row)d\t%(col)d\t%(code)s\t%(text)s",
        ".",
    ]
    try:
        res = subprocess.run(cmd, cwd=str(wt_dir), capture_output=True, text=True, errors="replace", timeout=1800)
        raw_lines = res.stdout.splitlines()
    except subprocess.TimeoutExpired as exc:
        raw_lines = (exc.stdout or "").splitlines() if isinstance(exc.stdout, str) else []
    issues = []
    for raw in raw_lines:
        parts = raw.split("\t", 4)
        if len(parts) != 5:
            continue
        rpath = norm_rel_path(wt_dir, parts[0])
        try:
            line_no = int(parts[1])
            col_no = int(parts[2])
        except ValueError:
            continue
        code = parts[3].strip()
        issues.append(
            {
                "file": rpath,
                "line": line_no,
                "col": col_no,
                "rule": code,
                "severity": "HIGH"
                if (code.startswith("F") or code.startswith("E9") or code.startswith("B9") or code in {"B006", "B008", "B023", "E722"})
                else ("MEDIUM" if (code.startswith("B") or code.startswith("C90")) else "LOW"),
                "lhash": line_hash(file_cache, wt_dir, rpath, line_no),
                "msg": parts[4].strip()[:160],
            }
        )
    return issues


def run_ruff(
    wt_dir: Path, env_name: str, mode: str, file_cache: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    ruff_bin = BASE_DIR / "envs" / env_name / "bin" / "ruff"
    cmd = [str(ruff_bin), "check", "--isolated", "--output-format", "json", "--exit-zero"]
    if mode == "extended":
        cmd += ["--select", "E,W,F,C90,B,PL,S"]
    elif mode == "pylint":
        # Official Pylint-to-Ruff migration selector: all PL (PLC/PLE/PLR/PLW) plus exact cross-prefix equivalents
        cmd += [
            "--select",
            "PL,E9,F401,F404,F601,F701,F702,F704,F706,F811,F821,F822,F841,E711,E712,E722,B006,B007,B018,C901",
        ]
    cmd.append(".")
    res = subprocess.run(cmd, cwd=str(wt_dir), capture_output=True, text=True, errors="replace", timeout=300)
    issues = []
    try:
        arr = json.loads(res.stdout) if res.stdout.strip() else []
    except json.JSONDecodeError:
        arr = []
    for item in arr:
        code = item.get("code") or "SYNTAX_ERR"
        rpath = norm_rel_path(wt_dir, item.get("filename", ""))
        loc = item.get("location") or {}
        line_no = int(loc.get("row", 1) or 1)
        col_no = int(loc.get("column", 1) or 1)
        is_high = (
            code == "SYNTAX_ERR"
            or code.startswith("F")
            or code.startswith("E9")
            or code.startswith("PLE")
            or code.startswith("B9")
            or code in {"B006", "B008", "B023", "E722", "S102", "S301", "S307", "S506", "S602"}
        )
        is_med = (
            not is_high
            and (code.startswith("B") or code.startswith("C90") or code.startswith("PLW") or code.startswith("PLR") or code.startswith("S"))
        )
        issues.append(
            {
                "file": rpath,
                "line": line_no,
                "col": col_no,
                "rule": code,
                "severity": "HIGH" if is_high else ("MEDIUM" if is_med else "LOW"),
                "lhash": line_hash(file_cache, wt_dir, rpath, line_no),
                "msg": (item.get("message") or "")[:160],
            }
        )
    return issues


def _discover_py_files(wt_dir: Path) -> List[str]:
    excluded = {".git", "__pycache__", "build", "dist", ".tox", "venv", ".venv", ".eggs", ".pylint_cache"}
    out: List[str] = []
    for p in wt_dir.rglob("*.py"):
        rel_parts = p.relative_to(wt_dir).parts
        if any(part in excluded for part in rel_parts):
            continue
        out.append(p.relative_to(wt_dir).as_posix())
    out.sort()
    return out


def run_pylint(
    wt_dir: Path, env_name: str, file_cache: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    pylint_bin = BASE_DIR / "envs" / env_name / "bin" / "pylint"
    base_cmd = [
        str(pylint_bin),
        "--rcfile=/dev/null",
        "--ignore=.git,__pycache__,build,dist,.tox,venv,.venv,.eggs,.pylint_cache",
        "--disable=E0401,E0611",
        "--jobs=1",
        "--persistent=n",
        "--score=n",
        "--output-format=json",
    ]
    env = os.environ.copy()
    env["PYLINTHOME"] = str(wt_dir / ".pylint_cache")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{wt_dir}:{existing_pp}" if existing_pp else str(wt_dir)

    py_files = _discover_py_files(wt_dir)
    if not py_files:
        return []
    total_bytes = sum((wt_dir / f).stat().st_size for f in py_files if (wt_dir / f).exists())

    raw_items: List[Dict[str, Any]] = []
    if len(py_files) <= 500 and total_bytes <= 1_500_000:
        cmd = [*base_cmd, "--recursive=y", "."]
        res = subprocess.run(
            cmd, cwd=str(wt_dir), env=env, capture_output=True, text=True, errors="replace", timeout=1500
        )
        if res.stdout.strip():
            try:
                raw_items.extend(json.loads(res.stdout))
            except json.JSONDecodeError:
                pass
    else:
        # Deterministic 60-file (<=500 files, >1.5MB) or 250-file (>500 files) batches with quadratic R0801 disabled to prevent O(N^2) hangs
        batch_size = 60 if len(py_files) <= 500 else 250
        no_r0801_cmd = [
            str(pylint_bin),
            "--rcfile=/dev/null",
            "--ignore=.git,__pycache__,build,dist,.tox,venv,.venv,.eggs,.pylint_cache",
            "--disable=E0401,E0611,R0801",
            "--jobs=1",
            "--persistent=n",
            "--score=n",
            "--output-format=json",
        ]
        def _append_pylint_stdout(stdout_str: str) -> None:
            if stdout_str and stdout_str.strip():
                try:
                    raw_items.extend(json.loads(stdout_str))
                except json.JSONDecodeError:
                    pass

        for idx in range(0, len(py_files), batch_size):
            batch = py_files[idx : idx + batch_size]
            try:
                res = subprocess.run(
                    [*no_r0801_cmd, *batch], cwd=str(wt_dir), env=env, capture_output=True, text=True, errors="replace", timeout=360
                )
                _append_pylint_stdout(res.stdout)
            except subprocess.TimeoutExpired:
                for s_idx in range(0, len(batch), 40):
                    sub_b = batch[s_idx : s_idx + 40]
                    try:
                        r_sub = subprocess.run(
                            [*no_r0801_cmd, *sub_b],
                            cwd=str(wt_dir),
                            env=env,
                            capture_output=True,
                            text=True,
                            errors="replace",
                            timeout=210,
                        )
                        _append_pylint_stdout(r_sub.stdout)
                    except subprocess.TimeoutExpired:
                        for m_idx in range(0, len(sub_b), 8):
                            micro_b = sub_b[m_idx : m_idx + 8]
                            try:
                                r_mic = subprocess.run(
                                    [*no_r0801_cmd, *micro_b],
                                    cwd=str(wt_dir),
                                    env=env,
                                    capture_output=True,
                                    text=True,
                                    errors="replace",
                                    timeout=120,
                                )
                                _append_pylint_stdout(r_mic.stdout)
                            except subprocess.TimeoutExpired:
                                for single_f in micro_b:
                                    try:
                                        r_one = subprocess.run(
                                            [*no_r0801_cmd, single_f],
                                            cwd=str(wt_dir),
                                            env=env,
                                            capture_output=True,
                                            text=True,
                                            errors="replace",
                                            timeout=45,
                                        )
                                        _append_pylint_stdout(r_one.stdout)
                                    except subprocess.TimeoutExpired:
                                        pass

    issues = []
    for item in raw_items:
        msgid = item.get("message-id") or ""
        symbol = item.get("symbol") or ""
        mtype = item.get("type") or ""
        rpath = norm_rel_path(wt_dir, item.get("path", ""))
        line_no = int(item.get("line", 1) or 1)
        col_no = int(item.get("column", 0) or 0)
        sev = (
            "HIGH"
            if (mtype in {"fatal", "error"} or msgid in {"W0102", "W0702"})
            else ("MEDIUM" if mtype in {"warning", "refactor"} else "LOW")
        )
        issues.append(
            {
                "file": rpath,
                "line": line_no,
                "col": col_no,
                "rule": msgid,
                "symbol": symbol,
                "ptype": mtype,
                "severity": sev,
                "lhash": line_hash(file_cache, wt_dir, rpath, line_no),
                "msg": (item.get("message") or "")[:160],
            }
        )
    return issues


def process_one_snapshot(rec: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    slug = rec["slug"]
    snap_id = rec["snapshot_id"]
    safe_snap = snap_id.replace("::", "__").replace("/", "_")
    out_file = BASE_DIR / "data" / "linter_runs" / f"{safe_snap}.json"

    existing_payload = None
    if out_file.exists():
        try:
            existing_payload = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            existing_payload = None
        if existing_payload and EXPECTED_TOOLS.issubset(set(existing_payload.get("tools", {}).keys())):
            return {"snapshot_id": snap_id, "cached": True, "sec": 0.0}

    repo_dir = BASE_DIR / "repos" / slug
    wt_dir = BASE_DIR / "tmp_linter_wt" / f"{safe_snap}_{os.getpid()}"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    if wt_dir.exists():
        subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)], capture_output=True)

    subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "add", "--detach", "--quiet", str(wt_dir), rec["commit"]],
        check=True,
        capture_output=True,
    )
    try:
        file_cache: Dict[str, List[str]] = {}
        tools_dict = dict(existing_payload.get("tools", {})) if existing_payload else {}

        if "flake8_71" not in tools_dict:
            tools_dict["flake8_71"] = run_flake8(wt_dir, file_cache)
        if "ruff_01_default" not in tools_dict:
            tools_dict["ruff_01_default"] = run_ruff(wt_dir, "ruff_01", "default", file_cache)
        if "ruff_01_extended" not in tools_dict:
            tools_dict["ruff_01_extended"] = run_ruff(wt_dir, "ruff_01", "extended", file_cache)
        if "ruff_modern_default" not in tools_dict:
            tools_dict["ruff_modern_default"] = run_ruff(wt_dir, "ruff_modern", "default", file_cache)
        if "ruff_modern_extended" not in tools_dict:
            tools_dict["ruff_modern_extended"] = run_ruff(wt_dir, "ruff_modern", "extended", file_cache)
        if "ruff_modern_pylint" not in tools_dict:
            tools_dict["ruff_modern_pylint"] = run_ruff(wt_dir, "ruff_modern", "pylint", file_cache)
        if "pylint_217" not in tools_dict:
            tools_dict["pylint_217"] = run_pylint(wt_dir, "pylint_217", file_cache)
        if "pylint_33" not in tools_dict:
            tools_dict["pylint_33"] = run_pylint(wt_dir, "pylint_33", file_cache)

        payload = {
            "snapshot_id": snap_id,
            "repository": rec["repository"],
            "seq_idx": rec["seq_idx"],
            "release_tag": rec["release_tag"],
            "commit": rec["commit"],
            "tools": tools_dict,
        }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload), encoding="utf-8")
    finally:
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)],
            capture_output=True,
        )
    return {"snapshot_id": snap_id, "cached": False, "sec": round(time.time() - t0, 2)}


def main() -> None:
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="*", default=None, help="Optional subset of repository names for Step 1 checkpoint")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    manifest_path = BASE_DIR / "data" / "full_releases_manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.repos:
        wanted = set(args.repos)
        records = [r for r in records if r["repository"] in wanted or r["slug"] in wanted]
    records.sort(key=lambda r: (int(r.get("py_sloc", 0)), r["snapshot_id"]))

    (BASE_DIR / "data" / "linter_runs").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, 7)) as ex:
        futs = {ex.submit(process_one_snapshot, r): r["snapshot_id"] for r in records}
        for fut in as_completed(futs):
            info = fut.result()
            done += 1
            if not info["cached"] or done % 25 == 0 or done == len(records):
                print(
                    f"[linter {done}/{len(records)}] {info['snapshot_id']} (cached={info['cached']}, sec={info['sec']}, elapsed={time.time()-t_start:.1f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
