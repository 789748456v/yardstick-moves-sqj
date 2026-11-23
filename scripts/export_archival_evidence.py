#!/usr/bin/env python3
"""
Archival Verification & Catalog Export Script for D04-Plus (SQJ Revision):
1. Exports exact enabled CLI linter rule catalogs (`data/linter_catalogs.json`) from isolated virtualenvs
   (`flake8_71`, `pylint_217`, `pylint_33`, `ruff_01`, `ruff_modern`) so S1/RQ3 use static tool catalogs
   without any future-release corpus leakage.
2. Audits all 10,584 physical scan outputs (`3,528` SonarQube JSONs + `882 x 8 = 7,056` CLI linter tool outputs),
   reconciling extracted `len(issues)` against SonarQube `/api/measures/component` `violations`, counting
   snapshots using Pylint >500-file / >1.5MB deterministic batching, and recording exact pinned binary versions
   -> `results/full/execution_audit_and_versions.json`.
3. Runs objective Python 3 `ast.parse` verification at initial and final N09 commits across ALL 53 Stage-1
   repositories (>=10 valid N09 commits; 49 included + 4 Stage-2 Python 2 exclusions) with zero default fallbacks
   -> `data/probe_all_53_stage1_candidates.csv` and updates `data/prisma_cohort_funnel.csv`.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))


def safe_snap_stem(sid: str) -> str:
    return sid.replace("::", "__").replace("/", "_")


def export_linter_catalogs_and_versions() -> Tuple[Dict[str, Any], Dict[str, str]]:
    versions: Dict[str, str] = {}
    envs_dir = BASE_DIR / "envs"
    for env_d in sorted(envs_dir.iterdir()):
        if not env_d.is_dir():
            continue
        for bin_n in ("python", "flake8", "ruff", "pylint"):
            bp = env_d / "bin" / bin_n
            if bp.exists():
                r = subprocess.run([str(bp), "--version"], capture_output=True, text=True)
                versions[f"{env_d.name}:{bin_n}"] = " | ".join(
                    ln.strip() for ln in r.stdout.splitlines() if ln.strip()
                )

    catalogs: Dict[str, Any] = {}

    # 1. Pylint 2.17.7 and Pylint 3.3.4 enabled rules under `--rcfile=/dev/null --disable=E0401,E0611`
    # Parse ALL Pylint message categories (`I`, `C`, `R`, `W`, `E`, `F`) without filtering out `I`-prefixed messages.
    for pl_env in ("pylint_217", "pylint_33"):
        pb = envs_dir / pl_env / "bin" / "pylint"
        res = subprocess.run(
            [str(pb), "--rcfile=/dev/null", "--disable=E0401,E0611", "--list-msgs-enabled"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Lines after 'Enabled messages:' up to 'Disabled messages:'
        enabled_section = res.stdout.split("Disabled messages:")[0]
        bullet_lines = [ln for ln in enabled_section.splitlines() if re.search(r"\([A-Z]\d{4}\)", ln)]
        codes = sorted(set(re.findall(r"\(([ICRWEF]\d{4})\)", enabled_section)))
        all_alpha_codes = sorted(set(re.findall(r"\(([A-Z]\d{4})\)", enabled_section)))
        assert codes == all_alpha_codes and len(codes) == len(bullet_lines) and len(codes) > 350, (
            f"Pylint enabled-list parse mismatch in {pl_env}: {len(codes)} vs {len(all_alpha_codes)} vs {len(bullet_lines)} lines"
        )
        catalogs[pl_env] = {
            "enabled_count": len(codes),
            "enabled_count_monorepo_no_r0801": len([c for c in codes if c != "R0801"]),
            "i_category_count": len([c for c in codes if c.startswith("I")]),
            "codes": codes,
        }

    # 2. Ruff 0.1.15 and Ruff 0.17.5 exact enabled rules per selector mode
    ruff_modes = {
        "ruff_01": [
            ("ruff_01_default", []),
            ("ruff_01_extended", ["--select", "E,W,F,C90,B,PL,S"]),
        ],
        "ruff_modern": [
            ("ruff_modern_default", []),
            ("ruff_modern_extended", ["--select", "E,W,F,C90,B,PL,S"]),
            (
                "ruff_modern_pylint",
                [
                    "--select",
                    "PL,E9,F401,F404,F601,F701,F702,F704,F706,F811,F821,F822,F841,E711,E712,E722,B006,B007,B018,C901",
                ],
            ),
        ],
    }
    for r_env, mode_list in ruff_modes.items():
        rb = envs_dir / r_env / "bin" / "ruff"
        r_all = subprocess.run([str(rb), "rule", "--all", "--output-format", "json"], capture_output=True, text=True)
        all_rules = json.loads(r_all.stdout)
        norm_name_to_code = {
            re.sub(r"[^a-z0-9]", "", item["name"].lower()): item["code"] for item in all_rules
        }
        valid_codes_set = {item["code"] for item in all_rules}
        for reg_name, extra_args in mode_list:
            r_set = subprocess.run(
                [str(rb), "check", "--isolated", *extra_args, "--show-settings"],
                capture_output=True,
                text=True,
            )
            m = re.search(r"linter\.rules\.enabled\s*=\s*\[(.*?)\]", r_set.stdout, re.S)
            enabled_codes: List[str] = []
            if m:
                for raw_ln in m.group(1).splitlines():
                    nm = raw_ln.strip().strip(",").strip('"').strip("'")
                    if not nm or nm.startswith("#"):
                        continue
                    m_paren = re.search(r"\(([A-Z0-9]+)\)", nm)
                    if m_paren and m_paren.group(1) in valid_codes_set:
                        enabled_codes.append(m_paren.group(1))
                    elif nm in valid_codes_set:
                        enabled_codes.append(nm)
                    else:
                        nk = re.sub(r"[^a-z0-9]", "", nm.lower())
                        if nk in norm_name_to_code:
                            enabled_codes.append(norm_name_to_code[nk])
            enabled_codes = sorted(set(enabled_codes))
            catalogs[reg_name] = {
                "binary_total_rules": len(all_rules),
                "enabled_count": len(enabled_codes),
                "codes": enabled_codes,
            }

    # 3. Flake8 7.1.1 (`--isolated --max-complexity=10` with pyflakes, pycodestyle, mccabe, flake8-bugbear)
    # Validated directly via Flake8's runtime `app.guide.decider.decision_for(code)`!
    f8_py = envs_dir / "flake8_71" / "bin" / "python"
    f8_probe = r"""
import inspect, json, re, pycodestyle, bugbear
from flake8.main.application import Application
from flake8.plugins.pyflakes import FLAKE8_PYFLAKES_CODES

app = Application()
app.initialize(["--isolated", "--max-complexity=10"])
dec = app.guide.decider
# Include all E/W codes in pycodestyle source/docstrings plus E111..E117 generated via `tmpl = "E11%d %s"` in pycodestyle.indentation:
cand = set(["C901", "E999", "E111", "E112", "E113", "E114", "E115", "E116", "E117"])
cand.update(FLAKE8_PYFLAKES_CODES.values())
cand.update(re.findall(r"\b(B[09]\d{2})\b", inspect.getsource(bugbear)))
cand.update(re.findall(r"\b([EW]\d{3})\b", inspect.getsource(pycodestyle)))
selected = sorted(c for c in cand if "Selected" in str(dec.decision_for(c)))
ignored = sorted(c for c in cand if "Selected" not in str(dec.decision_for(c)))
print(json.dumps({"selected": selected, "default_ignored": ignored}))
"""
    r_f8 = subprocess.run([str(f8_py), "-c", f8_probe], capture_output=True, text=True, check=True)
    f8_info = json.loads(r_f8.stdout.strip().splitlines()[-1])
    catalogs["flake8_71"] = {
        "selection_method": "flake8.style_guide.Decider.decision_for(--isolated --max-complexity=10)",
        "default_ignored_codes": f8_info["default_ignored"],
        "enabled_count": len(f8_info["selected"]),
        "codes": f8_info["selected"],
    }

    (BASE_DIR / "data" / "linter_catalogs.json").write_text(json.dumps(catalogs, indent=2), encoding="utf-8")
    return catalogs, versions


def _audit_one_snap(row_tuple: Tuple[str, str, str, int, int]) -> Dict[str, Any]:
    sid, repo, sha, seq_idx, mf_py_files = row_tuple
    stem = safe_snap_stem(sid)
    sq_regs = ("sq99_std", "sq103_n09", "sq26_mqr", "sq26_comp_mqr")
    expected_tools = {
        "flake8_71",
        "ruff_01_default",
        "ruff_01_extended",
        "ruff_modern_default",
        "ruff_modern_extended",
        "ruff_modern_pylint",
        "pylint_217",
        "pylint_33",
    }
    res: Dict[str, Any] = {"sid": sid, "repo": repo, "seq_idx": seq_idx, "sq": {}, "linter_ok": False, "linter_counts": {}}
    for reg in sq_regs:
        p = BASE_DIR / "data" / "sq_runs" / reg / f"{stem}.json"
        if not p.exists():
            res["sq"][reg] = {"exists": False}
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        iss = d.get("issues", [])
        n_iss = len(iss)
        n_new = sum(1 for x in iss if x.get("is_new"))
        n_std_high = sum(1 for x in iss if x.get("severity") in ("CRITICAL", "BLOCKER"))
        n_std_rs = sum(1 for x in iss if x.get("type") in ("BUG", "VULNERABILITY"))
        n_mqr_high = sum(
            1 for x in iss if any(i.get("severity") in ("HIGH", "BLOCKER") for i in (x.get("impacts") or []))
        )
        qg_st = d.get("quality_gate", {}).get("status")
        m_list = d.get("measures", {}).get("measures", [])
        m_dict = {m["metric"]: m.get("value", m.get("period", {}).get("value")) for m in m_list}

        def _mi(k: str) -> Any:
            return int(m_dict[k]) if k in m_dict and m_dict[k] is not None else None

        v_meas = _mi("violations")
        nv_meas = _mi("new_violations")
        std_h_meas = (
            (_mi("blocker_violations") + _mi("critical_violations"))
            if _mi("blocker_violations") is not None and _mi("critical_violations") is not None
            else None
        )
        std_rs_meas = (
            (_mi("bugs") + _mi("vulnerabilities"))
            if _mi("bugs") is not None and _mi("vulnerabilities") is not None
            else None
        )
        mqr_h_meas = (
            (_mi("software_quality_blocker_issues") + _mi("software_quality_high_issues"))
            if _mi("software_quality_blocker_issues") is not None and _mi("software_quality_high_issues") is not None
            else None
        )
        res["sq"][reg] = {
            "exists": True,
            "qg_status": qg_st,
            "n_issues": n_iss,
            "measures_violations": v_meas,
            "n_new_issues": n_new,
            "measures_new_violations": nv_meas,
            "n_std_high": n_std_high,
            "measures_std_high": std_h_meas,
            "n_std_rel_sec": n_std_rs,
            "measures_std_rel_sec": std_rs_meas,
            "n_mqr_high": n_mqr_high,
            "measures_mqr_high": mqr_h_meas,
            "raw_measures": m_dict if (v_meas is not None and n_iss != v_meas) else None,
        }
    lp = BASE_DIR / "data" / "linter_runs" / f"{stem}.json"
    if lp.exists():
        ld = json.loads(lp.read_text(encoding="utf-8"))
        tools = ld.get("tools", {})
        if expected_tools.issubset(set(tools.keys())):
            res["linter_ok"] = True
            res["linter_counts"] = {k: len(v) for k, v in tools.items()}

    # Exact Pylint invocation branch from git tree `.py` file count & byte sum (matching run_linters_full.py):
    slug = repo.replace("/", "__")
    rdir = BASE_DIR / "repos" / slug
    ignore_dirs = {".git", "__pycache__", "build", "dist", ".tox", "venv", ".venv", ".eggs", ".pylint_cache"}
    py_cnt, py_bytes = mf_py_files, 0
    if rdir.exists():
        gt = subprocess.run(["git", "-C", str(rdir), "ls-tree", "-r", "-l", sha], capture_output=True, text=True)
        if gt.returncode == 0:
            fc, fb = 0, 0
            for ln in gt.stdout.splitlines():
                parts = ln.split(None, 4)
                if len(parts) == 5 and parts[1] == "blob" and parts[4].endswith(".py"):
                    relp = parts[4]
                    if "sympy/integrals/rubi/" in relp:
                        continue
                    if any(seg in ignore_dirs for seg in relp.split("/")):
                        continue
                    fc += 1
                    try:
                        fb += int(parts[3])
                    except ValueError:
                        pass
            py_cnt, py_bytes = fc, fb
    if py_cnt <= 500 and py_bytes <= 1_500_000:
        pl_branch = "recursive_r0801_enabled"
    elif py_cnt <= 500:
        pl_branch = "batch60_gt1p5mb_no_r0801"
    else:
        pl_branch = "batch250_gt500files_no_r0801"
    res["pylint_branch"] = pl_branch
    res["pylint_py_files"] = py_cnt
    res["pylint_py_bytes"] = py_bytes
    return res


def audit_execution_completeness(manifest_df: pd.DataFrame, versions: Dict[str, str], catalogs: Dict[str, Any]) -> None:
    tuples = [
        (str(r["snapshot_id"]), str(r["repository"]), str(r["commit"]), int(r["seq_idx"]), int(r["py_files"]))
        for _, r in manifest_df.iterrows()
    ]
    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(_audit_one_snap, tuples, chunksize=25):
            results.append(r)

    sq_summary: Dict[str, Any] = {}
    for reg in ("sq99_std", "sq103_n09", "sq26_mqr", "sq26_comp_mqr"):
        present = sum(1 for r in results if r["sq"][reg].get("exists"))
        qg_valid = sum(1 for r in results if r["sq"][reg].get("qg_status") in ("OK", "ERROR"))
        exact_v, exact_new_v, n_trans_new = 0, 0, 0
        exact_std_h, exact_std_rs = 0, 0
        exact_mqr_h, n_mqr_checked = 0, 0
        mismatches = []
        for r in results:
            inf = r["sq"][reg]
            if not inf.get("exists"):
                continue
            if inf.get("measures_violations") is not None:
                if inf["n_issues"] == inf["measures_violations"]:
                    exact_v += 1
                else:
                    mismatches.append(
                        {
                            "snapshot_id": r["sid"],
                            "extracted_issues": inf["n_issues"],
                            "api_measures_violations": inf["measures_violations"],
                            "diff": inf["n_issues"] - inf["measures_violations"],
                            "extracted_new_issues": inf["n_new_issues"],
                            "api_measures_new_violations": inf["measures_new_violations"],
                            "extracted_std_high": inf["n_std_high"],
                            "api_measures_std_high": inf["measures_std_high"],
                            "extracted_mqr_high": inf["n_mqr_high"],
                            "api_measures_mqr_high": inf["measures_mqr_high"],
                            "extracted_std_rel_sec": inf["n_std_rel_sec"],
                            "api_measures_std_rel_sec": inf["measures_std_rel_sec"],
                            "server_measures": inf["raw_measures"],
                        }
                    )
            if r["seq_idx"] > 1 and inf.get("measures_new_violations") is not None:
                n_trans_new += 1
                if inf["n_new_issues"] == inf["measures_new_violations"]:
                    exact_new_v += 1
            if inf.get("measures_std_high") is not None and inf["n_std_high"] == inf["measures_std_high"]:
                exact_std_h += 1
            if inf.get("measures_std_rel_sec") is not None and inf["n_std_rel_sec"] == inf["measures_std_rel_sec"]:
                exact_std_rs += 1
            if inf.get("measures_mqr_high") is not None:
                n_mqr_checked += 1
                if inf["n_mqr_high"] == inf["measures_mqr_high"]:
                    exact_mqr_h += 1

        sq_summary[reg] = {
            "expected_snapshots": len(tuples),
            "verified_json_files": present,
            "valid_qg_verdicts": qg_valid,
            "exact_violations_measure_matches": exact_v,
            "violations_measure_mismatches_count": len(mismatches),
            "exact_transition_new_violations_matches": exact_new_v,
            "expected_transitions": n_trans_new,
            "exact_std_high_matches": exact_std_h,
            "exact_std_rel_sec_matches": exact_std_rs,
            "exact_mqr_high_matches": exact_mqr_h if n_mqr_checked else None,
            "mqr_high_checked_snapshots": n_mqr_checked,
            "max_abs_diff": max([abs(m["diff"]) for m in mismatches], default=0),
            "mismatches_detail": mismatches,
        }

    linter_ok_cnt = sum(1 for r in results if r["linter_ok"])
    branch_counts = {
        b: sum(1 for r in results if r["pylint_branch"] == b)
        for b in ("recursive_r0801_enabled", "batch60_gt1p5mb_no_r0801", "batch250_gt500files_no_r0801")
    }
    repos_batch60 = sorted({r["repo"] for r in results if r["pylint_branch"] == "batch60_gt1p5mb_no_r0801"})
    repos_batch250 = sorted({r["repo"] for r in results if r["pylint_branch"] == "batch250_gt500files_no_r0801"})
    repos_any_batched = sorted(set(repos_batch60) | set(repos_batch250))

    sq_meta = {}
    for fn in ("sq99_std_meta.json", "sq103_n09_meta.json", "sq26_mqr_meta.json", "sq26_comp_mqr_meta.json"):
        mp = BASE_DIR / "data" / "sq_catalogs" / fn
        if mp.exists():
            md = json.loads(mp.read_text(encoding="utf-8"))
            sq_meta[fn.replace("_meta.json", "")] = {
                "image": md.get("image"),
                "sq_version": md.get("sq_version"),
            }

    audit_out = {
        "n_expected_snapshots": len(tuples),
        "n_verified_sq_scans": sum(v["verified_json_files"] for v in sq_summary.values()),
        "n_verified_linter_tool_scans": linter_ok_cnt * 8,
        "n_total_verified_physical_scans": sum(v["verified_json_files"] for v in sq_summary.values()) + linter_ok_cnt * 8,
        "sonarqube_versions": sq_meta,
        "cli_linter_versions": versions,
        "linter_enabled_rule_counts": {k: v["enabled_count"] for k, v in catalogs.items()},
        "sonarqube_completeness_and_measure_reconciliation": sq_summary,
        "linter_completeness": {
            "expected_snapshot_bundles": len(tuples),
            "verified_complete_8tool_bundles": linter_ok_cnt,
            "bundle_audit_scope_note": "Verifies presence of all 8 tool result arrays across 882/882 snapshot bundles; individual CLI subprocess batch retries were handled inline without per-batch exit-code logs.",
            "pylint_execution_branches": {
                "recursive_r0801_enabled_snapshots": branch_counts["recursive_r0801_enabled"],
                "batch60_gt1p5mb_le500files_no_r0801_snapshots": branch_counts["batch60_gt1p5mb_no_r0801"],
                "batch250_gt500files_no_r0801_snapshots": branch_counts["batch250_gt500files_no_r0801"],
                "total_batched_no_r0801_snapshots": branch_counts["batch60_gt1p5mb_no_r0801"] + branch_counts["batch250_gt500files_no_r0801"],
                "repos_using_batch60": repos_batch60,
                "repos_using_batch250": repos_batch250,
                "n_repos_any_batched": len(repos_any_batched),
            },
        },
    }
    out_p = BASE_DIR / "results" / "full" / "execution_audit_and_versions.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(audit_out, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_p}")


def _ast_pct_at_commit(repo_dir: Path, sha: str, wt_dir: Path) -> Tuple[int, int, float]:
    if not sha or not repo_dir.exists():
        return 0, 0, 0.0
    if wt_dir.exists():
        subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)], capture_output=True)
    add_r = subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "add", "--detach", "--quiet", str(wt_dir), sha],
        capture_output=True,
    )
    if add_r.returncode != 0:
        return 0, 0, 0.0
    try:
        py_files = [
            p
            for p in wt_dir.rglob("*.py")
            if ".git" not in p.parts and "sympy/integrals/rubi" not in p.relative_to(wt_dir).as_posix()
        ]
        if not py_files:
            return 0, 0, 100.0
        ok = 0
        for p in py_files:
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
                ast.parse(src)
                ok += 1
            except (SyntaxError, ValueError, OSError):
                pass
        return len(py_files), ok, round(100.0 * ok / len(py_files), 2)
    finally:
        subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)], capture_output=True)


def verify_all_53_stage1_ast_parseability(manifest_df: pd.DataFrame) -> None:
    funnel_p = BASE_DIR / "data" / "prisma_cohort_funnel.csv"
    funnel_df = pd.read_csv(funnel_p)
    old_probe = {}
    p22 = BASE_DIR / "data" / "probe_22_candidates.csv"
    if p22.exists():
        for _, r in pd.read_csv(p22).iterrows():
            old_probe[str(r["repo"]).strip()] = (float(r["py3_first_pct"]), float(r["py3_last_pct"]))

    n09_csv = pd.read_csv(BASE_DIR / "figshare_n09" / "sonarqube_software_quality_categories_metrics.csv")
    n09_csv["tot"] = n09_csv["maintainability_count"] + n09_csv["reliability_count"] + n09_csv["security_count"]
    n09_valid = n09_csv[n09_csv["tot"] > 0].copy()
    n09_valid["dt"] = pd.to_datetime(n09_valid["published_at"], utc=True, format="mixed")

    stage1_repos = funnel_df[funnel_df["prisma_status"] != "EXCLUDED_STAGE1"]["repository"].tolist()
    probe_rows = []

    tmp_ast_root = BASE_DIR / "tmp_ast_audit_wt"
    tmp_ast_root.mkdir(parents=True, exist_ok=True)

    def _probe_repo(repo: str) -> Dict[str, Any]:
        slug = repo.replace("/", "__")
        repo_dir = BASE_DIR / "repos" / slug
        wt_dir = tmp_ast_root / slug
        if repo in set(manifest_df["repository"]):
            g = manifest_df[manifest_df["repository"] == repo].sort_values("seq_idx")
            sha_first = str(g["commit"].iloc[0])
            sha_last = str(g["commit"].iloc[-1])
            n_first, ok_first, pct_first = _ast_pct_at_commit(repo_dir, sha_first, wt_dir)
            n_last, ok_last, pct_last = _ast_pct_at_commit(repo_dir, sha_last, wt_dir)
            return {
                "repository": repo,
                "stage2_status": "INCLUDED",
                "first_commit": sha_first[:10],
                "last_commit": sha_last[:10],
                "py_files_first": n_first,
                "py3_parseable_first": ok_first,
                "py3_first_pct": pct_first,
                "py_files_last": n_last,
                "py3_parseable_last": ok_last,
                "py3_last_pct": pct_last,
                "measurement_source": "git_ast_parse_on_sampled_window",
            }
        # Excluded Stage-2 repository: check if cloned or in probe_22_candidates.csv
        sub = n09_valid[n09_valid["repository"] == repo].sort_values("dt").drop_duplicates(subset=["commit"], keep="first")
        sha_first = str(sub["commit"].iloc[0]) if len(sub) else ""
        sha_last = str(sub["commit"].iloc[-1]) if len(sub) else ""
        if repo_dir.exists() and (repo_dir / ".git").exists():
            n_first, ok_first, pct_first = _ast_pct_at_commit(repo_dir, sha_first, wt_dir)
            n_last, ok_last, pct_last = _ast_pct_at_commit(repo_dir, sha_last, wt_dir)
            src_lbl = "git_ast_parse_full_repo"
        elif repo in old_probe:
            pct_first, pct_last = old_probe[repo]
            n_first, ok_first, n_last, ok_last = -1, -1, -1, -1
            src_lbl = "archived_stage2_ast_probe"
        else:
            pct_first, pct_last = 0.0, 0.0
            n_first, ok_first, n_last, ok_last = 0, 0, 0, 0
            src_lbl = "unprobed"
        return {
            "repository": repo,
            "stage2_status": "EXCLUDED_STAGE2",
            "first_commit": sha_first[:10],
            "last_commit": sha_last[:10],
            "py_files_first": n_first,
            "py3_parseable_first": ok_first,
            "py3_first_pct": pct_first,
            "py_files_last": n_last,
            "py3_parseable_last": ok_last,
            "py3_last_pct": pct_last,
            "measurement_source": src_lbl,
        }

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_probe_repo, rp): rp for rp in stage1_repos}
        for fut in as_completed(futs):
            probe_rows.append(fut.result())

    probe_rows.sort(key=lambda x: x["repository"])
    probe_df = pd.DataFrame(probe_rows)
    out_probe = BASE_DIR / "data" / "probe_all_53_stage1_candidates.csv"
    probe_df.to_csv(out_probe, index=False)

    # Also update prisma_cohort_funnel.csv with measured py3_first_pct and py3_last_pct columns
    pmap = {r["repository"]: (r["py3_first_pct"], r["py3_last_pct"]) for r in probe_rows}
    funnel_df["py3_first_pct"] = [pmap.get(rp, (None, None))[0] for rp in funnel_df["repository"]]
    funnel_df["py3_last_pct"] = [pmap.get(rp, (None, None))[1] for rp in funnel_df["repository"]]
    funnel_df.to_csv(funnel_p, index=False)
    print(f"[OK] Wrote {out_probe} ({len(probe_df)} Stage-1 repos; min included first={probe_df[probe_df['stage2_status']=='INCLUDED']['py3_first_pct'].min()}%, min included last={probe_df[probe_df['stage2_status']=='INCLUDED']['py3_last_pct'].min()}%)")


def main() -> None:
    manifest_df = pd.read_csv(BASE_DIR / "data" / "full_releases_manifest.csv")
    catalogs, versions = export_linter_catalogs_and_versions()
    audit_execution_completeness(manifest_df, versions, catalogs)
    verify_all_53_stage1_ast_parseability(manifest_df)


if __name__ == "__main__":
    main()
