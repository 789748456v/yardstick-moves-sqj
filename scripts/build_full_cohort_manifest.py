#!/usr/bin/env python3
"""
Deterministic PRISMA Cohort Selection, Parallel Repository Cloning, and Release Manifest Builder
for D04-Plus Full Study (anchored to SQJ 2026 N09's 57-repository published benchmark).

Two-Stage Zero-Size-Cap PRISMA Inclusion Funnel on N09's 57 Repositories:
  1. Longitudinal Series Depth: >= 10 valid unique-commit records in N09 with >0 reported issues (53 / 57 pass)
  2. Python 3 AST Parseability: >= 85.0% at initial release and >= 95.0% at final release (49 / 53 pass; 0 size cap)
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from git_diff_newcode import extract_new_code_lines_and_churn

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))
MAX_CONSECUTIVE = 20
CLONE_WORKERS = 6


def run_git(repo_dir: Path, args: List[str]) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo_dir)] + args,
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    return res.stdout.strip()


def resolve_git_tag(repo_dir: Path, commit_sha: str, fallback_tag: str) -> str:
    clean_fb = str(fallback_tag).lstrip("'")
    out = run_git(repo_dir, ["tag", "--points-at", commit_sha])
    if out:
        tags = [t.strip() for t in out.splitlines() if t.strip()]
        if clean_fb in tags:
            return clean_fb
        if fallback_tag in tags:
            return str(fallback_tag)
        return tags[0]
    return clean_fb


def compute_py_loc_at_commit(worktree_dir: Path) -> Dict[str, int]:
    py_files = 0
    total_lines = 0
    sloc = 0
    for p in worktree_dir.rglob("*.py"):
        if ".git" in p.parts:
            continue
        py_files += 1
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in txt.splitlines():
            total_lines += 1
            s = raw.strip()
            if s and not s.startswith("#"):
                sloc += 1
    return {"py_files": py_files, "py_total_lines": total_lines, "py_sloc": sloc}


def ensure_repo_cloned(repo: str, repos_dir: Path) -> Tuple[str, bool, str]:
    slug = repo.replace("/", "__")
    target = repos_dir / slug
    if target.exists() and (target / ".git").exists():
        # Verify HEAD resolves cleanly (not an interrupted partial directory)
        chk = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True)
        if chk.returncode == 0:
            return repo, True, "cached"
    if target.exists():
        subprocess.run(["rm", "-rf", str(target)], capture_output=True)
    import os
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    for attempt in range(1, 4):
        res = subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--quiet", f"https://github.com/{repo}.git", str(target)],
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=300,
        )
        if res.returncode == 0:
            return repo, True, f"cloned_blobless_attempt_{attempt}"
    return repo, False, res.stderr[-300:] if res else "clone_failed"


def build_repo_manifest(
    repo: str,
    in_rq3: bool,
    sub_df: pd.DataFrame,
    repos_dir: Path,
    tmp_wt_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[int]]]]:
    slug = repo.replace("/", "__")
    repo_dir = repos_dir / slug
    sub = sub_df.copy()
    sub["dt"] = pd.to_datetime(sub["published_at"], utc=True, format="mixed")
    sub = sub.sort_values("dt").drop_duplicates(subset=["commit"], keep="first")

    valid_records: List[Dict[str, Any]] = []
    for _, row in sub.iterrows():
        sha = str(row["commit"]).strip()
        chk = subprocess.run(
            ["git", "-C", str(repo_dir), "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
        )
        if chk.returncode != 0:
            # Try fetching specific commit if not on default refs
            subprocess.run(
                ["git", "-C", str(repo_dir), "fetch", "--quiet", "origin", sha],
                capture_output=True,
                timeout=60,
            )
            chk = subprocess.run(
                ["git", "-C", str(repo_dir), "cat-file", "-e", f"{sha}^{{commit}}"],
                capture_output=True,
            )
        if chk.returncode == 0:
            commit_iso = run_git(repo_dir, ["show", "-s", "--format=%cI", sha])
            real_tag = resolve_git_tag(repo_dir, sha, str(row["release_tag"]))
            valid_records.append(
                {
                    "repository": repo,
                    "slug": slug,
                    "in_n09_rq3_sig": bool(in_rq3),
                    "release_tag": real_tag,
                    "n09_raw_tag": str(row["release_tag"]),
                    "commit": sha,
                    "published_at": row["dt"].isoformat(),
                    "commit_at": commit_iso,
                    "n09_maint_count": int(row["maintainability_count"]),
                    "n09_rel_count": int(row["reliability_count"]),
                    "n09_sec_count": int(row["security_count"]),
                    "n09_sqale_debt_ratio": float(row["sqale_debt_ratio"]),
                }
            )

    # Guard against bulk-imported GitHub releases published on the same day
    if len(valid_records) >= 2:
        t_span_days = (
            pd.to_datetime(valid_records[-1]["published_at"])
            - pd.to_datetime(valid_records[0]["published_at"])
        ).total_seconds() / 86400.0
        if t_span_days < 7.0:
            valid_records.sort(key=lambda r: pd.to_datetime(r["commit_at"]))

    if len(valid_records) < 10:
        raise RuntimeError(f"{repo} only resolved {len(valid_records)} valid git commits (<10)")

    chosen = valid_records[-MAX_CONSECUTIVE:]
    wt_dir = tmp_wt_root / slug
    if wt_dir.exists():
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)],
            capture_output=True,
        )
    subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "add", "--detach", str(wt_dir), chosen[0]["commit"]],
        check=True,
        capture_output=True,
    )

    rows: List[Dict[str, Any]] = []
    nc_local: Dict[str, Dict[str, List[int]]] = {}
    try:
        for idx, rec in enumerate(chosen, start=1):
            subprocess.run(
                ["git", "-C", str(wt_dir), "checkout", "--force", "--quiet", rec["commit"]],
                check=True,
            )
            subprocess.run(["git", "-C", str(wt_dir), "clean", "-fdx", "--quiet"], check=True)
            loc_stats = compute_py_loc_at_commit(wt_dir)
            rec["seq_idx"] = idx
            rec["series_len"] = len(chosen)
            rec.update(loc_stats)

            snap_key = f"{slug}::{idx:02d}::{rec['release_tag']}"
            rec["snapshot_id"] = snap_key

            if idx == 1:
                rec.update(
                    {
                        "prev_commit": "",
                        "prev_tag": "",
                        "modified_py_files": 0,
                        "py_lines_added": 0,
                        "py_lines_deleted": 0,
                        "py_lines_changed": 0,
                        "all_modified_files": 0,
                        "all_lines_changed": 0,
                    }
                )
            else:
                prev = chosen[idx - 2]
                new_lines, churn = extract_new_code_lines_and_churn(
                    repo_dir, prev["commit"], rec["commit"]
                )
                rec["prev_commit"] = prev["commit"]
                rec["prev_tag"] = prev["release_tag"]
                rec.update(churn)
                nc_local[snap_key] = {
                    fpath: sorted(lines) for fpath, lines in new_lines.items() if lines
                }
            rows.append(rec)
    finally:
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)],
            capture_output=True,
        )
    return rows, nc_local


def main() -> None:
    import os
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    fig_dir = BASE_DIR / "figshare_n09"
    repos_dir = BASE_DIR / "repos"
    out_dir = BASE_DIR / "data"
    repos_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = pd.read_excel(fig_dir / "repositories_general_data.xlsx", sheet_name="in")
    sp = pd.read_csv(fig_dir / "spearman_correlation_results.csv")
    rq3_repos = set(sp["Repository"].str.strip())

    n09_csv = pd.read_csv(fig_dir / "sonarqube_software_quality_categories_metrics.csv")
    n09_csv["tot"] = (
        n09_csv["maintainability_count"] + n09_csv["reliability_count"] + n09_csv["security_count"]
    )
    n09_valid = n09_csv[n09_csv["tot"] > 0].copy()
    valid_cnt_map = (
        n09_valid.drop_duplicates(subset=["repository", "commit"])
        .groupby("repository")["commit"]
        .count()
        .to_dict()
    )

    # Load complete measured Python 3 AST compatibility ledger across all 53 Stage-1 candidates (`v_cnt >= 10`)
    # Zero unmeasured `(100.0, 100.0)` fallbacks and zero hardcoded repository exclusion sets.
    probe_csv = out_dir / "probe_all_53_stage1_candidates.csv"
    if not probe_csv.exists():
        raise FileNotFoundError(
            f"Missing required measured Stage-2 AST parseability ledger: {probe_csv}"
        )
    pdf = pd.read_csv(probe_csv)
    probe_map: Dict[str, Dict[str, Any]] = {}
    for _, pr in pdf.iterrows():
        rp_key = str(pr["repository"]).strip()
        f_sha = str(pr["first_commit"]).strip()
        l_sha = str(pr["last_commit"]).strip()
        if not f_sha or not l_sha or pd.isna(pr["py3_first_pct"]) or pd.isna(pr["py3_last_pct"]):
            raise ValueError(f"Incomplete AST parseability measurement for {rp_key} in {probe_csv}")
        probe_map[rp_key] = {
            "first_commit": f_sha,
            "last_commit": l_sha,
            "py3_first_pct": float(pr["py3_first_pct"]),
            "py3_last_pct": float(pr["py3_last_pct"]),
            "measurement_source": str(pr["measurement_source"]).strip(),
        }

    funnel_rows = []
    included_repos = []
    for _, r in gen.iterrows():
        repo = str(r["repository"]).strip()
        v_cnt = int(valid_cnt_map.get(repo, 0))
        last_dt = pd.to_datetime(r["last_release_published_at"], utc=True, format="mixed")
        last_cloc = int(r["last_release_cloc"])
        in_rq3 = repo in rq3_repos

        if v_cnt < 10:
            status, reason = "EXCLUDED_STAGE1", f"valid_releases={v_cnt} < 10"
            p_first, p_last, m_src = None, None, "excluded_stage1_horizon"
        else:
            if repo not in probe_map:
                raise KeyError(
                    f"Stage-1 eligible candidate {repo} (valid_releases={v_cnt}) is missing from {probe_csv}"
                )
            pinfo = probe_map[repo]
            p_first = pinfo["py3_first_pct"]
            p_last = pinfo["py3_last_pct"]
            m_src = pinfo["measurement_source"]
            if p_last < 95.0 or p_first < 85.0:
                status = "EXCLUDED_STAGE2"
                reason = f"py2_syntax_incompatible (py3_first={p_first:.2f}%, py3_last={p_last:.2f}%)"
            else:
                status = "INCLUDED"
                reason = f"meets_all_criteria_zero_size_cap (py3_first={p_first:.2f}%, py3_last={p_last:.2f}%)"
                included_repos.append(repo)

        funnel_rows.append(
            {
                "repository": repo,
                "n09_total_releases": int(r["total_releases"]),
                "valid_commits_in_n09": v_cnt,
                "first_release_cloc": int(r["first_release_cloc"]),
                "last_release_cloc": last_cloc,
                "last_release_year": int(last_dt.year),
                "in_n09_rq3_sig23": in_rq3,
                "prisma_status": status,
                "prisma_reason": reason,
                "measured_py3_first_pct": p_first,
                "measured_py3_last_pct": p_last,
                "ast_measurement_source": m_src,
            }
        )

    funnel_df = pd.DataFrame(funnel_rows)
    funnel_df.to_csv(out_dir / "prisma_cohort_funnel.csv", index=False)
    print(f"PRISMA Funnel across {len(funnel_df)} N09 repos:")
    print(funnel_df["prisma_status"].value_counts().to_string())
    print(f"Included repos: {len(included_repos)}")

    # Clone all included repos in parallel
    print("Cloning / verifying git repositories...", flush=True)
    with ThreadPoolExecutor(max_workers=CLONE_WORKERS) as ex:
        futs = {ex.submit(ensure_repo_cloned, repo, repos_dir): repo for repo in included_repos}
        for fut in as_completed(futs):
            rp, ok, msg = fut.result()
            print(f"  [clone] {rp}: ok={ok} ({msg})", flush=True)
            if not ok:
                raise RuntimeError(f"Failed cloning {rp}: {msg}")

    # Reuse existing manifest & newcode records for already-built repositories (100% cache-preserving)
    existing_by_repo: Dict[str, List[Dict[str, Any]]] = {}
    existing_nc: Dict[str, Dict[str, List[int]]] = {}
    prev_m_json = out_dir / "full_releases_manifest.json"
    prev_nc_json = out_dir / "full_newcode_lines.json"
    if prev_m_json.exists() and prev_nc_json.exists():
        for item in json.loads(prev_m_json.read_text(encoding="utf-8")):
            existing_by_repo.setdefault(item["repository"], []).append(item)
        existing_nc = json.loads(prev_nc_json.read_text(encoding="utf-8"))

    tmp_wt_root = BASE_DIR / "tmp_manifest_wt_full"
    tmp_wt_root.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    all_newcode: Dict[str, Dict[str, List[int]]] = {}

    to_build = []
    for repo in sorted(included_repos):
        if repo in existing_by_repo and len(existing_by_repo[repo]) >= 10:
            c_rows = sorted(existing_by_repo[repo], key=lambda x: x["seq_idx"])
            all_rows.extend(c_rows)
            for cr in c_rows:
                sid = cr["snapshot_id"]
                if sid in existing_nc:
                    all_newcode[sid] = existing_nc[sid]
            print(f"  [manifest-cached] {repo}: {len(c_rows)} consecutive releases", flush=True)
        else:
            to_build.append(repo)

    with ThreadPoolExecutor(max_workers=CLONE_WORKERS) as ex:
        futs = {
            ex.submit(
                build_repo_manifest,
                repo,
                repo in rq3_repos,
                n09_valid[n09_valid["repository"] == repo],
                repos_dir,
                tmp_wt_root,
            ): repo
            for repo in to_build
        }
        for fut in as_completed(futs):
            repo = futs[fut]
            rows, nc_local = fut.result()
            all_rows.extend(rows)
            all_newcode.update(nc_local)
            print(
                f"  [manifest] {repo}: {len(rows)} consecutive releases ({rows[0]['release_tag']} -> {rows[-1]['release_tag']}, py_sloc={rows[-1]['py_sloc']})",
                flush=True,
            )

    all_rows.sort(key=lambda x: (x["repository"], x["seq_idx"]))
    manifest_df = pd.DataFrame(all_rows)
    for repo in sorted(included_repos):
        sub_r = manifest_df[manifest_df["repository"] == repo].sort_values("seq_idx")
        pinfo = probe_map[repo]
        if pinfo["measurement_source"] == "git_ast_parse_on_sampled_window":
            c_first = str(sub_r.iloc[0]["commit"])[: len(pinfo["first_commit"])]
            c_last = str(sub_r.iloc[-1]["commit"])[: len(pinfo["last_commit"])]
            if c_first != pinfo["first_commit"] or c_last != pinfo["last_commit"]:
                raise ValueError(
                    f"Commit window mismatch for {repo}: manifest ({c_first}, {c_last}) vs probe ({pinfo['first_commit']}, {pinfo['last_commit']})"
                )
    manifest_df.to_csv(out_dir / "full_releases_manifest.csv", index=False)
    (out_dir / "full_releases_manifest.json").write_text(
        json.dumps(all_rows, indent=2), encoding="utf-8"
    )
    (out_dir / "full_newcode_lines.json").write_text(
        json.dumps(all_newcode), encoding="utf-8"
    )
    print(
        f"=== MANIFEST DONE: {manifest_df['repository'].nunique()} repos, "
        f"{len(manifest_df)} snapshots, {(manifest_df['seq_idx'] > 1).sum()} transitions ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
