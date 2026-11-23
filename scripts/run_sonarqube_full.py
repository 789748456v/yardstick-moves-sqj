#!/usr/bin/env python3
"""
Full-study chronological SonarQube runner across the 49 N09 Python 3 repositories (882 consecutive releases).

Executes 4 physical container regimes (yielding 7 exact taxonomic/profile regimes):
  1. sq99_std      : SonarQube 9.9.8-community ('Sonar way' 183 rules, Standard Rating Gate)
  2. sq103_n09     : SonarQube 10.3.0-community ('Sonar way' 218 rules, exact N09 Baseline + CleanCode dual view)
  3. sq26_mqr      : SonarQube 26.11.0.134175-community ('Sonar way core' default 232 rules, MQR + exact Standard dual view)
  4. sq26_comp_mqr : SonarQube 26.11.0.134175-community ('Sonar way comprehensive' 407 rules, MQR + exact Standard dual view)

Runs in 2-container batches (--cpus=3.5 --memory=6g each => <= 7 vCPU / 12 GB RAM total, localhost ports 9001..9004).
Automatically reuses complete per-repository sequences already persisted on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sonarqube_client import SonarQubeInstance

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))

REGIMES = [
    {
        "label": "sq99_std",
        "image": "sonarqube:9.9.8-community",
        "port": 9001,
        "scanner": BASE_DIR / "scanner" / "sonar-scanner-5.0.1.3006-linux" / "bin" / "sonar-scanner",
        "mode": None,
        "profile_name": None,
    },
    {
        "label": "sq103_n09",
        "image": "sonarqube:10.3.0-community",
        "port": 9002,
        "scanner": BASE_DIR / "scanner" / "sonar-scanner-5.0.1.3006-linux" / "bin" / "sonar-scanner",
        "mode": None,
        "profile_name": None,
    },
    {
        "label": "sq26_mqr",
        "image": "sonarqube:26.11.0.134175-community",
        "port": 9003,
        "scanner": BASE_DIR / "scanner" / "sonar-scanner-6.2.1.4610-linux-x64" / "bin" / "sonar-scanner",
        "mode": "MQR",
        "profile_name": None,  # Uses built-in default 'Sonar way core' (232 rules)
    },
    {
        "label": "sq26_comp_mqr",
        "image": "sonarqube:26.11.0.134175-community",
        "port": 9004,
        "scanner": BASE_DIR / "scanner" / "sonar-scanner-6.2.1.4610-linux-x64" / "bin" / "sonar-scanner",
        "mode": "MQR",
        "profile_name": "Sonar way comprehensive",  # Full 407-rule built-in profile ablation
    },
]


def is_repo_complete(out_regime_dir: Path, repo_records: List[Dict[str, Any]]) -> bool:
    for rec in repo_records:
        safe_snap = rec["snapshot_id"].replace("::", "__").replace("/", "_")
        if not (out_regime_dir / f"{safe_snap}.json").exists():
            return False
    return True


def scan_repo_sequence(
    sq: SonarQubeInstance,
    slug: str,
    repo_records: List[Dict[str, Any]],
    out_regime_dir: Path,
) -> Dict[str, Any]:
    if is_repo_complete(out_regime_dir, repo_records):
        return {"slug": slug, "n": len(repo_records), "cached": True, "sec": 0.0}

    t0 = time.time()
    repo_dir = BASE_DIR / "repos" / slug
    project_key = f"d04p_{sq.label}_{slug}"
    sq.ensure_project_previous_version_newcode(project_key, project_key)

    wt_dir = BASE_DIR / "tmp_sq_wt" / f"{sq.label}_{slug}"
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    if wt_dir.exists():
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)],
            capture_output=True,
        )
    subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "add", "--detach", "--quiet", str(wt_dir), repo_records[0]["commit"]],
        check=True,
        capture_output=True,
    )

    last_dt_epoch = 0.0
    try:
        for rec in repo_records:
            safe_snap = rec["snapshot_id"].replace("::", "__").replace("/", "_")
            out_file = out_regime_dir / f"{safe_snap}.json"

            dt = datetime.fromisoformat(rec["published_at"].replace("Z", "+00:00")).astimezone(timezone.utc)
            if dt.timestamp() <= last_dt_epoch:
                dt = datetime.fromtimestamp(last_dt_epoch + 120, tz=timezone.utc)
            last_dt_epoch = dt.timestamp()
            sq_date = dt.strftime("%Y-%m-%dT%H:%M:%S+0000")

            subprocess.run(
                ["git", "-C", str(wt_dir), "checkout", "--force", "--quiet", rec["commit"]],
                check=True,
            )
            subprocess.run(["git", "-C", str(wt_dir), "clean", "-fdx", "--quiet"], check=True)

            scan_res = sq.scan_snapshot_and_collect(
                wt_dir=wt_dir,
                project_key=project_key,
                release_tag=rec["release_tag"],
                project_date_iso=sq_date,
            )
            payload = {
                "regime": sq.label,
                "snapshot_id": rec["snapshot_id"],
                "repository": rec["repository"],
                "slug": slug,
                "seq_idx": rec["seq_idx"],
                "release_tag": rec["release_tag"],
                "commit": rec["commit"],
                "sq_project_date": sq_date,
                **scan_res,
            }
            out_file.write_text(json.dumps(payload), encoding="utf-8")
        print(
            f"  [{sq.label}] DONE {slug} ({len(repo_records)} snaps) in {time.time()-t0:.1f}s",
            flush=True,
        )
    finally:
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)],
            capture_output=True,
        )
    return {"slug": slug, "n": len(repo_records), "cached": False, "sec": round(time.time() - t0, 1)}


def run_one_regime(
    cfg: Dict[str, Any],
    by_slug: Dict[str, List[Dict[str, Any]]],
    sq_out_root: Path,
    catalog_dir: Path,
    max_repo_workers: int,
    cpus: str,
    memory: str,
) -> None:
    label = cfg["label"]
    out_regime_dir = sq_out_root / label
    out_regime_dir.mkdir(parents=True, exist_ok=True)

    pending_slugs = [slug for slug, recs in by_slug.items() if not is_repo_complete(out_regime_dir, recs)]
    pending_slugs.sort(key=lambda s: int(by_slug[s][-1].get("py_sloc", 0)))
    if not pending_slugs and (catalog_dir / f"{label}_rules.json").exists():
        print(f"=== [{label}] All {len(by_slug)} repos already complete on disk, skipping ===", flush=True)
        return

    print(
        f"=== Starting [{label}] ({cfg['image']}, mode={cfg['mode']}, profile={cfg['profile_name']}, pending_repos={len(pending_slugs)}/{len(by_slug)}) ===",
        flush=True,
    )
    sq = SonarQubeInstance(
        label=label,
        image=cfg["image"],
        port=cfg["port"],
        scanner_bin=cfg["scanner"],
        mode=cfg["mode"],
        profile_name=cfg["profile_name"],
        cpus=cpus,
        memory=memory,
    )
    sq.start_clean()
    try:
        meta = sq.get_mode_and_gates_metadata()
        rules = sq.fetch_active_python_rules()
        (catalog_dir / f"{label}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (catalog_dir / f"{label}_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
        print(
            f"  [{label}] UP! version={meta.get('sq_version')} profile={cfg['profile_name'] or 'default'} active_py_rules={len(rules)}",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=max_repo_workers) as ex:
            futs = {
                ex.submit(scan_repo_sequence, sq, slug, by_slug[slug], out_regime_dir): slug
                for slug in pending_slugs
            }
            done_cnt = 0
            for fut in as_completed(futs):
                slug = futs[fut]
                res = fut.result()
                done_cnt += 1
                print(
                    f"  [{label} {done_cnt}/{len(pending_slugs)}] repo={slug} sec={res['sec']}",
                    flush=True,
                )
    finally:
        sq.stop_and_remove()


def main() -> None:
    import os
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", nargs="*", default=[r["label"] for r in REGIMES])
    parser.add_argument("--repos", nargs="*", default=None, help="Optional subset of repository names/slugs")
    parser.add_argument("--parallel-containers", type=int, default=2, help="Max concurrent SonarQube containers (<=2 keeps <=7 vCPU / 12GB RAM)")
    parser.add_argument("--max-repo-workers", type=int, default=2)
    parser.add_argument("--cpus", type=str, default=None)
    parser.add_argument("--memory", type=str, default=None)
    args = parser.parse_args()

    manifest = json.loads((BASE_DIR / "data" / "full_releases_manifest.json").read_text(encoding="utf-8"))
    if args.repos:
        wanted = set(args.repos)
        manifest = [r for r in manifest if r["repository"] in wanted or r["slug"] in wanted]

    by_slug: Dict[str, List[Dict[str, Any]]] = {}
    for rec in manifest:
        by_slug.setdefault(rec["slug"], []).append(rec)
    for slug in by_slug:
        by_slug[slug].sort(key=lambda x: x["seq_idx"])

    sq_out_root = BASE_DIR / "data" / "sq_runs"
    sq_out_root.mkdir(parents=True, exist_ok=True)
    catalog_dir = BASE_DIR / "data" / "sq_catalogs"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    selected_cfgs = [c for c in REGIMES if c["label"] in args.regimes]
    par = max(1, min(args.parallel_containers, 2))
    cpus_per = args.cpus or ("3.5" if par == 2 else "4.0")
    mem_per = args.memory or ("6.5g" if par == 2 else "7g")

    for i in range(0, len(selected_cfgs), par):
        batch = selected_cfgs[i : i + par]
        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futs = [
                ex.submit(
                    run_one_regime,
                    cfg,
                    by_slug,
                    sq_out_root,
                    catalog_dir,
                    args.max_repo_workers,
                    cpus_per,
                    mem_per,
                )
                for cfg in batch
            ]
            for f in futs:
                f.result()


if __name__ == "__main__":
    main()
