#!/usr/bin/env python3
"""
Publication-Grade Memory-Bounded Repo-Streaming Statistical, Causal 4D Attribution, Multi-Gate Sensitivity,
Dual-Track Calibration, and 100-Slice Source-Code Root-Cause Analyzer for D04-Plus 49-Repo Study (SQJ Revision).

Key Methodological & Reproducibility Guarantees:
  1. Portable `BASE_DIR` via `D04_BASE_DIR` env var or relative repository root (zero hardcoded paths).
  2. Static enabled-rule catalogs (`data/sq_catalogs/*.json` and `data/linter_catalogs.json`) used for RQ3 4D
     decomposition and RQ4 S1 (`R_cap`) filtering — zero hindsight leakage from post-switch releases.
  3. Pair-specific canonical correspondence for `Pylint <-> Ruff (PL)` (16 cross-prefix groups + `PL:*`, with
     no false `W0613 <-> B007` mapping) vs. exact 1-to-1 code matching for `Flake8 <-> Ruff` and within-tool pairs.
  4. Explicit separation of SonarQube `SECURITY_HOTSPOT -> ISSUE` API-channel reclassification (32–33 rules) from
     non-hotspot `/api/issues/search` detector drift (`table3b_hotspot_sensitivity.csv`), plus metric-weighted
     `+B_cap / -D_cap` decomposition satisfying `+B_cap^(w) - D_cap^(w) == E_impl_drift` identically.
  5. RQ2 reports raw `p < 0.05`, Benjamini-Hochberg FDR `q < 0.05`, Hamed-Rao autocorrelation-adjusted `p_HR < 0.05`,
     and true unrounded Theil-Sen slope sign inversions.
  6. RQ4 Track A separates the algebraic 1-anchor count-ratchet property from persistent-anchor fingerprint
     ratcheting and rolling step-fingerprint strict gating, with count-matched `S1` comparators; Track B reports
     both `Wilcoxon p vs. S0` and direct paired `Wilcoxon p vs. S1` across Early, Mid, and Late horizons.
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, spearmanr, theilslopes, wilcoxon

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))
SQ_PHYS_REGIMES = ("sq99_std", "sq103_n09", "sq26_mqr", "sq26_comp_mqr")

# Documented cross-prefix correspondence between Pylint msgids and Ruff's enabled non-PL codes in
# `--select PL,E9,F401,F404,F601,F701,F702,F704,F706,F811,F821,F822,F841,E711,E712,E722,B006,B007,B018,C901`
# Note: W0613 (unused-argument) corresponds to Ruff ARG001..005 (not enabled in PL selector), NOT B007!
# Meanwhile W0612 (unused-variable) in Pylint covers both local variables (F841) and loop variables (B007),
# E0103 (not-in-loop) covers both break (F701) and continue (F702),
# C0121 (singleton-comparison) covers both None (E711) and bool (E712),
# and F811 covers both E0102 (function-redefined) and W0404 (reimported).
PYLINT_CROSS_TO_CANON: Dict[str, str] = {
    "W0611": "EQ:F401_UNUSED_IMPORT",
    "W0410": "EQ:F404_LATE_FUTURE",
    "W0109": "EQ:F601_DUP_KEY",
    "E0103": "EQ:E0103_NOT_IN_LOOP",
    "E0105": "EQ:F704_YIELD_OUTSIDE",
    "E0104": "EQ:F706_RETURN_OUTSIDE",
    "E0102": "EQ:F811_REDEFINED",
    "W0404": "EQ:F811_REDEFINED",
    "E0602": "EQ:F821_UNDEFINED_NAME",
    "E0603": "EQ:F822_UNDEFINED_ALL",
    "W0612": "EQ:W0612_UNUSED_VAR",
    "C0121": "EQ:C0121_SINGLETON_CMP",
    "W0702": "EQ:E722_BARE_EXCEPT",
    "W0102": "EQ:B006_MUTABLE_DEFAULT",
    "W0104": "EQ:B018_POINTLESS_STMT",
    "R1260": "EQ:C901_COMPLEXITY",
    "E0001": "EQ:E999_SYNTAX_ERR",
}

RUFF_CROSS_TO_CANON_PYLINT: Dict[str, str] = {
    "F401": "EQ:F401_UNUSED_IMPORT",
    "F404": "EQ:F404_LATE_FUTURE",
    "F601": "EQ:F601_DUP_KEY",
    "F701": "EQ:E0103_NOT_IN_LOOP",
    "F702": "EQ:E0103_NOT_IN_LOOP",
    "F704": "EQ:F704_YIELD_OUTSIDE",
    "F706": "EQ:F706_RETURN_OUTSIDE",
    "F811": "EQ:F811_REDEFINED",
    "F821": "EQ:F821_UNDEFINED_NAME",
    "F822": "EQ:F822_UNDEFINED_ALL",
    "F841": "EQ:W0612_UNUSED_VAR",
    "B007": "EQ:W0612_UNUSED_VAR",
    "E711": "EQ:C0121_SINGLETON_CMP",
    "E712": "EQ:C0121_SINGLETON_CMP",
    "E722": "EQ:E722_BARE_EXCEPT",
    "B006": "EQ:B006_MUTABLE_DEFAULT",
    "B018": "EQ:B018_POINTLESS_STMT",
    "C901": "EQ:C901_COMPLEXITY",
    "E999": "EQ:E999_SYNTAX_ERR",
    "SYNTAX_ERR": "EQ:E999_SYNTAX_ERR",
}


def canonical_linter_rule(pair_mode: str, tool: str, raw_rule: str) -> str:
    """
    Pair-specific canonical rule identifier:
    - `pair_mode == 'pylint_ruff'`: maps Pylint msgids and Ruff PL + 19 documented cross-prefix codes into
      16 cross-prefix canonical groups + `PL:<msgid>` (without conflating W0613 with B007).
    - `pair_mode == 'direct'`: preserves exact 1-to-1 rule codes for Flake8<->Ruff and within-linter upgrades
      (normalizing Ruff's fallback 'SYNTAX_ERR' string to 'E999').
    """
    if pair_mode == "pylint_ruff":
        if tool.startswith("pylint"):
            if raw_rule in PYLINT_CROSS_TO_CANON:
                return PYLINT_CROSS_TO_CANON[raw_rule]
            return f"PL:{raw_rule}"
        if tool.startswith("ruff"):
            if raw_rule in RUFF_CROSS_TO_CANON_PYLINT:
                return RUFF_CROSS_TO_CANON_PYLINT[raw_rule]
            if raw_rule.startswith(("PLC", "PLE", "PLR", "PLW")):
                return f"PL:{raw_rule[2:]}"
            return f"RUFF:{raw_rule}"
    if raw_rule == "SYNTAX_ERR":
        return "E999"
    return raw_rule


def is_raw_linter_rule_high(tool: str, code: str) -> bool:
    """Deterministic static rule-severity predicate matching `run_linters_full.py`."""
    if tool.startswith("flake8"):
        return (
            code.startswith(("F", "E9", "B9"))
            or code in {"B006", "B008", "B023", "E722"}
        )
    if tool.startswith("ruff"):
        return (
            code in {"SYNTAX_ERR", "E999"}
            or code.startswith(("F", "E9", "PLE", "B9"))
            or code in {"B006", "B008", "B023", "E722", "S102", "S301", "S307", "S506", "S602"}
        )
    if tool.startswith("pylint"):
        return code.startswith(("E", "F")) or code in {"W0102", "W0702"}
    return False


def sq_std_is_high(sev: str | None) -> bool:
    return (sev or "").upper() in {"CRITICAL", "BLOCKER"}


def sq_mqr_is_high(impacts: List[Dict[str, Any]] | None) -> bool:
    for imp in impacts or []:
        if (imp.get("severity") or "").upper() in {"HIGH", "BLOCKER"}:
            return True
    return False


def sq_std_is_rel_sec(itype: str | None) -> bool:
    return (itype or "").upper() in {"BUG", "VULNERABILITY"}


def sq_mqr_is_rel_sec(impacts: List[Dict[str, Any]] | None) -> bool:
    for imp in impacts or []:
        if (imp.get("softwareQuality") or "").upper() in {"RELIABILITY", "SECURITY"}:
            return True
    return False


def sq_mqr_is_rel_sec_high(impacts: List[Dict[str, Any]] | None) -> bool:
    for imp in impacts or []:
        if (imp.get("softwareQuality") or "").upper() in {"RELIABILITY", "SECURITY"} and (
            imp.get("severity") or ""
        ).upper() in {"HIGH", "BLOCKER"}:
            return True
    return False


def eval_sq_static_native_qg(qg_obj: Dict[str, Any]) -> bool:
    conds = qg_obj.get("conditions") or []
    static_conds = [
        c
        for c in conds
        if c.get("metricKey")
        not in {"new_coverage", "coverage", "new_duplicated_lines_density", "duplicated_lines_density"}
    ]
    if not static_conds:
        return qg_obj.get("status") == "OK"
    return all(c.get("status") == "OK" for c in static_conds)


def bh_fdr(pvals: List[float]) -> List[float]:
    n = len(pvals)
    if n == 0:
        return []
    arr = np.asarray(pvals, dtype=float)
    order = np.argsort(arr)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        val = min(prev, arr[idx] * n / (i + 1))
        ranked[idx] = val
        prev = val
    return ranked.tolist()


def hamed_rao_mk_p(y: np.ndarray) -> float:
    """Hamed & Rao (1998) variance-corrected Mann-Kendall two-sided p-value accounting for serial autocorrelation."""
    n = len(y)
    if n < 4 or np.all(y == y[0]):
        return 1.0
    s = 0
    for i in range(n - 1):
        diff = y[i + 1 :] - y[i]
        s += int(np.sum(diff > 0) - np.sum(diff < 0))
    # Tie-adjusted base variance Var(S)
    _, counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return 1.0
    # Detrend by Theil-Sen slope and compute rank autocorrelation lags
    x = np.arange(n, dtype=float)
    sl = float(theilslopes(y, x)[0])
    resid = y - sl * x
    ranks = pd.Series(resid).rank(method="average").to_numpy(dtype=float)
    r_mean = float(np.mean(ranks))
    denom = float(np.sum((ranks - r_mean) ** 2))
    corr_factor = 1.0
    if denom > 0:
        crit = 1.96 / np.sqrt(n)
        add_sum = 0.0
        for lag in range(1, n - 2):
            rk = float(np.sum((ranks[:-lag] - r_mean) * (ranks[lag:] - r_mean))) / denom
            if abs(rk) > crit:
                add_sum += (n - lag) * (n - lag - 1) * (n - lag - 2) * rk
        corr_factor = max(0.25, 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * add_sum)
    var_adj = var_s * corr_factor
    z = (s - 1) / np.sqrt(var_adj) if s > 0 else ((s + 1) / np.sqrt(var_adj) if s < 0 else 0.0)
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def bootstrap_ci_pct(binary_arr: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> Tuple[float, float]:
    if len(binary_arr) == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, len(binary_arr), size=(n_boot, len(binary_arr)))
    means = binary_arr[idx].mean(axis=1) * 100.0
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fetch_git_snippet_and_verbatim(
    repo_dir: Path, commit: str, rel_file: str, line_no: int, ctx: int = 1
) -> Tuple[str, str]:
    """Return both compact pipe-joined snippet AND exact verbatim untruncated Git source lines."""
    if not repo_dir.exists() or not rel_file or line_no <= 0:
        return "", ""
    res = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{commit}:{rel_file}"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if res.returncode != 0:
        return "", ""
    lines = res.stdout.splitlines()
    if line_no > len(lines):
        return "", ""
    lo = max(1, line_no - ctx)
    hi = min(len(lines), line_no + ctx)
    compact = []
    verbatim = []
    for ln in range(lo, hi + 1):
        marker = ">>" if ln == line_no else "  "
        compact.append(f"{marker} L{ln}: {lines[ln - 1].strip()[:120]}")
        verbatim.append(f"{ln}: {lines[ln - 1]}")
    return " | ".join(compact), "\n".join(verbatim)


def safe_snap_stem(sid: str) -> str:
    return sid.replace("::", "__").replace("/", "_")


def discover_observed_linter_rules(
    manifest_df: pd.DataFrame,
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Pass 1: Stream linter JSONs one snapshot at a time to collect raw observed code sets & total alert counts."""
    observed_raw_codes: Dict[str, Set[str]] = defaultdict(set)
    linter_alert_counts: Dict[str, int] = Counter()
    l_root = BASE_DIR / "data" / "linter_runs"

    for idx, sid in enumerate(manifest_df["snapshot_id"].tolist(), 1):
        jf = l_root / f"{safe_snap_stem(sid)}.json"
        if not jf.exists():
            continue
        d = json.loads(jf.read_text(encoding="utf-8"))
        for tname, issues in d.get("tools", {}).items():
            rset = observed_raw_codes[tname]
            linter_alert_counts[tname] += len(issues)
            for x in issues:
                rset.add(x["rule"])
        del d
        if idx % 200 == 0:
            gc.collect()
            print(f"  [Pass 1 telemetry] {idx}/{len(manifest_df)} snapshots scanned...", flush=True)
    gc.collect()
    return observed_raw_codes, dict(linter_alert_counts)


def load_single_repo_runs(
    sids: List[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    sq_repo: Dict[str, Dict[str, Dict[str, Any]]] = {reg: {} for reg in SQ_PHYS_REGIMES}
    sq_root = BASE_DIR / "data" / "sq_runs"
    l_root = BASE_DIR / "data" / "linter_runs"
    linter_repo: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)

    for sid in sids:
        stem = safe_snap_stem(sid)
        for reg in SQ_PHYS_REGIMES:
            jf = sq_root / reg / f"{stem}.json"
            if jf.exists():
                sq_repo[reg][sid] = json.loads(jf.read_text(encoding="utf-8"))
        ljf = l_root / f"{stem}.json"
        if ljf.exists():
            ld = json.loads(ljf.read_text(encoding="utf-8"))
            for tname, issues in ld.get("tools", {}).items():
                linter_repo[tname][sid] = issues
    return sq_repo, linter_repo


def analyze_full() -> None:
    out_dir = BASE_DIR / "results" / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261113)

    manifest_df = pd.read_csv(BASE_DIR / "data" / "full_releases_manifest.csv")
    raw_nc = json.loads((BASE_DIR / "data" / "full_newcode_lines.json").read_text(encoding="utf-8"))
    nc_map: Dict[str, Dict[str, Set[int]]] = {
        s: {fp: set(lns) for fp, lns in fd.items()} for s, fd in raw_nc.items()
    }
    del raw_nc
    gc.collect()

    sq_rules: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for p in (BASE_DIR / "data" / "sq_catalogs").glob("*_rules.json"):
        lbl = p.name.replace("_rules.json", "")
        sq_rules[lbl] = {r["key"]: r for r in json.loads(p.read_text(encoding="utf-8"))}

    linter_cat_file = BASE_DIR / "data" / "linter_catalogs.json"
    linter_static_catalogs: Dict[str, Set[str]] = {}
    if linter_cat_file.exists():
        raw_lcat = json.loads(linter_cat_file.read_text(encoding="utf-8"))
        for k, v in raw_lcat.items():
            linter_static_catalogs[k] = set(v.get("codes", []))

    print(f"Loaded manifest: {manifest_df['repository'].nunique()} repos, {len(manifest_df)} snapshots.", flush=True)

    # =========================================================================
    # TABLE 0: Cohort & Scale Tier Summary (Transition churn over seq_idx > 1!)
    # =========================================================================
    trans_churn_med = (
        manifest_df[manifest_df["seq_idx"] > 1]
        .groupby("repository")["py_lines_changed"]
        .median()
        .to_dict()
    )
    trans_all_churn_med = (
        manifest_df[manifest_df["seq_idx"] > 1]
        .groupby("repository")["all_lines_changed"]
        .median()
        .to_dict()
    )
    cohort_summary = (
        manifest_df.groupby(["repository", "in_n09_rq3_sig"])
        .agg(
            n_releases=("seq_idx", "count"),
            first_tag=("release_tag", "first"),
            last_tag=("release_tag", "last"),
            sloc_first=("py_sloc", "first"),
            sloc_last=("py_sloc", "last"),
            sloc_min=("py_sloc", "min"),
            sloc_max=("py_sloc", "max"),
        )
        .reset_index()
        .sort_values("repository")
    )
    cohort_summary["median_py_lines_changed"] = [
        round(float(trans_churn_med[rp]), 1) for rp in cohort_summary["repository"]
    ]
    cohort_summary["median_all_lines_changed"] = [
        round(float(trans_all_churn_med[rp]), 1) for rp in cohort_summary["repository"]
    ]
    cohort_summary["scale_tier"] = np.where(
        cohort_summary["sloc_last"] < 25_000,
        "Small (<25k SLOC)",
        np.where(cohort_summary["sloc_last"] <= 150_000, "Medium (25k-150k SLOC)", "Large/Monorepo (>150k SLOC)"),
    )
    cohort_summary.to_csv(out_dir / "table0_cohort_49repos.csv", index=False)
    repo_to_tier = dict(zip(cohort_summary["repository"], cohort_summary["scale_tier"]))

    # =========================================================================
    # PASS 1: Observed Linter Codes & Raw Alert Counts
    # =========================================================================
    observed_raw_codes, linter_alert_counts = discover_observed_linter_rules(manifest_df)
    # Static catalogs come strictly from exported binary/plugin registries (`data/linter_catalogs.json`)
    # with ZERO corpus-alert leakage and ZERO artificial parser-code injection.
    non_catalog_parser_diagnostics: Dict[str, Dict[str, int]] = {}
    for tname in observed_raw_codes.keys():
        if tname not in linter_static_catalogs:
            raise RuntimeError(f"Missing static catalog for {tname} in data/linter_catalogs.json")
        linter_static_catalogs[tname] = set(linter_static_catalogs[tname])
        extra_obs = sorted(observed_raw_codes[tname] - linter_static_catalogs[tname])
        if extra_obs:
            non_catalog_parser_diagnostics[tname] = {c: 1 for c in extra_obs}
    # Assert Flake8 and Pylint static catalogs cover 100% of observed raw codes (including Pylint `I*` messages)
    for strict_cat_tool in ("flake8_71", "pylint_217", "pylint_33", "ruff_01_default", "ruff_01_extended"):
        assert observed_raw_codes[strict_cat_tool].issubset(linter_static_catalogs[strict_cat_tool]), (
            f"Unexpected {strict_cat_tool} code outside static catalog: {observed_raw_codes[strict_cat_tool] - linter_static_catalogs[strict_cat_tool]}"
        )

    def get_linter_pair_sets(t0: str, t1: str, pair_mode: str) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str], Set[str], Set[str]]:
        cat0 = {canonical_linter_rule(pair_mode, t0, c) for c in linter_static_catalogs[t0]}
        cat1 = {canonical_linter_rule(pair_mode, t1, c) for c in linter_static_catalogs[t1]}
        obs0 = {canonical_linter_rule(pair_mode, t0, c) for c in (observed_raw_codes[t0] & linter_static_catalogs[t0])}
        obs1 = {canonical_linter_rule(pair_mode, t1, c) for c in (observed_raw_codes[t1] & linter_static_catalogs[t1])}
        t0_high_canon = {
            canonical_linter_rule(pair_mode, t0, c)
            for c in linter_static_catalogs[t0]
            if is_raw_linter_rule_high(t0, c)
        }
        return cat0, cat1, (cat0 & cat1), obs0, obs1, (obs0 & obs1), t0_high_canon

    sq_alert_counts: Dict[str, int] = Counter()
    observed_sq_rules: Dict[str, Set[str]] = defaultdict(set)
    verified_sq_scans_loaded = 0
    verified_linter_scans_loaded = 0

    sq_derived_specs = [
        ("sq99_std", "sq99_std", False),
        ("sq103_n09", "sq103_std", False),
        ("sq103_n09", "sq103_cleancode", True),
        ("sq26_mqr", "sq26_core_std", False),
        ("sq26_mqr", "sq26_core_mqr", True),
        ("sq26_comp_mqr", "sq26_comp_std", False),
        ("sq26_comp_mqr", "sq26_comp_mqr", True),
    ]
    sq_4d_specs = [
        ("SQ 10.3 [N09] -> SQ 26.11 Core MQR", "HIGH", "sq103_n09", False, "sq26_mqr", True),
        ("SQ 10.3 [N09] -> SQ 26.11 Comp MQR", "HIGH", "sq103_n09", False, "sq26_comp_mqr", True),
        ("SQ 9.9 LTA -> SQ 26.11 Core MQR", "HIGH", "sq99_std", False, "sq26_mqr", True),
        ("SQ 9.9 LTA -> SQ 26.11 Comp MQR", "HIGH", "sq99_std", False, "sq26_comp_mqr", True),
        ("SQ 26.11 Core: Std -> MQR", "HIGH", "sq26_mqr", False, "sq26_mqr", True),
        ("SQ 26.11 Comp: Std -> MQR", "HIGH", "sq26_comp_mqr", False, "sq26_comp_mqr", True),
        ("SQ 26.11 Core: Std -> MQR (Rel/Sec)", "REL_SEC", "sq26_mqr", False, "sq26_mqr", True),
        ("SQ 26.11 Comp: Std -> MQR (Rel/Sec)", "REL_SEC", "sq26_comp_mqr", False, "sq26_comp_mqr", True),
        ("SQ 10.3 [N09] -> SQ 26.11 Comp MQR (Rel/Sec)", "REL_SEC", "sq103_n09", False, "sq26_comp_mqr", True),
    ]
    linter_4d_specs = [
        ("Flake8 7.1 -> Ruff 0.17.5 (Extended)", "HIGH", "flake8_71", "ruff_modern_extended", "direct"),
        ("Flake8 7.1 -> Ruff 0.17.5 (Extended)", "TOTAL", "flake8_71", "ruff_modern_extended", "direct"),
        ("Pylint 3.3 -> Ruff 0.17.5 (Pylint-PL Aligned)", "HIGH", "pylint_33", "ruff_modern_pylint", "pylint_ruff"),
        ("Pylint 3.3 -> Ruff 0.17.5 (Pylint-PL Aligned)", "TOTAL", "pylint_33", "ruff_modern_pylint", "pylint_ruff"),
        ("Pylint 2.17 -> Pylint 3.3", "TOTAL", "pylint_217", "pylint_33", "direct"),
        ("Ruff 0.1.15 -> Ruff 0.17.5 (Extended)", "HIGH", "ruff_01_extended", "ruff_modern_extended", "direct"),
    ]
    calib_pairs = [
        ("SQ 10.3 [N09] -> SQ 26.11 Core MQR", "sq103_std", "sq26_core_mqr", "sq"),
        ("SQ 10.3 [N09] -> SQ 26.11 Comp MQR", "sq103_std", "sq26_comp_mqr", "sq"),
        ("SQ 9.9 LTA Std -> SQ 26.11 Core MQR", "sq99_std", "sq26_core_mqr", "sq"),
        ("Flake8 7.1 -> Ruff 0.17.5 (Extended)", "flake8_71", "ruff_modern_extended", "direct"),
        ("Pylint 3.3 -> Ruff 0.17.5 (Pylint-PL Aligned)", "pylint_33", "ruff_modern_pylint", "pylint_ruff"),
    ]
    phys_sq_lookup = {
        "sq99_std": ("sq99_std", False),
        "sq103_std": ("sq103_n09", False),
        "sq26_core_mqr": ("sq26_mqr", True),
        "sq26_comp_mqr": ("sq26_comp_mqr", True),
    }

    snap_rows: List[Dict[str, Any]] = []
    decomp_rows_by_spec: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    hs_sens_rows_by_spec: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    # Store both density and raw counts on static catalog R_cap for calibration (fixes count-vs-density comparator!)
    rcap_series_map: Dict[Tuple[str, str], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    # Store pair-matched high-severity fingerprints for Track A
    high_fp_pair_map: Dict[Tuple[str, str, str], List[Counter[Tuple[str, str, str]]]] = {}

    cand_s1: List[Dict[str, Any]] = []
    cand_s2_hs: List[Dict[str, Any]] = []
    cand_s2_births_nonhs: List[Dict[str, Any]] = []
    cand_s2_drops: List[Dict[str, Any]] = []
    cand_s3: List[Dict[str, Any]] = []
    cand_s4: List[Dict[str, Any]] = []

    # =========================================================================
    # PASS 2: Single Repo-by-Repo Streaming Pass across 49 Repositories
    # =========================================================================
    all_repos_sorted = sorted(manifest_df["repository"].unique())
    for repo_idx, repo in enumerate(all_repos_sorted, 1):
        grp = manifest_df[manifest_df["repository"] == repo].sort_values("seq_idx")
        snaps = grp["snapshot_id"].tolist()
        slug = str(grp["slug"].iloc[0])
        repo_dir = BASE_DIR / "repos" / slug
        sq_repo, linter_repo = load_single_repo_runs(snaps)

        for reg_p in SQ_PHYS_REGIMES:
            verified_sq_scans_loaded += len(sq_repo[reg_p])
            for pay in sq_repo[reg_p].values():
                iss_p = pay.get("issues", [])
                sq_alert_counts[reg_p] += len(iss_p)
                rset_sq = observed_sq_rules[reg_p]
                for x_sq in iss_p:
                    rset_sq.add(x_sq["rule"])
        for t_l_map in linter_repo.values():
            verified_linter_scans_loaded += len(t_l_map)

        # 2A. Unified Snapshot Metrics
        for _, mrow in grp.iterrows():
            sid = mrow["snapshot_id"]
            seq_idx = int(mrow["seq_idx"])
            ksloc = max(float(mrow["py_sloc"]) / 1000.0, 0.001)
            nc_files = nc_map.get(sid, {})

            for phys_reg, reg_name, is_mqr_view in sq_derived_specs:
                if sid not in sq_repo.get(phys_reg, {}):
                    continue
                payload = sq_repo[phys_reg][sid]
                issues = payload["issues"]
                rmap = sq_rules.get(phys_reg, {})
                qg_ok = eval_sq_static_native_qg(payload["quality_gate"])
                m_dict = {
                    m["metric"]: m.get("value", m.get("period", {}).get("value"))
                    for m in payload.get("measures", {}).get("measures", [])
                }

                extracted_tot_cnt = len(issues)
                tot_cnt = (
                    max(extracted_tot_cnt, int(m_dict["violations"]))
                    if m_dict.get("violations") is not None
                    else extracted_tot_cnt
                )
                new_cnt = sum(1 for x in issues if x["is_new"])
                high_cnt, new_high_cnt = 0, 0
                rel_sec_cnt, new_rel_sec_cnt = 0, 0
                rel_sec_high_cnt, new_rel_sec_high_cnt = 0, 0

                for x in issues:
                    rk = x["rule"]
                    rinfo = rmap.get(rk, {})
                    if is_mqr_view:
                        imps = x.get("impacts") or rinfo.get("impacts") or []
                        is_h = sq_mqr_is_high(imps)
                        is_rs = sq_mqr_is_rel_sec(imps)
                        is_rsh = sq_mqr_is_rel_sec_high(imps)
                    else:
                        sev = x.get("severity") or rinfo.get("severity")
                        itype = x.get("type") or rinfo.get("type")
                        is_h = sq_std_is_high(sev)
                        is_rs = sq_std_is_rel_sec(itype)
                        is_rsh = is_h and is_rs
                    if is_h:
                        high_cnt += 1
                        if x["is_new"]:
                            new_high_cnt += 1
                    if is_rs:
                        rel_sec_cnt += 1
                        if x["is_new"]:
                            new_rel_sec_cnt += 1
                    if is_rsh:
                        rel_sec_high_cnt += 1
                        if x["is_new"]:
                            new_rel_sec_high_cnt += 1

                # Reconcile aggregate Standard BUG+VULNERABILITY count against server measures when >10k sharding truncated MINOR issues
                if (not is_mqr_view) and m_dict.get("bugs") is not None and m_dict.get("vulnerabilities") is not None:
                    rel_sec_cnt = max(rel_sec_cnt, int(m_dict["bugs"]) + int(m_dict["vulnerabilities"]))
                elif is_mqr_view and m_dict.get("vulnerabilities") is not None:
                    # Every VULNERABILITY in SQ 26.11 carries a SECURITY impact; account for any unextracted non-HIGH vulnerabilities
                    ext_vuln = sum(1 for x in issues if (x.get("type") or rmap.get(x["rule"], {}).get("type")) == "VULNERABILITY")
                    missing_vuln = max(0, int(m_dict["vulnerabilities"]) - ext_vuln)
                    rel_sec_cnt += missing_vuln

                snap_rows.append(
                    {
                        "snapshot_id": sid,
                        "repository": repo,
                        "slug": slug,
                        "in_n09_rq3_sig": bool(mrow["in_n09_rq3_sig"]),
                        "seq_idx": seq_idx,
                        "series_len": int(mrow["series_len"]),
                        "release_tag": mrow["release_tag"],
                        "py_sloc": int(mrow["py_sloc"]),
                        "all_modified_files": int(mrow["all_modified_files"]),
                        "all_lines_changed": int(mrow["all_lines_changed"]),
                        "py_lines_changed": int(mrow["py_lines_changed"]),
                        "family": "sonarqube",
                        "regime": reg_name,
                        "native_qg_pass": None if seq_idx == 1 else qg_ok,
                        "cayc_zero_new_pass": None if seq_idx == 1 else (new_cnt == 0),
                        "new_high_gate_pass": None if seq_idx == 1 else (new_high_cnt == 0),
                        "new_rel_sec_gate_pass": None if seq_idx == 1 else (new_rel_sec_cnt == 0),
                        "new_rel_sec_high_gate_pass": None if seq_idx == 1 else (new_rel_sec_high_cnt == 0),
                        "total_issues": tot_cnt,
                        "extracted_total_issues": extracted_tot_cnt,
                        "new_issues": new_cnt,
                        "high_issues": high_cnt,
                        "new_high_issues": new_high_cnt,
                        "rel_sec_issues": rel_sec_cnt,
                        "new_rel_sec_issues": new_rel_sec_cnt,
                        "density_total_ksloc": tot_cnt / ksloc,
                        "density_high_ksloc": high_cnt / ksloc,
                        "density_rel_sec_ksloc": rel_sec_cnt / ksloc,
                    }
                )

            for tname, tmap in linter_repo.items():
                if sid not in tmap:
                    continue
                issues = tmap[sid]
                tot_cnt = len(issues)
                high_cnt = sum(1 for x in issues if x["severity"] == "HIGH")
                new_tot = sum(1 for x in issues if int(x.get("line") or 0) in nc_files.get(x["file"], set()))
                new_high = sum(
                    1
                    for x in issues
                    if x["severity"] == "HIGH" and int(x.get("line") or 0) in nc_files.get(x["file"], set())
                )
                snap_rows.append(
                    {
                        "snapshot_id": sid,
                        "repository": repo,
                        "slug": slug,
                        "in_n09_rq3_sig": bool(mrow["in_n09_rq3_sig"]),
                        "seq_idx": seq_idx,
                        "series_len": int(mrow["series_len"]),
                        "release_tag": mrow["release_tag"],
                        "py_sloc": int(mrow["py_sloc"]),
                        "all_modified_files": int(mrow["all_modified_files"]),
                        "all_lines_changed": int(mrow["all_lines_changed"]),
                        "py_lines_changed": int(mrow["py_lines_changed"]),
                        "family": "linter",
                        "regime": tname,
                        "native_qg_pass": None if seq_idx == 1 else (new_high == 0),
                        "cayc_zero_new_pass": None if seq_idx == 1 else (new_tot == 0),
                        "new_high_gate_pass": None if seq_idx == 1 else (new_high == 0),
                        "new_rel_sec_gate_pass": None if seq_idx == 1 else (new_high == 0),
                        "new_rel_sec_high_gate_pass": None if seq_idx == 1 else (new_high == 0),
                        "total_issues": tot_cnt,
                        "new_issues": new_tot,
                        "high_issues": high_cnt,
                        "new_high_issues": new_high_cnt if False else new_high,
                        "rel_sec_issues": high_cnt,
                        "new_rel_sec_issues": new_high,
                        "density_total_ksloc": tot_cnt / ksloc,
                        "density_high_ksloc": high_cnt / ksloc,
                        "density_rel_sec_ksloc": high_cnt / ksloc,
                    }
                )

        # 2B. SQ 4D Decomposition + Hotspot-Migrated Rule Sensitivity
        for family_name, target_kind, phys0, mqr0, phys1, mqr1 in sq_4d_specs:
            if not all(s in sq_repo.get(phys0, {}) and s in sq_repo.get(phys1, {}) for s in snaps):
                continue
            cat0 = sq_rules.get(phys0, {})
            cat1 = sq_rules.get(phys1, {})
            r0_keys, r1_keys = set(cat0.keys()), set(cat1.keys())
            r_cap = r0_keys & r1_keys
            hs_migrated = {
                k for k in r_cap if (cat0[k].get("type") == "SECURITY_HOTSPOT") != (cat1[k].get("type") == "SECURITY_HOTSPOT")
            }
            r_cap_no_hs = r_cap - hs_migrated

            def weight_under_t0(iss: Dict[str, Any], from_1: bool) -> int:
                rk = iss["rule"]
                rinfo = (cat0.get(rk) or cat1.get(rk) or {}) if from_1 else (cat0.get(rk) or {})
                if target_kind == "HIGH":
                    if mqr0:
                        return 1 if sq_mqr_is_high(iss.get("impacts") or rinfo.get("impacts") or []) else 0
                    sev = (rinfo.get("severity") if from_1 else iss.get("severity")) or iss.get("severity") or rinfo.get("severity")
                    return 1 if sq_std_is_high(sev) else 0
                if mqr0:
                    return 1 if sq_mqr_is_rel_sec(iss.get("impacts") or rinfo.get("impacts") or []) else 0
                itype = (rinfo.get("type") if from_1 else iss.get("type")) or iss.get("type") or rinfo.get("type")
                return 1 if sq_std_is_rel_sec(itype) else 0

            def weight_under_t1(iss: Dict[str, Any]) -> int:
                rk = iss["rule"]
                rinfo = cat1.get(rk, {})
                if target_kind == "HIGH":
                    if mqr1:
                        return 1 if sq_mqr_is_high(iss.get("impacts") or rinfo.get("impacts") or []) else 0
                    return 1 if sq_std_is_high(iss.get("severity") or rinfo.get("severity")) else 0
                if mqr1:
                    return 1 if sq_mqr_is_rel_sec(iss.get("impacts") or rinfo.get("impacts") or []) else 0
                return 1 if sq_std_is_rel_sec(iss.get("type") or rinfo.get("type")) else 0

            n_steps = len(snaps) - 1
            abs_sums = np.zeros(4, dtype=float)
            abs_sums_split_hs = np.zeros(5, dtype=float)  # [code, rule, impl_nonhs, impl_hs, tax]
            gross_b_all, gross_d_all = 0, 0
            gross_b_w, gross_d_w = 0, 0
            gross_b_w_nonhs, gross_b_w_hs = 0, 0

            for i in range(1, len(snaps)):
                sp, sc = snaps[i - 1], snaps[i]
                i0_p = sq_repo[phys0][sp]["issues"]
                i0_c = sq_repo[phys0][sc]["issues"]
                i1_c = sq_repo[phys1][sc]["issues"]

                y0_p = sum(weight_under_t0(x, False) for x in i0_p)
                y0_c = sum(weight_under_t0(x, False) for x in i0_c)
                y1_c_t0 = sum(weight_under_t0(x, True) for x in i1_c)
                y1_c_t1 = sum(weight_under_t1(x) for x in i1_c)

                e_code = y0_c - y0_p
                e_rule = sum(weight_under_t0(x, True) for x in i1_c if x["rule"] not in r_cap) - sum(
                    weight_under_t0(x, False) for x in i0_c if x["rule"] not in r_cap
                )
                e_impl_nonhs = sum(weight_under_t0(x, True) for x in i1_c if x["rule"] in r_cap_no_hs) - sum(
                    weight_under_t0(x, False) for x in i0_c if x["rule"] in r_cap_no_hs
                )
                e_impl_hs = sum(weight_under_t0(x, True) for x in i1_c if x["rule"] in hs_migrated) - sum(
                    weight_under_t0(x, False) for x in i0_c if x["rule"] in hs_migrated
                )
                e_impl = e_impl_nonhs + e_impl_hs
                e_tax = y1_c_t1 - y1_c_t0
                d_obs = y1_c_t1 - y0_p
                assert (e_code + e_rule + e_impl + e_tax) == d_obs
                abs_sums += np.abs([e_code, e_rule, e_impl, e_tax])
                abs_sums_split_hs += np.abs([e_code, e_rule, e_impl_nonhs, e_impl_hs, e_tax])

                if phys0 != phys1:
                    c0_all = Counter((x["rule"], x["file"], x.get("hash") or str(x.get("line"))) for x in i0_c if x["rule"] in r_cap)
                    c1_all = Counter((x["rule"], x["file"], x.get("hash") or str(x.get("line"))) for x in i1_c if x["rule"] in r_cap)
                    for k in set(c0_all) | set(c1_all):
                        dk = c1_all[k] - c0_all[k]
                        if dk > 0:
                            gross_b_all += dk
                        elif dk < 0:
                            gross_d_all += -dk

                    c0_w = Counter((x["rule"], x["file"], x.get("hash") or str(x.get("line"))) for x in i0_c if x["rule"] in r_cap and weight_under_t0(x, False) == 1)
                    c1_w = Counter((x["rule"], x["file"], x.get("hash") or str(x.get("line"))) for x in i1_c if x["rule"] in r_cap and weight_under_t0(x, True) == 1)
                    for k in set(c0_w) | set(c1_w):
                        dk = c1_w[k] - c0_w[k]
                        if dk > 0:
                            gross_b_w += dk
                            if k[0] in hs_migrated:
                                gross_b_w_hs += dk
                            else:
                                gross_b_w_nonhs += dk
                        elif dk < 0:
                            gross_d_w += -dk

            raw_sum = float(abs_sums.sum())
            is_act = int(raw_sum > 0.0)
            denom = raw_sum if raw_sum > 0.0 else np.nan
            decomp_rows_by_spec[(family_name, target_kind)].append(
                {
                    "transition_family": family_name,
                    "metric_kind": target_kind,
                    "repository": repo,
                    "is_active_repo": is_act,
                    "r0_rules": len(r0_keys),
                    "r1_rules": len(r1_keys),
                    "common_rules_rcap": len(r_cap),
                    "r0_observed_rules": len(r0_keys),
                    "r1_observed_rules": len(r1_keys),
                    "rcap_observed_rules": len(r_cap),
                    "hs_migrated_rules_in_rcap": len(hs_migrated),
                    "mean_abs_E_code": round(float(abs_sums[0] / n_steps), 3),
                    "mean_abs_E_rule_set": round(float(abs_sums[1] / n_steps), 3),
                    "mean_abs_E_impl_drift": round(float(abs_sums[2] / n_steps), 3),
                    "mean_abs_E_tax_remap": round(float(abs_sums[3] / n_steps), 3),
                    "share_code_pct": round(100.0 * abs_sums[0] / denom, 2) if is_act else np.nan,
                    "share_ruleset_pct": round(100.0 * abs_sums[1] / denom, 2) if is_act else np.nan,
                    "share_impl_pct": round(100.0 * abs_sums[2] / denom, 2) if is_act else np.nan,
                    "share_tax_pct": round(100.0 * abs_sums[3] / denom, 2) if is_act else np.nan,
                    "rcap_births_per_snap": round(gross_b_all / n_steps, 2),
                    "rcap_deaths_per_snap": round(gross_d_all / n_steps, 2),
                    "rcap_weighted_births_per_snap": round(gross_b_w / n_steps, 2),
                    "rcap_weighted_deaths_per_snap": round(gross_d_w / n_steps, 2),
                }
            )
            if phys0 != phys1:
                tot5 = float(abs_sums_split_hs.sum())
                d5 = tot5 if tot5 > 0.0 else np.nan
                hs_sens_rows_by_spec[(family_name, target_kind)].append(
                    {
                        "transition_family": family_name,
                        "metric_kind": target_kind,
                        "repository": repo,
                        "is_active": int(tot5 > 0.0),
                        "rcap_total": len(r_cap),
                        "rcap_non_hotspot": len(r_cap_no_hs),
                        "rcap_hotspot_migrated": len(hs_migrated),
                        "share_code_pct": 100.0 * abs_sums_split_hs[0] / d5 if tot5 > 0 else np.nan,
                        "share_ruleset_pct": 100.0 * abs_sums_split_hs[1] / d5 if tot5 > 0 else np.nan,
                        "share_impl_nonhs_detector_pct": 100.0 * abs_sums_split_hs[2] / d5 if tot5 > 0 else np.nan,
                        "share_impl_hotspot_channel_pct": 100.0 * abs_sums_split_hs[3] / d5 if tot5 > 0 else np.nan,
                        "share_tax_pct": 100.0 * abs_sums_split_hs[4] / d5 if tot5 > 0 else np.nan,
                        "weighted_births_nonhs_per_snap": gross_b_w_nonhs / n_steps,
                        "weighted_births_hotspot_per_snap": gross_b_w_hs / n_steps,
                        "weighted_deaths_per_snap": gross_d_w / n_steps,
                    }
                )

        # 2C. Linter 4D Decomposition (Static Enabled Catalogs + Pair-Specific Correspondence)
        for family_name, target_kind, t0, t1, pair_mode in linter_4d_specs:
            if not all(s in linter_repo.get(t0, {}) and s in linter_repo.get(t1, {}) for s in snaps):
                continue
            r0_cat, r1_cat, r_cap, r0_obs, r1_obs, r_cap_obs, t0_high_set = get_linter_pair_sets(t0, t1, pair_mode)

            def w0(x: Dict[str, Any], tname: str, is_t1: bool) -> int:
                if target_kind == "TOTAL":
                    return 1
                cr = canonical_linter_rule(pair_mode, tname, x["rule"])
                if is_t1 and cr in r_cap:
                    return 1 if cr in t0_high_set else 0
                return 1 if x["severity"] == "HIGH" else 0

            def w1(x: Dict[str, Any]) -> int:
                if target_kind == "TOTAL":
                    return 1
                return 1 if x["severity"] == "HIGH" else 0

            n_steps = len(snaps) - 1
            abs_sums = np.zeros(4, dtype=float)
            gross_b_all, gross_d_all = 0, 0
            gross_b_w, gross_d_w = 0, 0

            for i in range(1, len(snaps)):
                sp, sc = snaps[i - 1], snaps[i]
                i0_p, i0_c, i1_c = linter_repo[t0][sp], linter_repo[t0][sc], linter_repo[t1][sc]
                y0_p = sum(w0(x, t0, False) for x in i0_p)
                y0_c = sum(w0(x, t0, False) for x in i0_c)
                y1_c_w0 = sum(w0(x, t1, True) for x in i1_c)
                y1_c_w1 = sum(w1(x) for x in i1_c)

                e_code = y0_c - y0_p
                e_rule = sum(w0(x, t1, True) for x in i1_c if canonical_linter_rule(pair_mode, t1, x["rule"]) not in r_cap) - sum(
                    w0(x, t0, False) for x in i0_c if canonical_linter_rule(pair_mode, t0, x["rule"]) not in r_cap
                )
                e_impl = sum(w0(x, t1, True) for x in i1_c if canonical_linter_rule(pair_mode, t1, x["rule"]) in r_cap) - sum(
                    w0(x, t0, False) for x in i0_c if canonical_linter_rule(pair_mode, t0, x["rule"]) in r_cap
                )
                e_tax = y1_c_w1 - y1_c_w0
                assert (e_code + e_rule + e_impl + e_tax) == (y1_c_w1 - y0_p)
                abs_sums += np.abs([e_code, e_rule, e_impl, e_tax])

                c0_all = Counter(
                    (canonical_linter_rule(pair_mode, t0, x["rule"]), x["file"], x.get("lhash") or str(x["line"]))
                    for x in i0_c
                    if canonical_linter_rule(pair_mode, t0, x["rule"]) in r_cap
                )
                c1_all = Counter(
                    (canonical_linter_rule(pair_mode, t1, x["rule"]), x["file"], x.get("lhash") or str(x["line"]))
                    for x in i1_c
                    if canonical_linter_rule(pair_mode, t1, x["rule"]) in r_cap
                )
                for k in set(c0_all) | set(c1_all):
                    dk = c1_all[k] - c0_all[k]
                    if dk > 0:
                        gross_b_all += dk
                    elif dk < 0:
                        gross_d_all += -dk

                c0_w = Counter(
                    (canonical_linter_rule(pair_mode, t0, x["rule"]), x["file"], x.get("lhash") or str(x["line"]))
                    for x in i0_c
                    if canonical_linter_rule(pair_mode, t0, x["rule"]) in r_cap and w0(x, t0, False) == 1
                )
                c1_w = Counter(
                    (canonical_linter_rule(pair_mode, t1, x["rule"]), x["file"], x.get("lhash") or str(x["line"]))
                    for x in i1_c
                    if canonical_linter_rule(pair_mode, t1, x["rule"]) in r_cap and w0(x, t1, True) == 1
                )
                for k in set(c0_w) | set(c1_w):
                    dk = c1_w[k] - c0_w[k]
                    if dk > 0:
                        gross_b_w += dk
                    elif dk < 0:
                        gross_d_w += -dk

            raw_sum = float(abs_sums.sum())
            is_act = int(raw_sum > 0.0)
            denom = raw_sum if raw_sum > 0.0 else np.nan
            decomp_rows_by_spec[(family_name, target_kind)].append(
                {
                    "transition_family": family_name,
                    "metric_kind": target_kind,
                    "repository": repo,
                    "is_active_repo": is_act,
                    "r0_rules": len(r0_cat),
                    "r1_rules": len(r1_cat),
                    "common_rules_rcap": len(r_cap),
                    "r0_observed_rules": len(r0_obs),
                    "r1_observed_rules": len(r1_obs),
                    "rcap_observed_rules": len(r_cap_obs),
                    "hs_migrated_rules_in_rcap": 0,
                    "mean_abs_E_code": round(float(abs_sums[0] / n_steps), 3),
                    "mean_abs_E_rule_set": round(float(abs_sums[1] / n_steps), 3),
                    "mean_abs_E_impl_drift": round(float(abs_sums[2] / n_steps), 3),
                    "mean_abs_E_tax_remap": round(float(abs_sums[3] / n_steps), 3),
                    "share_code_pct": round(100.0 * abs_sums[0] / denom, 2) if is_act else np.nan,
                    "share_ruleset_pct": round(100.0 * abs_sums[1] / denom, 2) if is_act else np.nan,
                    "share_impl_pct": round(100.0 * abs_sums[2] / denom, 2) if is_act else np.nan,
                    "share_tax_pct": round(100.0 * abs_sums[3] / denom, 2) if is_act else np.nan,
                    "rcap_births_per_snap": round(gross_b_all / n_steps, 2),
                    "rcap_deaths_per_snap": round(gross_d_all / n_steps, 2),
                    "rcap_weighted_births_per_snap": round(gross_b_w / n_steps, 2),
                    "rcap_weighted_deaths_per_snap": round(gross_d_w / n_steps, 2),
                }
            )

        # 2D. Precompute Calibration R_cap Series (Static Catalogs, both Counts and Densities!) & Track A Fingerprints
        kslocs = np.maximum(grp["py_sloc"].to_numpy(dtype=float) / 1000.0, 0.001)
        for label_p, r0_lbl, r1_lbl, pmode in calib_pairs:
            if pmode == "sq":
                p0, m0 = phys_sq_lookup[r0_lbl]
                p1, m1 = phys_sq_lookup[r1_lbl]
                if not all(s in sq_repo.get(p0, {}) and s in sq_repo.get(p1, {}) for s in snaps):
                    continue
                rmap0, rmap1 = sq_rules.get(p0, {}), sq_rules.get(p1, {})
                rcap = set(rmap0.keys()) & set(rmap1.keys())
                c0_rc = np.array(
                    [
                        sum(
                            1
                            for x in sq_repo[p0][s]["issues"]
                            if x["rule"] in rcap
                            and (
                                sq_mqr_is_high(x.get("impacts") or rmap0.get(x["rule"], {}).get("impacts"))
                                if m0
                                else sq_std_is_high(x.get("severity") or rmap0.get(x["rule"], {}).get("severity"))
                            )
                        )
                        for s in snaps
                    ],
                    dtype=float,
                )
                c1_rc = np.array(
                    [
                        sum(
                            1
                            for x in sq_repo[p1][s]["issues"]
                            if x["rule"] in rcap
                            and (
                                sq_mqr_is_high(x.get("impacts") or rmap1.get(x["rule"], {}).get("impacts"))
                                if m1
                                else sq_std_is_high(x.get("severity") or rmap1.get(x["rule"], {}).get("severity"))
                            )
                        )
                        for s in snaps
                    ],
                    dtype=float,
                )
                rcap_series_map[(r0_lbl, r1_lbl)][repo] = (c0_rc / kslocs, c1_rc / kslocs, c0_rc, c1_rc)

                fp0_list, fp1_list = [], []
                for sid in snaps:
                    cnt0: Counter[Tuple[str, str, str]] = Counter()
                    for x in sq_repo[p0][sid]["issues"]:
                        rk = x["rule"]
                        is_h = (
                            sq_mqr_is_high(x.get("impacts") or rmap0.get(rk, {}).get("impacts"))
                            if m0
                            else sq_std_is_high(x.get("severity") or rmap0.get(rk, {}).get("severity"))
                        )
                        if is_h:
                            cnt0[(sys.intern(rk), sys.intern(x["file"]), str(x.get("hash") or x.get("line") or ""))] += 1
                    fp0_list.append(cnt0)

                    cnt1: Counter[Tuple[str, str, str]] = Counter()
                    for x in sq_repo[p1][sid]["issues"]:
                        rk = x["rule"]
                        is_h = (
                            sq_mqr_is_high(x.get("impacts") or rmap1.get(rk, {}).get("impacts"))
                            if m1
                            else sq_std_is_high(x.get("severity") or rmap1.get(rk, {}).get("severity"))
                        )
                        if is_h:
                            cnt1[(sys.intern(rk), sys.intern(x["file"]), str(x.get("hash") or x.get("line") or ""))] += 1
                    fp1_list.append(cnt1)
                high_fp_pair_map[(label_p, repo, "r0")] = fp0_list
                high_fp_pair_map[(label_p, repo, "r1")] = fp1_list
            else:
                if not all(s in linter_repo.get(r0_lbl, {}) and s in linter_repo.get(r1_lbl, {}) for s in snaps):
                    continue
                _, _, rcap, _, _, _, _ = get_linter_pair_sets(r0_lbl, r1_lbl, pmode)
                c0_rc = np.array(
                    [
                        sum(
                            1
                            for x in linter_repo[r0_lbl][s]
                            if canonical_linter_rule(pmode, r0_lbl, x["rule"]) in rcap and x["severity"] == "HIGH"
                        )
                        for s in snaps
                    ],
                    dtype=float,
                )
                c1_rc = np.array(
                    [
                        sum(
                            1
                            for x in linter_repo[r1_lbl][s]
                            if canonical_linter_rule(pmode, r1_lbl, x["rule"]) in rcap and x["severity"] == "HIGH"
                        )
                        for s in snaps
                    ],
                    dtype=float,
                )
                rcap_series_map[(r0_lbl, r1_lbl)][repo] = (c0_rc / kslocs, c1_rc / kslocs, c0_rc, c1_rc)

                fp0_list, fp1_list = [], []
                for sid in snaps:
                    cnt0_l: Counter[Tuple[str, str, str]] = Counter()
                    for x in linter_repo[r0_lbl][sid]:
                        if x["severity"] == "HIGH":
                            cnt0_l[
                                (
                                    sys.intern(canonical_linter_rule(pmode, r0_lbl, x["rule"])),
                                    sys.intern(x["file"]),
                                    str(x.get("lhash") or x.get("line") or ""),
                                )
                            ] += 1
                    fp0_list.append(cnt0_l)

                    cnt1_l: Counter[Tuple[str, str, str]] = Counter()
                    for x in linter_repo[r1_lbl][sid]:
                        if x["severity"] == "HIGH":
                            cnt1_l[
                                (
                                    sys.intern(canonical_linter_rule(pmode, r1_lbl, x["rule"])),
                                    sys.intern(x["file"]),
                                    str(x.get("lhash") or x.get("line") or ""),
                                )
                            ] += 1
                    fp1_list.append(cnt1_l)
                high_fp_pair_map[(label_p, repo, "r0")] = fp0_list
                high_fp_pair_map[(label_p, repo, "r1")] = fp1_list

        # 2E. Collect Candidate Qualitative Slices (with Explicit Verification Basis & Verbatim Git Source)
        meta_by_sid = grp.set_index("snapshot_id").to_dict("index")

        # Stratum 1: MQR Taxonomy Promotions (100% Catalog-Metadata-Verified)
        s1_repo_rules: Set[str] = set()
        for phys_reg in ("sq26_comp_mqr", "sq26_mqr"):
            rmap = sq_rules.get(phys_reg, {})
            for sid in sorted(sq_repo.get(phys_reg, {}).keys()):
                if len(s1_repo_rules) >= 3:
                    break
                sm = meta_by_sid[sid]
                for x in sq_repo[phys_reg][sid]["issues"]:
                    rk = x["rule"]
                    if rk in s1_repo_rules or not x.get("is_new"):
                        continue
                    rinfo = rmap.get(rk, {})
                    itype = x.get("type") or rinfo.get("type") or ""
                    sev = x.get("severity") or rinfo.get("severity") or ""
                    imps = x.get("impacts") or rinfo.get("impacts") or []
                    promoted_rel_sec = (itype == "CODE_SMELL") and sq_mqr_is_rel_sec(imps)
                    promoted_high = (not sq_std_is_high(sev)) and sq_mqr_is_high(imps)
                    if not (promoted_rel_sec or promoted_high):
                        continue
                    snip, verb = fetch_git_snippet_and_verbatim(repo_dir, sm["commit"], x["file"], int(x.get("line") or 0))
                    if not snip:
                        continue
                    s1_repo_rules.add(rk)
                    imp_str = "+".join(f"{i.get('softwareQuality')}:{i.get('severity')}" for i in imps)
                    if promoted_rel_sec and promoted_high:
                        r0_v = f"{itype}/{sev} (Rel/Sec & High Gates: PASS)"
                        r1_v = f"{imp_str} (Rel/Sec & High Gates: FAIL)"
                        rc_lbl = "Multi-impact MQR remapping + upward HIGH severity promotion"
                    elif promoted_rel_sec:
                        r0_v = f"{itype}/{sev} (Rel/Sec Attr Gate: PASS)"
                        r1_v = f"{imp_str} (Rel/Sec Attr Gate: FAIL)"
                        rc_lbl = "Multi-impact MQR remapping: legacy CODE_SMELL assigned RELIABILITY/SECURITY impact"
                    else:
                        r0_v = f"{itype}/{sev} (Zero New High Gate: PASS)"
                        r1_v = f"{imp_str} (Zero New High Gate: FAIL)"
                        rc_lbl = "MQR severity upward recalibration: legacy MAJOR/MINOR rule promoted to HIGH/BLOCKER"
                    cand_s1.append(
                        {
                            "slice_category": "1. E_tax_remap (Standard -> MQR Gate Flip)",
                            "regime_pair": f"{phys_reg}: Standard -> MQR",
                            "repository": repo,
                            "release_tag": sm["release_tag"],
                            "commit": sm["commit"][:10],
                            "full_commit": sm["commit"],
                            "file_line": f"{x['file']}:{x.get('line')}",
                            "canonical_rule": rk,
                            "rule_id": rk,
                            "rule_name": rinfo.get("name", ""),
                            "regime0_verdict": r0_v,
                            "regime1_verdict": r1_v,
                            "root_cause": rc_lbl,
                            "verification_basis": "Catalog-Metadata-Verified (exact Standard type/severity vs MQR impacts on identical JSON issue)",
                            "code_snippet": snip,
                            "verbatim_code": verb,
                            "_rk": rk,
                        }
                    )
                    if len(s1_repo_rules) >= 3:
                        break

        # Stratum 2: SQ 10.3 -> SQ 26.11 R_cap (Explicitly Separating Hotspot-to-Issue Reclassification from Non-Hotspot Drift!)
        r103 = sq_rules.get("sq103_n09", {})
        r26c = sq_rules.get("sq26_comp_mqr", {})
        rcap_sq = set(r103.keys()) & set(r26c.keys())
        hs_sq_set = {k for k in rcap_sq if r103[k].get("type") == "SECURITY_HOTSPOT" and r26c[k].get("type") != "SECURITY_HOTSPOT"}

        s2_hs_rules: Set[str] = set()
        s2_nonhs_rules: Set[str] = set()
        s2_d_rules: Set[str] = set()
        for sid in sorted(sq_repo.get("sq103_n09", {}).keys()):
            if sid not in sq_repo.get("sq26_comp_mqr", {}):
                continue
            if len(s2_hs_rules) >= 1 and len(s2_nonhs_rules) >= 2 and len(s2_d_rules) >= 2:
                break
            sm = meta_by_sid[sid]
            m0 = {(x["rule"], x["file"], int(x.get("line") or 0)): x for x in sq_repo["sq103_n09"][sid]["issues"] if x["rule"] in rcap_sq}
            m1 = {(x["rule"], x["file"], int(x.get("line") or 0)): x for x in sq_repo["sq26_comp_mqr"][sid]["issues"] if x["rule"] in rcap_sq}
            for k in sorted(set(m1) - set(m0)):
                rk0 = k[0]
                if rk0 in hs_sq_set and len(s2_hs_rules) < 1 and rk0 not in s2_hs_rules:
                    snip, verb = fetch_git_snippet_and_verbatim(repo_dir, sm["commit"], k[1], k[2])
                    if snip:
                        s2_hs_rules.add(rk0)
                        cand_s2_hs.append(
                            {
                                "slice_category": "2. E_impl_drift (SonarQube Shared-Rule Drift)",
                                "regime_pair": "SQ 10.3 -> SQ 26.11 (Shared Rule R_cap)",
                                "repository": repo,
                                "release_tag": sm["release_tag"],
                                "commit": sm["commit"][:10],
                                "full_commit": sm["commit"],
                                "file_line": f"{k[1]}:{k[2]}",
                                "canonical_rule": rk0,
                                "rule_id": rk0,
                                "rule_name": r103.get(rk0, {}).get("name", ""),
                                "regime0_verdict": "ABSENT IN /api/issues (SQ 10.3 type=SECURITY_HOTSPOT)",
                                "regime1_verdict": f"IN /api/issues ({r26c.get(rk0, {}).get('type')}/{m1[k].get('severity')})",
                                "root_cause": "Hotspot-to-Issue API & type migration: legacy SECURITY_HOTSPOT reclassified to VULNERABILITY/CODE_SMELL",
                                "verification_basis": "Catalog-Metadata-Verified (SQ 10.3 type=SECURITY_HOTSPOT -> SQ 26.11 type=VULNERABILITY/CODE_SMELL)",
                                "code_snippet": snip,
                                "verbatim_code": verb,
                                "_rk": rk0,
                            }
                        )
                elif rk0 not in hs_sq_set and len(s2_nonhs_rules) < 2 and rk0 not in s2_nonhs_rules:
                    snip, verb = fetch_git_snippet_and_verbatim(repo_dir, sm["commit"], k[1], k[2])
                    if snip:
                        s2_nonhs_rules.add(rk0)
                        cand_s2_births_nonhs.append(
                            {
                                "slice_category": "2. E_impl_drift (SonarQube Shared-Rule Drift)",
                                "regime_pair": "SQ 10.3 -> SQ 26.11 (Shared Rule R_cap)",
                                "repository": repo,
                                "release_tag": sm["release_tag"],
                                "commit": sm["commit"][:10],
                                "full_commit": sm["commit"],
                                "file_line": f"{k[1]}:{k[2]}",
                                "canonical_rule": rk0,
                                "rule_id": rk0,
                                "rule_name": r103.get(rk0, {}).get("name", ""),
                                "regime0_verdict": f"NOT FLAGGED in /api/issues (SQ 10.3 type={r103.get(rk0, {}).get('type')})",
                                "regime1_verdict": f"NEWLY FLAGGED ({m1[k].get('severity')})",
                                "root_cause": "Non-hotspot issue-detector drift: upgraded Python 3.12+ AST / cross-module symbolic resolution on R_cap",
                                "verification_basis": "Source-&-Catalog-Verified (non-hotspot rule active in both catalogs; candidate engine mechanism)",
                                "code_snippet": snip,
                                "verbatim_code": verb,
                                "_rk": rk0,
                            }
                        )
            if len(s2_d_rules) < 2:
                for k in sorted(set(m0) - set(m1)):
                    if k[0] in s2_d_rules:
                        continue
                    snip, verb = fetch_git_snippet_and_verbatim(repo_dir, sm["commit"], k[1], k[2])
                    if not snip:
                        continue
                    s2_d_rules.add(k[0])
                    is_test_path = "test" in k[1].lower()
                    rc_sq_drop = (
                        "Test-file scope exclusion silencing legacy alert inside test module (-D_cap)"
                        if is_test_path
                        else "Refined analyzer reachability / false-positive pruning silencing legacy alert (-D_cap)"
                    )
                    cand_s2_drops.append(
                        {
                            "slice_category": "2. E_impl_drift (SonarQube Shared-Rule Drift)",
                            "regime_pair": "SQ 10.3 -> SQ 26.11 (Shared Rule R_cap)",
                            "repository": repo,
                            "release_tag": sm["release_tag"],
                            "commit": sm["commit"][:10],
                            "full_commit": sm["commit"],
                            "file_line": f"{k[1]}:{k[2]}",
                            "canonical_rule": k[0],
                            "rule_id": k[0],
                            "rule_name": r103.get(k[0], {}).get("name", ""),
                            "regime0_verdict": f"FLAGGED ({m0[k].get('severity')})",
                            "regime1_verdict": "SILENCED (0 issues in SQ 26.11)",
                            "root_cause": rc_sq_drop,
                            "verification_basis": "Path-&-Catalog-Verified (test-module path pattern)" if is_test_path else "Source-&-Catalog-Verified (candidate analyzer pruning mechanism)",
                            "code_snippet": snip,
                            "verbatim_code": verb,
                            "_rk": k[0],
                        }
                    )
                    if len(s2_d_rules) >= 2:
                        break

        # Strata 3 & 4: Flake8 -> Ruff (direct) & Pylint -> Ruff PL (pylint_ruff, NO W0613<->B007!)
        for strat_idx, pair_lbl, t0, t1, pmode, target_pool in [
            ("3", "Flake8 7.1 -> Ruff 0.17.5", "flake8_71", "ruff_modern_extended", "direct", cand_s3),
            ("4", "Pylint 3.3 -> Ruff 0.17.5 (PL)", "pylint_33", "ruff_modern_pylint", "pylint_ruff", cand_s4),
        ]:
            _, _, rcap_l, _, _, _, _ = get_linter_pair_sets(t0, t1, pmode)
            repo_rdirs: Set[Tuple[str, str]] = set()
            for sid in sorted(linter_repo.get(t0, {}).keys()):
                if sid not in linter_repo.get(t1, {}) or len(repo_rdirs) >= 3:
                    break
                sm = meta_by_sid[sid]
                a = {
                    (canonical_linter_rule(pmode, t0, x["rule"]), x["file"], int(x["line"])): x
                    for x in linter_repo[t0][sid]
                    if canonical_linter_rule(pmode, t0, x["rule"]) in rcap_l
                }
                b = {
                    (canonical_linter_rule(pmode, t1, x["rule"]), x["file"], int(x["line"])): x
                    for x in linter_repo[t1][sid]
                    if canonical_linter_rule(pmode, t1, x["rule"]) in rcap_l
                }
                for k in sorted(set(a) ^ set(b)):
                    if not k[1].endswith(".py"):
                        continue
                    direction = "T0_ONLY" if k in a else "T1_ONLY"
                    if (k[0], direction) in repo_rdirs:
                        continue
                    snip, verb = fetch_git_snippet_and_verbatim(repo_dir, sm["commit"], k[1], k[2])
                    if not snip:
                        continue
                    repo_rdirs.add((k[0], direction))
                    iss_obj = a[k] if k in a else b[k]
                    raw_c = iss_obj["rule"]
                    has_pragma = ("noqa" in verb.lower()) or ("pylint:" in verb.lower())
                    if strat_idx == "3":
                        if raw_c.startswith("F"):
                            rc_l = "Scope binding / __all__ re-export / import resolution divergence (Pyflakes vs Ruff)"
                        elif raw_c.startswith(("B", "C90")):
                            rc_l = "Bugbear / McCabe AST visitor & inline # noqa pragma dialect divergence (Flake8 vs Ruff)"
                        else:
                            rc_l = "PEP 8 token, multibyte Unicode width & docstring indentation span difference (pycodestyle vs Ruff)"
                    else:
                        if direction == "T0_ONLY":
                            rc_l = "Pylint-only alert: Astroid cross-module/type-inference check or narrower Ruff AST scope"
                        else:
                            rc_l = "Ruff-only alert: single-file syntactic AST match without Astroid context or # pylint: disable suppression"
                    if strat_idx == "3":
                        v_basis = (
                            "Source-Pragma-Verified (inline # noqa comment in checked-out source line)"
                            if has_pragma
                            else "Source-Construct-Verified (checked-out source token/AST span; candidate pyflakes/pycodestyle/bugbear vs Ruff checker mechanism)"
                        )
                    else:
                        v_basis = (
                            "Source-Pragma-Verified (inline # noqa / # pylint: disable comment in checked-out source line)"
                            if has_pragma
                            else "Source-Construct-Verified (checked-out source construct; candidate Astroid cross-module inference vs single-file Ruff AST mechanism)"
                        )
                    target_pool.append(
                        {
                            "slice_category": f"{strat_idx}. E_impl_drift ({pair_lbl} Mapped Rule)",
                            "regime_pair": pair_lbl,
                            "repository": repo,
                            "release_tag": sm["release_tag"],
                            "commit": sm["commit"][:10],
                            "full_commit": sm["commit"],
                            "file_line": f"{k[1]}:{k[2]}",
                            "canonical_rule": k[0],
                            "rule_id": f"{k[0]} ({raw_c})" if k[0] != raw_c else raw_c,
                            "rule_name": iss_obj.get("msg", "")[:90],
                            "regime0_verdict": f"FLAGGED ({a[k]['rule']}, {a[k]['severity']})" if k in a else "NOT FLAGGED",
                            "regime1_verdict": f"FLAGGED ({b[k]['rule']}, {b[k]['severity']})" if k in b else "NOT FLAGGED",
                            "root_cause": rc_l,
                            "verification_basis": v_basis,
                            "code_snippet": snip,
                            "verbatim_code": verb,
                            "_rk": f"{k[0]}:{direction}",
                        }
                    )
                    if len(repo_rdirs) >= 3:
                        break

        del sq_repo, linter_repo
        gc.collect()
        if repo_idx % 10 == 0 or repo_idx == len(all_repos_sorted):
            print(f"  [Pass 2 repo stream] {repo_idx}/{len(all_repos_sorted)} repos processed (latest={repo})", flush=True)

    # Save Corpus Telemetry Summary (counted directly from verified loaded scan files)
    telemetry = {
        "n_repositories": int(manifest_df["repository"].nunique()),
        "n_snapshots": int(len(manifest_df)),
        "n_transitions": int(sum(manifest_df["seq_idx"] > 1)),
        "n_verified_sq_scans_loaded": int(verified_sq_scans_loaded),
        "n_verified_linter_scans_loaded": int(verified_linter_scans_loaded),
        "n_physical_scans": int(verified_sq_scans_loaded + verified_linter_scans_loaded),
        "n_regime_snapshots": int(len(snap_rows)),
        "sq_alert_counts": dict(sq_alert_counts),
        "sq_observed_rule_counts": {k: len(v) for k, v in observed_sq_rules.items()},
        "linter_alert_counts": linter_alert_counts,
        "linter_observed_rule_counts": {k: len(v) for k, v in observed_raw_codes.items()},
        "linter_non_catalog_parser_diagnostics": non_catalog_parser_diagnostics,
        "total_sq_alerts": int(sum(sq_alert_counts.values())),
        "total_linter_alerts": int(sum(linter_alert_counts.values())),
        "total_raw_alerts": int(sum(sq_alert_counts.values()) + sum(linter_alert_counts.values())),
    }
    (out_dir / "corpus_telemetry_49repos.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")

    # Finalize Unified Snapshot Metrics Table
    snap_df = pd.DataFrame(snap_rows)
    snap_df = snap_df.sort_values(["regime", "repository", "seq_idx"]).reset_index(drop=True)
    snap_df["prev_high_issues"] = snap_df.groupby(["regime", "repository"])["high_issues"].shift(1)
    snap_df["prev_rel_sec_issues"] = snap_df.groupby(["regime", "repository"])["rel_sec_issues"].shift(1)
    snap_df["ratchet_high_gate_pass"] = np.where(
        snap_df["seq_idx"] == 1, None, snap_df["high_issues"] <= snap_df["prev_high_issues"]
    )
    snap_df["ratchet_rel_sec_gate_pass"] = np.where(
        snap_df["seq_idx"] == 1, None, snap_df["rel_sec_issues"] <= snap_df["prev_rel_sec_issues"]
    )
    snap_df["overall_high_gate_pass"] = snap_df["density_high_ksloc"] <= 1.0
    snap_df["overall_zero_rel_sec_pass"] = snap_df["rel_sec_issues"] == 0
    snap_df.to_csv(out_dir / "full_unified_metrics.csv", index=False)

    # =========================================================================
    # TABLE 1: Multi-Gate Flip Rates & Cross-Project Ranking
    # =========================================================================
    pair_specs = [
        ("SQ 26.11 Core: Standard -> MQR (Pure Taxonomy, 232 rules)", "sq26_core_std", "sq26_core_mqr"),
        ("SQ 26.11 Comprehensive: Standard -> MQR (Pure Taxonomy, 407 rules)", "sq26_comp_std", "sq26_comp_mqr"),
        ("SQ 26.11 MQR: Comprehensive (407) -> Default Core (232) (Pure Profile Split)", "sq26_comp_mqr", "sq26_core_mqr"),
        ("SQ 10.3 [N09] Std -> CleanCode Impact (Same Version Dual View)", "sq103_std", "sq103_cleancode"),
        ("SQ 10.3 [N09] -> SQ 26.11 Core MQR (Default Post-N09 Upgrade)", "sq103_std", "sq26_core_mqr"),
        ("SQ 10.3 [N09] -> SQ 26.11 Comprehensive MQR (Rule-Preserving Upgrade)", "sq103_std", "sq26_comp_mqr"),
        ("SQ 9.9 LTA Std -> SQ 26.11 Core Std (Pure LTA Engine+Profile Upgrade)", "sq99_std", "sq26_core_std"),
        ("SQ 9.9 LTA Std -> SQ 26.11 Core MQR (LTA -> Modern Default MQR)", "sq99_std", "sq26_core_mqr"),
        ("SQ 9.9 LTA Std -> SQ 26.11 Comprehensive MQR (LTA -> Comprehensive MQR)", "sq99_std", "sq26_comp_mqr"),
        ("Flake8 7.1 -> Ruff 0.17.5 (Default Profile)", "flake8_71", "ruff_modern_default"),
        ("Flake8 7.1 -> Ruff 0.17.5 (Extended Profile)", "flake8_71", "ruff_modern_extended"),
        ("Pylint 3.3 -> Ruff 0.17.5 (Pylint-PL Aligned Profile)", "pylint_33", "ruff_modern_pylint"),
        ("Pylint 3.3 -> Ruff 0.17.5 (Extended Profile)", "pylint_33", "ruff_modern_extended"),
        ("Pylint 2.17 -> Pylint 3.3 (Cross-Major Upgrade)", "pylint_217", "pylint_33"),
        ("Ruff 0.1.15 -> Ruff 0.17.5 (Extended Cross-Version)", "ruff_01_extended", "ruff_modern_extended"),
    ]

    avail_regs = set(snap_df["regime"].unique())
    t1_rows = []
    for desc, r0, r1 in pair_specs:
        if r0 not in avail_regs or r1 not in avail_regs:
            continue
        d0 = snap_df[snap_df["regime"] == r0].set_index("snapshot_id")
        d1 = snap_df[snap_df["regime"] == r1].set_index("snapshot_id")
        common_all = [s for s in d0.index if s in d1.index]
        trans = [s for s in common_all if d0.loc[s, "seq_idx"] > 1]

        def calc_flip(col: str, s_list: List[str]) -> Tuple[int, float, str, int, int]:
            arr = np.array([1 if bool(d0.loc[s, col]) != bool(d1.loc[s, col]) else 0 for s in s_list], dtype=float)
            p2f = sum(1 for s in s_list if bool(d0.loc[s, col]) and not bool(d1.loc[s, col]))
            f2p = sum(1 for s in s_list if not bool(d0.loc[s, col]) and bool(d1.loc[s, col]))
            flips = int(arr.sum())
            pct = round(100.0 * flips / max(len(s_list), 1), 2)
            lo, hi = bootstrap_ci_pct(arr, rng)
            return flips, pct, f"[{lo:.1f}, {hi:.1f}]", p2f, f2p

        nat_f, nat_pct, nat_ci, nat_p2f, nat_f2p = calc_flip("native_qg_pass", trans)
        cayc_f, cayc_pct, _, _, _ = calc_flip("cayc_zero_new_pass", trans)
        nh_f, nh_pct, nh_ci, nh_p2f, nh_f2p = calc_flip("new_high_gate_pass", trans)
        nrs_f, nrs_pct, nrs_ci, nrs_p2f, nrs_f2p = calc_flip("new_rel_sec_gate_pass", trans)
        rat_f, rat_pct, _, _, _ = calc_flip("ratchet_high_gate_pass", trans)
        rat_rs_f, rat_rs_pct, _, _, _ = calc_flip("ratchet_rel_sec_gate_pass", trans)
        ov_rs_f, ov_rs_pct, _, _, _ = calc_flip("overall_zero_rel_sec_pass", common_all)

        taus_h, taus_rs, taus_tot = [], [], []
        inv_h, tot_pairs = 0, 0
        slice_selectors = [
            (d0[d0["seq_idx"] == sidx], d1[d1["seq_idx"] == sidx]) for sidx in range(1, 11)
        ] + [(d0[d0["seq_idx"] == d0["series_len"]], d1[d1["seq_idx"] == d1["series_len"]])]
        for s0_raw, s1_raw in slice_selectors:
            s0 = s0_raw.sort_values("repository")
            s1 = s1_raw.sort_values("repository")
            if len(s0) >= 10 and len(s0) == len(s1) and list(s0["repository"]) == list(s1["repository"]):
                vh0, vh1 = s0["density_high_ksloc"].to_numpy(), s1["density_high_ksloc"].to_numpy()
                vrs0, vrs1 = s0["density_rel_sec_ksloc"].to_numpy(), s1["density_rel_sec_ksloc"].to_numpy()
                vt0, vt1 = s0["density_total_ksloc"].to_numpy(), s1["density_total_ksloc"].to_numpy()
                th, _ = kendalltau(vh0, vh1)
                trs, _ = kendalltau(vrs0, vrs1)
                tt, _ = kendalltau(vt0, vt1)
                if not np.isnan(th):
                    taus_h.append(float(th))
                if not np.isnan(trs):
                    taus_rs.append(float(trs))
                if not np.isnan(tt):
                    taus_tot.append(float(tt))
                m = len(vh0)
                for i in range(m):
                    for j in range(i + 1, m):
                        tot_pairs += 1
                        if (vh0[i] - vh0[j]) * (vh1[i] - vh1[j]) < 0:
                            inv_h += 1

        t1_rows.append(
            {
                "comparison": desc,
                "r0": r0,
                "r1": r1,
                "n_transitions": len(trans),
                "native_qg_flip_pct": nat_pct,
                "native_qg_ci95": nat_ci,
                "native_p2f_f2p": f"{nat_p2f}/{nat_f2p}",
                "cayc_allnew_flip_pct": cayc_pct,
                "new_high_flip_pct": nh_pct,
                "new_high_ci95": nh_ci,
                "new_high_p2f_f2p": f"{nh_p2f}/{nh_f2p}",
                "new_rel_sec_attr_flip_pct": nrs_pct,
                "new_rel_sec_ci95": nrs_ci,
                "new_rel_sec_p2f_f2p": f"{nrs_p2f}/{nrs_f2p}",
                "ratchet_high_flip_pct": rat_pct,
                "ratchet_rel_sec_flip_pct": rat_rs_pct,
                "overall_zero_rel_sec_flip_pct": ov_rs_pct,
                "kendall_tau_high_density": round(float(np.mean(taus_h)) if taus_h else np.nan, 3),
                "kendall_tau_rel_sec_density": round(float(np.mean(taus_rs)) if taus_rs else np.nan, 3),
                "kendall_tau_total_density": round(float(np.mean(taus_tot)) if taus_tot else np.nan, 3),
                "pairwise_rank_inversion_high_pct": round(100.0 * inv_h / max(tot_pairs, 1), 2),
            }
        )

    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(out_dir / "table1_multigate_flips_and_ranking.csv", index=False)

    # Sensitivity Check 1B: Non-decreasing numeric version-tuple release transitions (n = 781)
    def _parse_ver(tag_s: str) -> Tuple[int, ...]:
        nums = [int(x) for x in re.findall(r"\d+", str(tag_s))[:4]]
        return tuple(nums) if nums else (0,)

    mono_snap_ids: Set[str] = set()
    for _, grp_m in manifest_df.groupby("repository"):
        grp_m = grp_m.sort_values("seq_idx")
        sids = grp_m["snapshot_id"].tolist()
        tags = grp_m["release_tag"].tolist()
        for i in range(1, len(sids)):
            if _parse_ver(tags[i]) >= _parse_ver(tags[i - 1]):
                mono_snap_ids.add(sids[i])

    t1_mono_rows = []
    for desc, r0, r1 in pair_specs:
        if r0 not in avail_regs or r1 not in avail_regs:
            continue
        d0 = snap_df[snap_df["regime"] == r0].set_index("snapshot_id")
        d1 = snap_df[snap_df["regime"] == r1].set_index("snapshot_id")
        m_trans = [s for s in d0.index if s in d1.index and s in mono_snap_ids]
        if not m_trans:
            continue
        t1_mono_rows.append(
            {
                "comparison": desc,
                "n_monotone_transitions": len(m_trans),
                "native_qg_flip_pct": round(100.0 * sum(1 for s in m_trans if bool(d0.loc[s, "native_qg_pass"]) != bool(d1.loc[s, "native_qg_pass"])) / len(m_trans), 2),
                "cayc_allnew_flip_pct": round(100.0 * sum(1 for s in m_trans if bool(d0.loc[s, "cayc_zero_new_pass"]) != bool(d1.loc[s, "cayc_zero_new_pass"])) / len(m_trans), 2),
                "new_high_flip_pct": round(100.0 * sum(1 for s in m_trans if bool(d0.loc[s, "new_high_gate_pass"]) != bool(d1.loc[s, "new_high_gate_pass"])) / len(m_trans), 2),
                "new_rel_sec_attr_flip_pct": round(100.0 * sum(1 for s in m_trans if bool(d0.loc[s, "new_rel_sec_gate_pass"]) != bool(d1.loc[s, "new_rel_sec_gate_pass"])) / len(m_trans), 2),
                "ratchet_high_flip_pct": round(100.0 * sum(1 for s in m_trans if bool(d0.loc[s, "ratchet_high_gate_pass"]) != bool(d1.loc[s, "ratchet_high_gate_pass"])) / len(m_trans), 2),
            }
        )
    pd.DataFrame(t1_mono_rows).to_csv(out_dir / "table1_semver_monotone_sensitivity.csv", index=False)

    # Sensitivity Check 1C: Scale-Stratified Gate Flip Rates across Repository Size Tiers
    tier_order = ["Small (<25k SLOC)", "Medium (25k-150k SLOC)", "Large/Monorepo (>150k SLOC)"]
    t1c_rows = []
    for desc, r0, r1 in pair_specs:
        if r0 not in avail_regs or r1 not in avail_regs:
            continue
        d0 = snap_df[snap_df["regime"] == r0].set_index("snapshot_id")
        d1 = snap_df[snap_df["regime"] == r1].set_index("snapshot_id")
        trans_all = [s for s in d0.index if s in d1.index and d0.loc[s, "seq_idx"] > 1]
        for tier_lbl in tier_order:
            t_snaps = [s for s in trans_all if repo_to_tier.get(str(d0.loc[s, "repository"])) == tier_lbl]
            if not t_snaps:
                continue
            n_t = len(t_snaps)
            n_rp = len({str(d0.loc[s, "repository"]) for s in t_snaps})
            t1c_rows.append(
                {
                    "comparison": desc,
                    "scale_tier": tier_lbl,
                    "n_repos": n_rp,
                    "n_transitions": n_t,
                    "native_qg_flip_pct": round(
                        100.0 * sum(1 for s in t_snaps if bool(d0.loc[s, "native_qg_pass"]) != bool(d1.loc[s, "native_qg_pass"])) / n_t, 2
                    ),
                    "cayc_allnew_flip_pct": round(
                        100.0 * sum(1 for s in t_snaps if bool(d0.loc[s, "cayc_zero_new_pass"]) != bool(d1.loc[s, "cayc_zero_new_pass"])) / n_t, 2
                    ),
                    "new_high_flip_pct": round(
                        100.0 * sum(1 for s in t_snaps if bool(d0.loc[s, "new_high_gate_pass"]) != bool(d1.loc[s, "new_high_gate_pass"])) / n_t, 2
                    ),
                    "new_rel_sec_attr_flip_pct": round(
                        100.0 * sum(1 for s in t_snaps if bool(d0.loc[s, "new_rel_sec_gate_pass"]) != bool(d1.loc[s, "new_rel_sec_gate_pass"])) / n_t, 2
                    ),
                    "ratchet_high_flip_pct": round(
                        100.0 * sum(1 for s in t_snaps if bool(d0.loc[s, "ratchet_high_gate_pass"]) != bool(d1.loc[s, "ratchet_high_gate_pass"])) / n_t, 2
                    ),
                }
            )
    pd.DataFrame(t1c_rows).to_csv(out_dir / "table1c_scale_stratified_flips.csv", index=False)

    # =========================================================================
    # TABLE 2: Longitudinal Evolution Trend Reversals (Raw p, BH-FDR q, Hamed-Rao p_HR, Unrounded Sen Slope)
    # =========================================================================
    trend_rows = []
    for (reg, repo), grp in snap_df.groupby(["regime", "repository"]):
        grp = grp.sort_values("seq_idx")
        in_rq3 = bool(grp["in_n09_rq3_sig"].iloc[0])
        x = grp["seq_idx"].to_numpy(dtype=float)
        for mcol in ["density_total_ksloc", "density_high_ksloc", "density_rel_sec_ksloc", "total_issues", "high_issues"]:
            y = grp[mcol].to_numpy(dtype=float)
            if np.all(y == y[0]):
                tau_mk, p_mk, p_hr, sl_raw = 0.0, 1.0, 1.0, 0.0
            else:
                tau_mk, p_mk = kendalltau(x, y)
                p_hr = hamed_rao_mk_p(y)
                sl_raw = float(theilslopes(y, x)[0])
            tclass = (
                "SIG_INC"
                if (p_mk < 0.05 and tau_mk > 0)
                else ("SIG_DEC" if (p_mk < 0.05 and tau_mk < 0) else ("POS_NS" if tau_mk > 0 else ("NEG_NS" if tau_mk < 0 else "FLAT")))
            )
            trend_rows.append(
                {
                    "repository": repo,
                    "in_n09_rq3_sig": in_rq3,
                    "regime": reg,
                    "metric": mcol,
                    "stat": round(float(tau_mk), 4),
                    "stat_raw": float(tau_mk),
                    "p_val": float(p_mk),
                    "p_hamed_rao": round(float(p_hr), 5),
                    "sen_slope": round(sl_raw, 5),
                    "sen_slope_raw": sl_raw,
                    "first_val": round(float(y[0]), 3),
                    "last_val": round(float(y[-1]), 3),
                    "class": tclass,
                }
            )

        g_tr = grp[grp["seq_idx"] >= 2]
        for target_col, label_m in [("total_issues", "rq3_lines_vs_total"), ("high_issues", "rq3_lines_vs_high")]:
            yv = g_tr[target_col].to_numpy(dtype=float)
            xv = g_tr["all_lines_changed"].to_numpy(dtype=float)
            if np.all(yv == yv[0]) or np.all(xv == xv[0]):
                rho, pv = 0.0, 1.0
            else:
                rho, pv = spearmanr(xv, yv)
            c_str = "SIG_POS" if (pv < 0.05 and rho > 0) else ("SIG_NEG" if (pv < 0.05 and rho < 0) else "NS")
            trend_rows.append(
                {
                    "repository": repo,
                    "in_n09_rq3_sig": in_rq3,
                    "regime": reg,
                    "metric": label_m,
                    "stat": round(float(rho), 4),
                    "stat_raw": float(rho),
                    "p_val": float(pv),
                    "p_hamed_rao": float(pv),
                    "sen_slope": 0.0,
                    "sen_slope_raw": float(rho),
                    "first_val": 0.0,
                    "last_val": 0.0,
                    "class": c_str,
                }
            )

    trend_df = pd.DataFrame(trend_rows)
    trend_df["q_fdr"] = 1.0
    for (reg, met), idx_grp in trend_df.groupby(["regime", "metric"]).groups.items():
        pvals = trend_df.loc[idx_grp, "p_val"].tolist()
        trend_df.loc[idx_grp, "q_fdr"] = np.round(bh_fdr(pvals), 5)

    # Save clean columns to table2_per_repo_trends.csv
    trend_df.drop(columns=["stat_raw", "sen_slope_raw"]).to_csv(out_dir / "table2_per_repo_trends.csv", index=False)

    rev_summary_rows = []
    for desc, r0, r1 in pair_specs:
        if r0 not in avail_regs or r1 not in avail_regs:
            continue
        for met in ["density_total_ksloc", "density_high_ksloc", "density_rel_sec_ksloc", "rq3_lines_vs_total"]:
            sub0 = trend_df[(trend_df["regime"] == r0) & (trend_df["metric"] == met)].set_index("repository")
            sub1 = trend_df[(trend_df["regime"] == r1) & (trend_df["metric"] == met)].set_index("repository")
            repos = [r for r in sub0.index if r in sub1.index]
            hard_rev_raw, hard_rev_fdr, hard_rev_hr = [], [], []
            any_flip_raw, any_flip_fdr, any_flip_hr = [], [], []
            sen_sign_flip, tau_sign_flip = [], []
            for rp in repos:
                c0, c1 = sub0.loc[rp, "class"], sub1.loc[rp, "class"]
                tau0, tau1 = float(sub0.loc[rp, "stat_raw"]), float(sub1.loc[rp, "stat_raw"])
                sl0, sl1 = float(sub0.loc[rp, "sen_slope_raw"]), float(sub1.loc[rp, "sen_slope_raw"])
                q0, q1 = float(sub0.loc[rp, "q_fdr"]), float(sub1.loc[rp, "q_fdr"])
                phr0, phr1 = float(sub0.loc[rp, "p_hamed_rao"]), float(sub1.loc[rp, "p_hamed_rao"])

                if (c0 in {"SIG_INC", "SIG_POS"} and c1 in {"SIG_DEC", "SIG_NEG"}) or (
                    c0 in {"SIG_DEC", "SIG_NEG"} and c1 in {"SIG_INC", "SIG_POS"}
                ):
                    hard_rev_raw.append(rp)
                if (tau0 * tau1 < 0) and (q0 < 0.05) and (q1 < 0.05):
                    hard_rev_fdr.append(rp)
                if (tau0 * tau1 < 0) and (phr0 < 0.05) and (phr1 < 0.05):
                    hard_rev_hr.append(rp)

                if c0 != c1 and ("SIG" in c0 or "SIG" in c1):
                    any_flip_raw.append(rp)

                fdr_c0 = "SIG_POS" if (q0 < 0.05 and tau0 > 0) else ("SIG_NEG" if (q0 < 0.05 and tau0 < 0) else "NS")
                fdr_c1 = "SIG_POS" if (q1 < 0.05 and tau1 > 0) else ("SIG_NEG" if (q1 < 0.05 and tau1 < 0) else "NS")
                if fdr_c0 != fdr_c1:
                    any_flip_fdr.append(rp)

                hr_c0 = "SIG_POS" if (phr0 < 0.05 and tau0 > 0) else ("SIG_NEG" if (phr0 < 0.05 and tau0 < 0) else "NS")
                hr_c1 = "SIG_POS" if (phr1 < 0.05 and tau1 > 0) else ("SIG_NEG" if (phr1 < 0.05 and tau1 < 0) else "NS")
                if hr_c0 != hr_c1:
                    any_flip_hr.append(rp)

                if sl0 * sl1 < 0:
                    sen_sign_flip.append(rp)
                if tau0 * tau1 < 0:
                    tau_sign_flip.append(rp)

            rev_summary_rows.append(
                {
                    "comparison": desc,
                    "r0": r0,
                    "r1": r1,
                    "metric": met,
                    "n_repos": len(repos),
                    "hard_sig_sign_reversals_n": len(hard_rev_raw),
                    "hard_sig_sign_reversals_pct": round(100.0 * len(hard_rev_raw) / max(len(repos), 1), 1),
                    "hard_fdr_sign_reversals_n": len(hard_rev_fdr),
                    "hard_fdr_sign_reversals_pct": round(100.0 * len(hard_rev_fdr) / max(len(repos), 1), 1),
                    "hard_hamed_rao_reversals_n": len(hard_rev_hr),
                    "hard_hamed_rao_reversals_pct": round(100.0 * len(hard_rev_hr) / max(len(repos), 1), 1),
                    "sig_conclusion_flips_n": len(any_flip_raw),
                    "sig_conclusion_flips_pct": round(100.0 * len(any_flip_raw) / max(len(repos), 1), 1),
                    "fdr_conclusion_flips_n": len(any_flip_fdr),
                    "fdr_conclusion_flips_pct": round(100.0 * len(any_flip_fdr) / max(len(repos), 1), 1),
                    "hamed_rao_conclusion_flips_n": len(any_flip_hr),
                    "hamed_rao_conclusion_flips_pct": round(100.0 * len(any_flip_hr) / max(len(repos), 1), 1),
                    "sen_slope_sign_inversions_n": len(sen_sign_flip),
                    "sen_slope_sign_inversions_pct": round(100.0 * len(sen_sign_flip) / max(len(repos), 1), 1),
                    "tau_sign_inversions_pct": round(100.0 * len(tau_sign_flip) / max(len(repos), 1), 1),
                    "hard_reversal_examples": "; ".join(hard_rev_raw[:6]),
                    "hard_fdr_reversal_examples": "; ".join(hard_rev_fdr[:6]),
                    "hard_hr_reversal_examples": "; ".join(hard_rev_hr[:6]),
                }
            )
    pd.DataFrame(rev_summary_rows).to_csv(out_dir / "table2_longitudinal_trend_reversals_summary.csv", index=False)

    # =========================================================================
    # TABLE 3 & TABLE 3B: Zero-Residual 4D Decomposition + Hotspot Sensitivity
    # =========================================================================
    # Populate true corpus-observed rule counts for SonarQube specs (matching CLI linter observed counts)
    sq_spec_obs_map: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    for f_nm, t_kd, ph0, _, ph1, _ in sq_4d_specs:
        rk0 = set(sq_rules.get(ph0, {}).keys())
        rk1 = set(sq_rules.get(ph1, {}).keys())
        obs0_sq = rk0 & observed_sq_rules[ph0]
        obs1_sq = rk1 & observed_sq_rules[ph1]
        sq_spec_obs_map[(f_nm, t_kd)] = (len(obs0_sq), len(obs1_sq), len(obs0_sq & obs1_sq))

    ordered_4d_keys = [(f, k) for f, k, *_ in sq_4d_specs] + [(f, k) for f, k, *_ in linter_4d_specs]
    decomp_per_repo: List[Dict[str, Any]] = []
    for key_4d in ordered_4d_keys:
        rows_k = decomp_rows_by_spec.get(key_4d, [])
        if key_4d in sq_spec_obs_map:
            ob0, ob1, obc = sq_spec_obs_map[key_4d]
            for rk_row in rows_k:
                rk_row["r0_observed_rules"] = ob0
                rk_row["r1_observed_rules"] = ob1
                rk_row["rcap_observed_rules"] = obc
        decomp_per_repo.extend(rows_k)
    d3_repo_df = pd.DataFrame(decomp_per_repo)
    d3_repo_df.to_csv(out_dir / "table3_per_repo_4d.csv", index=False)

    d3_sum = (
        d3_repo_df.groupby(["transition_family", "metric_kind"], sort=False)
        .agg(
            n_repos=("repository", "count"),
            n_active_repos=("is_active_repo", "sum"),
            r0_rules=("r0_rules", "first"),
            r1_rules=("r1_rules", "first"),
            common_rules_rcap=("common_rules_rcap", "first"),
            r0_observed_rules=("r0_observed_rules", "first"),
            r1_observed_rules=("r1_observed_rules", "first"),
            rcap_observed_rules=("rcap_observed_rules", "first"),
            hs_migrated_rules_in_rcap=("hs_migrated_rules_in_rcap", "first"),
            mean_E_code=("mean_abs_E_code", "mean"),
            mean_E_ruleset=("mean_abs_E_rule_set", "mean"),
            mean_E_impl_drift=("mean_abs_E_impl_drift", "mean"),
            mean_E_tax_remap=("mean_abs_E_tax_remap", "mean"),
            mean_share_code_pct=("share_code_pct", "mean"),
            mean_share_ruleset_pct=("share_ruleset_pct", "mean"),
            mean_share_impl_pct=("share_impl_pct", "mean"),
            max_share_impl_pct=("share_impl_pct", "max"),
            mean_share_tax_pct=("share_tax_pct", "mean"),
            mean_rcap_births=("rcap_births_per_snap", "mean"),
            mean_rcap_deaths=("rcap_deaths_per_snap", "mean"),
            mean_rcap_weighted_births=("rcap_weighted_births_per_snap", "mean"),
            mean_rcap_weighted_deaths=("rcap_weighted_deaths_per_snap", "mean"),
        )
        .round(2)
        .reset_index()
    )
    d3_sum.to_csv(out_dir / "table3_4d_effect_decomposition_summary.csv", index=False)

    hs_all: List[Dict[str, Any]] = []
    for k4 in [(f, k) for f, k, *_ in sq_4d_specs]:
        hs_all.extend(hs_sens_rows_by_spec.get(k4, []))
    if hs_all:
        hs_df = pd.DataFrame(hs_all)
        hs_sum = (
            hs_df.groupby(["transition_family", "metric_kind"], sort=False)
            .agg(
                n_active_repos=("is_active", "sum"),
                rcap_total=("rcap_total", "first"),
                rcap_non_hotspot=("rcap_non_hotspot", "first"),
                rcap_hotspot_migrated=("rcap_hotspot_migrated", "first"),
                mean_share_code_pct=("share_code_pct", "mean"),
                mean_share_ruleset_pct=("share_ruleset_pct", "mean"),
                mean_share_impl_nonhs_detector_pct=("share_impl_nonhs_detector_pct", "mean"),
                mean_share_impl_hotspot_channel_pct=("share_impl_hotspot_channel_pct", "mean"),
                mean_share_tax_pct=("share_tax_pct", "mean"),
                mean_weighted_births_nonhs=("weighted_births_nonhs_per_snap", "mean"),
                mean_weighted_births_hotspot=("weighted_births_hotspot_per_snap", "mean"),
                mean_weighted_deaths=("weighted_deaths_per_snap", "mean"),
            )
            .round(2)
            .reset_index()
        )
        hs_sum.to_csv(out_dir / "table3b_hotspot_sensitivity.csv", index=False)

    # =========================================================================
    # TABLE 4 / 4A / 4B: Dual-Run Calibration Protocol & Direct S2 vs S1 Tests
    # =========================================================================
    strategies = [
        ("S0_Naive_Unadjusted", 0),
        ("S1_CommonRule_Catalog_Filter", 0),
        ("S2_SingleAnchor_Additive_Offset", 1),
        ("S3_SingleAnchor_Safeguarded_Ratio", 1),
        ("S4_Window3_Moment_Affine", 3),
    ]

    mid_repo_slope_errs: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)

    def eval_calibration_horizon(anchor_mode: str) -> List[Dict[str, Any]]:
        rows_h: List[Dict[str, Any]] = []
        for label_p, r0, r1, _ in calib_pairs:
            if r0 not in avail_regs or r1 not in avail_regs:
                continue
            d0 = snap_df[snap_df["regime"] == r0]
            d1 = snap_df[snap_df["regime"] == r1]
            rcap_series = rcap_series_map.get((r0, r1), {})

            per_strat_slope_errs: Dict[str, List[float]] = {}
            temp_rows: List[Dict[str, Any]] = []

            for sname, extra_runs in strategies:
                cliffs, sl_err_hist, sl_err_post_target = [], [], []
                sign_hist, sign_target, gate_agree, gate_total = 0, 0, 0, 0
                n_rep = 0

                for repo in all_repos_sorted:
                    g0 = d0[d0["repository"] == repo].sort_values("seq_idx")
                    g1 = d1[d1["repository"] == repo].sort_values("seq_idx")
                    if len(g0) < 10 or len(g0) != len(g1) or repo not in rcap_series:
                        continue
                    n_rep += 1
                    T = len(g0)
                    if anchor_mode == "Early (0.35T)":
                        k = max(3, int(round(0.35 * T)))
                    elif anchor_mode == "Late (0.65T)":
                        k = min(T - 3, int(round(0.65 * T)))
                    else:
                        k = T // 2
                    idx_k = k - 1
                    x_full = np.arange(1, T + 1, dtype=float)
                    x_post = np.arange(k + 1, T + 1, dtype=float)

                    y0 = g0["density_high_ksloc"].to_numpy(dtype=float)
                    y1 = g1["density_high_ksloc"].to_numpy(dtype=float)

                    if sname == "S0_Naive_Unadjusted":
                        y_stitch = np.concatenate([y0[:k], y1[k:]])
                    elif sname == "S1_CommonRule_Catalog_Filter":
                        v0_rc, v1_rc, _, _ = rcap_series[repo]
                        y_stitch = np.concatenate([v0_rc[:k], v1_rc[k:]])
                    elif sname == "S2_SingleAnchor_Additive_Offset":
                        offset = y1[idx_k] - y0[idx_k]
                        y_stitch = np.concatenate([y0[:k], np.maximum(0.0, y1[k:] - offset)])
                    elif sname == "S3_SingleAnchor_Safeguarded_Ratio":
                        if y1[idx_k] > 0.05 and y0[idx_k] > 0.05:
                            ratio = float(np.clip(y0[idx_k] / y1[idx_k], 0.05, 25.0))
                            y_post = y1[k:] * ratio
                        else:
                            offset = y1[idx_k] - y0[idx_k]
                            y_post = np.maximum(0.0, y1[k:] - offset)
                        y_stitch = np.concatenate([y0[:k], y_post])
                    else:
                        w_lo = max(0, k - 3)
                        m0 = float(np.mean(y0[w_lo:k]))
                        m1 = float(np.mean(y1[w_lo:k]))
                        sd0 = float(np.std(y0[w_lo:k]))
                        sd1 = float(np.std(y1[w_lo:k]))
                        if sd1 > 0.02 and sd0 > 0.02:
                            sc = float(np.clip(sd0 / sd1, 0.08, 15.0))
                        elif m1 > 0.05 and m0 > 0.05:
                            sc = float(np.clip(m0 / m1, 0.08, 15.0))
                        else:
                            sc = 1.0
                        y_post = np.maximum(0.0, m0 + sc * (y1[k:] - m1))
                        y_stitch = np.concatenate([y0[:k], y_post])

                    true_step = y1[k] - y1[idx_k]
                    obs_step = y_stitch[k] - y_stitch[idx_k]
                    cliffs.append(abs(obs_step - true_step))

                    sl0 = float(theilslopes(y0, x_full)[0]) if not np.all(y0 == y0[0]) else 0.0
                    sl1 = float(theilslopes(y1, x_full)[0]) if not np.all(y1 == y1[0]) else 0.0
                    sls = float(theilslopes(y_stitch, x_full)[0]) if not np.all(y_stitch == y_stitch[0]) else 0.0
                    sl_err_val = abs(sls - sl0)
                    sl_err_hist.append(sl_err_val)
                    if anchor_mode == "Mid (0.50T)":
                        mid_repo_slope_errs[(label_p, sname)][repo] = sl_err_val

                    y1_p = y1[k:]
                    ys_p = y_stitch[k:]
                    sl1_p = float(theilslopes(y1_p, x_post)[0]) if not np.all(y1_p == y1_p[0]) else 0.0
                    sls_p = float(theilslopes(ys_p, x_post)[0]) if not np.all(ys_p == ys_p[0]) else 0.0
                    sl_err_post_target.append(abs(sls_p - sl1_p))

                    if np.sign(sls) == np.sign(sl0):
                        sign_hist += 1
                    if np.sign(sls) == np.sign(sl1):
                        sign_target += 1

                    for tidx in range(k, T):
                        gate_total += 1
                        true_pass = y1[tidx] <= y1[tidx - 1] + 1e-9
                        cal_pass = y_stitch[tidx] <= y_stitch[tidx - 1] + 1e-9
                        if true_pass == cal_pass:
                            gate_agree += 1

                per_strat_slope_errs[sname] = sl_err_hist
                p_vs_naive = np.nan
                if sname != "S0_Naive_Unadjusted" and len(sl_err_hist) >= 5:
                    diff_arr = np.array(per_strat_slope_errs["S0_Naive_Unadjusted"]) - np.array(sl_err_hist)
                    if not np.all(diff_arr == 0):
                        _, p_vs_naive = wilcoxon(diff_arr)

                err_np = np.array(sl_err_hist, dtype=float)
                med_e = float(np.median(err_np)) if len(err_np) else np.nan
                q25_e, q75_e = np.percentile(err_np, [25.0, 75.0]) if len(err_np) else (np.nan, np.nan)

                temp_rows.append(
                    {
                        "anchor_horizon": anchor_mode,
                        "transition": label_p,
                        "strategy": sname,
                        "extra_scans_per_repo": extra_runs,
                        "n_repos": n_rep,
                        "boundary_cliff_mae_ksloc": round(float(np.mean(cliffs)), 3),
                        "full_slope_mae_vs_historical": round(float(np.mean(sl_err_hist)), 4),
                        "full_slope_median_iqr_vs_historical": f"{med_e:.4f} [{q25_e:.4f}, {q75_e:.4f}]",
                        "post_slope_mae_vs_target": round(float(np.mean(sl_err_post_target)), 4),
                        "wilcoxon_p_vs_naive": round(float(p_vs_naive), 5) if not np.isnan(p_vs_naive) else np.nan,
                        "wilcoxon_p_vs_s1": np.nan,
                        "full_series_hist_sign_match_pct": round(100.0 * sign_hist / max(n_rep, 1), 1),
                        "full_series_target_sign_match_pct": round(100.0 * sign_target / max(n_rep, 1), 1),
                        "post_switch_density_ratchet_agree_pct": round(100.0 * gate_agree / max(gate_total, 1), 1),
                        # Keep backward-compatible aliases for figure scripts:
                        "historical_trend_sign_match_pct": round(100.0 * sign_hist / max(n_rep, 1), 1),
                        "target_trend_sign_match_pct": round(100.0 * sign_target / max(n_rep, 1), 1),
                        "post_switch_ratchet_gate_agree_pct": round(100.0 * gate_agree / max(gate_total, 1), 1),
                    }
                )

            # Direct paired Wilcoxon tests against S1_CommonRule_Catalog_Filter
            s1_errs = np.array(per_strat_slope_errs["S1_CommonRule_Catalog_Filter"], dtype=float)
            for r_item in temp_rows:
                sn = r_item["strategy"]
                if sn not in ("S0_Naive_Unadjusted", "S1_CommonRule_Catalog_Filter"):
                    cur_errs = np.array(per_strat_slope_errs[sn], dtype=float)
                    d_s1 = s1_errs - cur_errs
                    if len(d_s1) >= 5 and not np.all(d_s1 == 0):
                        _, p_s1 = wilcoxon(d_s1)
                        r_item["wilcoxon_p_vs_s1"] = round(float(p_s1), 5)
            rows_h.extend(temp_rows)
        return rows_h

    t4_mid_rows = eval_calibration_horizon("Mid (0.50T)")
    t4_df = pd.DataFrame([{k: v for k, v in r.items() if k != "anchor_horizon"} for r in t4_mid_rows])
    t4_df.to_csv(out_dir / "table4_dual_run_calibration_5strategies.csv", index=False)

    t4b_all_rows = (
        eval_calibration_horizon("Early (0.35T)")
        + t4_mid_rows
        + eval_calibration_horizon("Late (0.65T)")
    )
    pd.DataFrame(t4b_all_rows).to_csv(out_dir / "table4b_multi_anchor_sensitivity.csv", index=False)

    # Track A (CI Release Gate Track): Exact Separation of Count-Ratchet Identity, Persistent-Anchor Baseline, & Rolling Step-FP Strict Gate
    t4a_rows = []
    for label_p, r0, r1, _ in calib_pairs:
        if r0 not in avail_regs or r1 not in avail_regs:
            continue
        d1_idx = snap_df[snap_df["regime"] == r1].set_index("snapshot_id")
        rcap_s = rcap_series_map.get((r0, r1), {})

        sw_naive_ok, sw_s1_cnt_ok, sw_1anchor_cnt_ok, sw_persist_rat_ok, n_sw = 0, 0, 0, 0, 0
        post_s1_cnt_ok, post_1anchor_cnt_ok, post_persist_rat_ok = 0, 0, 0
        post_step_fp_strict_ok, post_persist_fp_strict_ok, n_post = 0, 0, 0
        naive_unfiltered_alarms_at_rk1 = []
        tracka_unsuppressed_alarms_at_rk1 = []
        debt_retired_monotonic_pcts = []
        debt_retired_endpoint_pcts = []

        for repo in all_repos_sorted:
            grp = manifest_df[manifest_df["repository"] == repo].sort_values("seq_idx")
            snaps = grp["snapshot_id"].tolist()
            if len(snaps) < 10 or repo not in rcap_s or (label_p, repo, "r1") not in high_fp_pair_map:
                continue
            T = len(snaps)
            k = T // 2
            idx_k = k - 1  # Anchor release r_k; index `k` is first post-switch release r_{k+1}

            fp0_seq = high_fp_pair_map[(label_p, repo, "r0")]
            fp1_seq = high_fp_pair_map[(label_p, repo, "r1")]
            c0_k = sum(fp0_seq[idx_k].values())
            c1_seq = [sum(c.values()) for c in fp1_seq]
            _, _, c0_rc, c1_rc = rcap_s[repo]

            n_sw += 1
            # Reference count ratchet at r_{k+1} (index `k` vs `idx_k`)
            true_sw_ratchet = c1_seq[k] <= c1_seq[idx_k]
            naive_sw_ratchet = c1_seq[k] <= c0_k
            # Count-matched S1 R_cap ratchet (c1_rc[k] <= c0_rc[idx_k])
            s1_sw_ratchet = c1_rc[k] <= c0_rc[idx_k]
            # Exact 1-anchor count ratchet using dual-run scan at r_k (algebraically 100% identical to true_sw_ratchet)
            anchor_cnt_sw_ratchet = c1_seq[k] <= c1_seq[idx_k]

            # Persistent-anchor baseline suppression (Algorithm 1): B_act initialized to F_k at r_k, shrunk monotonically via B_act <- B_act & F_t
            b_act = Counter(fp1_seq[idx_k])
            b_init_total = sum(b_act.values())
            n_prev = 0  # |N_k| = |F_k \ B_k| = 0 at anchor r_k

            for t in range(k, T):
                n_post += 1
                ft = fp1_seq[t]
                net_new_persist = ft - b_act
                n_cur = sum(net_new_persist.values())
                b_act = b_act & ft  # Monotonic retirement of resolved legacy fingerprints

                true_rat = c1_seq[t] <= c1_seq[t - 1]
                s1_prev_cnt = c0_rc[idx_k] if t == k else c1_rc[t - 1]
                if (c1_rc[t] <= s1_prev_cnt) == true_rat:
                    post_s1_cnt_ok += 1

                # Rolling 1-anchor / step count ratchet: |F_t \ F_{t-1}| <= |F_{t-1} \ F_t| <=> |F_t| <= |F_{t-1}|
                prev_fp = fp1_seq[idx_k] if t == k else fp1_seq[t - 1]
                step_cnt_rat = sum((ft - prev_fp).values()) <= sum((prev_fp - ft).values())
                if step_cnt_rat == true_rat:
                    post_1anchor_cnt_ok += 1

                # Persistent-baseline net-new ratchet: |N_t| <= |N_{t-1}| (with |N_k| = 0 at t = k+1)
                persist_rat = n_cur <= n_prev
                if persist_rat == true_rat:
                    post_persist_rat_ok += 1

                if t == k:
                    if naive_sw_ratchet == true_sw_ratchet:
                        sw_naive_ok += 1
                    if s1_sw_ratchet == true_sw_ratchet:
                        sw_s1_cnt_ok += 1
                    if anchor_cnt_sw_ratchet == true_sw_ratchet:
                        sw_1anchor_cnt_ok += 1
                    if persist_rat == true_sw_ratchet:
                        sw_persist_rat_ok += 1
                    naive_unfiltered_alarms_at_rk1.append(c1_seq[k])
                    tracka_unsuppressed_alarms_at_rk1.append(n_cur)

                n_prev = n_cur

                # Strict zero-new-issue gate agreement against native/git-diff `new_high_gate_pass`
                native_zero_new = bool(d1_idx.loc[snaps[t], "new_high_gate_pass"])
                step_fp_zero_new = sum((ft - prev_fp).values()) == 0
                persist_fp_zero_new = n_cur == 0
                if step_fp_zero_new == native_zero_new:
                    post_step_fp_strict_ok += 1
                if persist_fp_zero_new == native_zero_new:
                    post_persist_fp_strict_ok += 1

            if b_init_total > 0:
                surviving_monotonic = sum(b_act.values())
                surviving_endpoint = sum((fp1_seq[idx_k] & fp1_seq[-1]).values())
                debt_retired_monotonic_pcts.append(100.0 * (1.0 - surviving_monotonic / b_init_total))
                debt_retired_endpoint_pcts.append(100.0 * (1.0 - surviving_endpoint / b_init_total))

        mean_uncal = float(np.mean(naive_unfiltered_alarms_at_rk1))
        mean_unsupp = float(np.mean(tracka_unsuppressed_alarms_at_rk1))
        alarm_red_pct = 100.0 * (1.0 - mean_unsupp / max(mean_uncal, 1e-9))

        t4a_rows.append(
            {
                "transition": label_p,
                "n_repos": n_sw,
                "n_repos_with_anchor_debt": len(debt_retired_monotonic_pcts),
                "n_post_transitions": n_post,
                "rk1_naive_ratchet_agree_pct": round(100.0 * sw_naive_ok / max(n_sw, 1), 1),
                "rk1_s1_rcap_count_ratchet_agree_pct": round(100.0 * sw_s1_cnt_ok / max(n_sw, 1), 1),
                "rk1_1anchor_count_ratchet_agree_pct": round(100.0 * sw_1anchor_cnt_ok / max(n_sw, 1), 1),
                "rk1_persist_fp_ratchet_agree_pct": round(100.0 * sw_persist_rat_ok / max(n_sw, 1), 1),
                "post_s1_rcap_count_ratchet_agree_pct": round(100.0 * post_s1_cnt_ok / max(n_post, 1), 1),
                "post_1anchor_count_ratchet_agree_pct": round(100.0 * post_1anchor_cnt_ok / max(n_post, 1), 1),
                "post_persist_fp_ratchet_agree_pct": round(100.0 * post_persist_rat_ok / max(n_post, 1), 1),
                "post_step_fp_strict_vs_newcode_agree_pct": round(100.0 * post_step_fp_strict_ok / max(n_post, 1), 1),
                "post_persist_fp_strict_vs_newcode_agree_pct": round(100.0 * post_persist_fp_strict_ok / max(n_post, 1), 1),
                "mean_uncalibrated_high_alarms_at_rk1": round(mean_uncal, 1),
                "mean_trackA_unsuppressed_new_alarms_at_rk1": round(mean_unsupp, 2),
                "rk1_alarm_flood_reduction_pct": round(alarm_red_pct, 1),
                "mean_baseline_retired_monotonic_by_T_pct": round(float(np.mean(debt_retired_monotonic_pcts)) if debt_retired_monotonic_pcts else 0.0, 1),
                "mean_baseline_retired_endpoint_by_T_pct": round(float(np.mean(debt_retired_endpoint_pcts)) if debt_retired_endpoint_pcts else 0.0, 1),
            }
        )
    pd.DataFrame(t4a_rows).to_csv(out_dir / "table4a_trackA_fingerprint_ci_calibration.csv", index=False)

    # Sensitivity check excluding `open-mmlab/mmdetection` (N = 48) for SQ 26.11 Comprehensive per-alert metrics
    mmdet_sens_rows = []
    rq3_mmdet_specs = [
        ("SQ 10.3 [N09] -> SQ 26.11 Comp MQR", "HIGH"),
        ("SQ 9.9 LTA -> SQ 26.11 Comp MQR", "HIGH"),
        ("SQ 26.11 Comp: Std -> MQR", "HIGH"),
        ("SQ 26.11 Comp: Std -> MQR (Rel/Sec)", "REL_SEC"),
        ("SQ 10.3 [N09] -> SQ 26.11 Comp MQR (Rel/Sec)", "REL_SEC"),
    ]
    for tf_s, mk_s in rq3_mmdet_specs:
        s_all = d3_repo_df[
            (d3_repo_df["transition_family"] == tf_s)
            & (d3_repo_df["metric_kind"] == mk_s)
            & (d3_repo_df["is_active_repo"] == 1)
        ]
        s_48 = s_all[s_all["repository"] != "open-mmlab/mmdetection"]
        assert len(s_all) == 49 and len(s_48) == 48, (
            f"Unexpected active repo counts for ({tf_s}, {mk_s}): {len(s_all)} vs {len(s_48)}"
        )
        mmdet_sens_rows.append(
            {
                "analysis": "RQ3_4D_Shares",
                "transition": tf_s,
                "metric": mk_s,
                "n49_all_repos": f"{s_all['share_code_pct'].mean():.2f}/{s_all['share_ruleset_pct'].mean():.2f}/{s_all['share_impl_pct'].mean():.2f}/{s_all['share_tax_pct'].mean():.2f}",
                "n48_excl_mmdetection": f"{s_48['share_code_pct'].mean():.2f}/{s_48['share_ruleset_pct'].mean():.2f}/{s_48['share_impl_pct'].mean():.2f}/{s_48['share_tax_pct'].mean():.2f}",
                "max_abs_diff": round(
                    float(
                        max(
                            abs(s_all[col].mean() - s_48[col].mean())
                            for col in ["share_code_pct", "share_ruleset_pct", "share_impl_pct", "share_tax_pct"]
                        )
                    ),
                    2,
                ),
            }
        )

    comp_cal_lbl = "SQ 10.3 [N09] -> SQ 26.11 Comp MQR"
    s0_map = mid_repo_slope_errs[(comp_cal_lbl, "S0_Naive_Unadjusted")]
    s2_map = mid_repo_slope_errs[(comp_cal_lbl, "S2_SingleAnchor_Additive_Offset")]
    assert len(s0_map) == 49 and len(s2_map) == 49 and "open-mmlab/mmdetection" in s0_map, (
        f"Missing RQ4 Mid calibration map for {comp_cal_lbl}"
    )
    s0_49 = float(np.mean(list(s0_map.values())))
    s2_49 = float(np.mean(list(s2_map.values())))
    s0_48 = float(np.mean([v for rp, v in s0_map.items() if rp != "open-mmlab/mmdetection"]))
    s2_48 = float(np.mean([v for rp, v in s2_map.items() if rp != "open-mmlab/mmdetection"]))
    mmdet_sens_rows.append(
        {
            "analysis": "RQ4_TrackB_Mid_HistSlopeMAE",
            "transition": comp_cal_lbl,
            "metric": "S0_Naive / S2_1Anchor",
            "n49_all_repos": f"{s0_49:.4f} / {s2_49:.4f}",
            "n48_excl_mmdetection": f"{s0_48:.4f} / {s2_48:.4f}",
            "max_abs_diff": round(float(max(abs(s0_49 - s0_48), abs(s2_49 - s2_48))), 4),
        }
    )
    assert len(mmdet_sens_rows) == 6, f"Expected 6 mmdetection sensitivity rows, got {len(mmdet_sens_rows)}"
    pd.DataFrame(mmdet_sens_rows).to_csv(out_dir / "table_mmdetection_truncation_sensitivity.csv", index=False)

    # =========================================================================
    # TABLE 5: Stratified 100 Source-Code Slices (10 Auditable Categories)
    # =========================================================================
    def select_diverse_slices(pool: List[Dict[str, Any]], target_n: int) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        used_idx: Set[int] = set()
        repo_cnt: Counter[str] = Counter()
        rk_cnt: Counter[str] = Counter()
        for max_repo, max_rk in [(1, 2), (2, 3), (3, 5), (5, 8)]:
            if len(selected) >= target_n:
                break
            for idx, item in enumerate(pool):
                if idx in used_idx:
                    continue
                rp = item["repository"]
                rk = item["_rk"]
                if repo_cnt[rp] >= max_repo or rk_cnt[rk] >= max_rk:
                    continue
                used_idx.add(idx)
                repo_cnt[rp] += 1
                rk_cnt[rk] += 1
                clean_item = {k: v for k, v in item.items() if k != "_rk"}
                selected.append(clean_item)
                if len(selected) >= target_n:
                    break
        return selected[:target_n]

    s1_sel = select_diverse_slices(cand_s1, 25)
    s2_hs_sel = select_diverse_slices(cand_s2_hs, 5)
    s2_nonhs_sel = select_diverse_slices(cand_s2_births_nonhs, 10)
    s2_d_sel = select_diverse_slices(cand_s2_drops, 10)
    s3_sel = select_diverse_slices(cand_s3, 25)
    s4_sel = select_diverse_slices(cand_s4, 25)
    slices = s1_sel + s2_nonhs_sel + s2_hs_sel + s2_d_sel + s3_sel + s4_sel

    slices_df = pd.DataFrame(slices[:100])
    slices_df.to_csv(out_dir / "table5_qualitative_100_slices.csv", index=False)
    (out_dir / "table5_qualitative_100_slices.json").write_text(
        json.dumps(slices[:100], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rc_tax_df = (
        slices_df.groupby(["slice_category", "root_cause"], sort=False)
        .agg(
            n_slices=("rule_id", "count"),
            n_distinct_canonical_rules=("canonical_rule", "nunique"),
            n_distinct_display_rules=("rule_id", "nunique"),
            n_distinct_repos=("repository", "nunique"),
            verification_basis=("verification_basis", "first"),
            example_rules=("canonical_rule", lambda s: "; ".join(list(dict.fromkeys(s))[:4])),
        )
        .reset_index()
    )
    rc_tax_df.to_csv(out_dir / "table5_root_cause_taxonomy_summary.csv", index=False)

    print("=== REVISED 49-REPO FULL STUDY ANALYSIS COMPLETE ===", flush=True)
    print(f"Extracted slices: {len(slices_df)}, categories: {len(rc_tax_df)}, unique canonical rules: {slices_df['canonical_rule'].nunique()}, unique repos: {slices_df['repository'].nunique()}", flush=True)


if __name__ == "__main__":
    analyze_full()
