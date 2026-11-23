#!/usr/bin/env python3
"""
Supplementary Archival Analysis & Figure Generation for SQJ Manuscript (D04-Plus Revision).
Computes strictly from local verified 49-repo outputs and rule catalogs:
  1. Repository-Cluster Block Bootstrap (2,000 resamples over 49 repo clusters) + per-repo Median/IQR flip rates
     -> results/full/table1d_cluster_bootstrap_and_iqr.csv
  2. Programmatic audit of N09 (Berciu et al., 2026, Tables 4 & 5) Top-17 Python rules across SQ 9.9, SQ 10.3,
     SQ 26.11 Core, and SQ 26.11 Comprehensive
     -> results/full/table1b_n09_top17_rules_fate.csv
  3. Synchronized verbatim Git source-code slices matching the manuscript Case 1-4 listings
     -> results/full/selected_code_listings.json
  4. Archival vector figures:
     -> results/full/figures/fig5_4d_repo_distributions.{pdf,png}
     -> results/full/figures/fig6_multi_anchor_and_debt_decay.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))
DATA_DIR = BASE_DIR / "data"
FULL_DIR = BASE_DIR / "results" / "full"
FIG_DIR = FULL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _parse_bool(val: str) -> int:
    v = val.strip().lower()
    return 1 if v in ("true", "1", "yes") else 0


def load_comparisons():
    comps = []
    with (FULL_DIR / "table1_multigate_flips_and_ranking.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            comps.append((r["comparison"], r["r0"], r["r1"]))
    return comps


def load_unified_metrics():
    rows_by_snap_regime = {}
    repos = set()
    with (FULL_DIR / "full_unified_metrics.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            snap = r["snapshot_id"]
            reg = r["regime"]
            seq_idx = int(r["seq_idx"])
            if seq_idx < 2:
                continue
            rows_by_snap_regime[(snap, reg)] = {
                "repository": r["repository"],
                "seq_idx": seq_idx,
                "native_qg_pass": _parse_bool(r["native_qg_pass"]),
                "cayc_zero_new_pass": _parse_bool(r["cayc_zero_new_pass"]),
                "new_high_gate_pass": _parse_bool(r["new_high_gate_pass"]),
                "new_rel_sec_gate_pass": _parse_bool(r["new_rel_sec_gate_pass"]),
                "ratchet_high_gate_pass": _parse_bool(r["ratchet_high_gate_pass"]),
                "overall_high_gate_pass": _parse_bool(r["overall_high_gate_pass"]),
            }
            repos.add(r["repository"])
    return rows_by_snap_regime, sorted(repos)


def compute_cluster_bootstrap_table(rows_by_snap_regime, repos, comparisons):
    rng = np.random.default_rng(20261115)
    B = 2000

    snaps_by_repo = defaultdict(set)
    for (snap, _), info in rows_by_snap_regime.items():
        snaps_by_repo[info["repository"]].add(snap)
    for repo in snaps_by_repo:
        snaps_by_repo[repo] = sorted(snaps_by_repo[repo])

    gates = [
        ("native_qg", "native_qg_pass"),
        ("cayc_allnew", "cayc_zero_new_pass"),
        ("new_high", "new_high_gate_pass"),
        ("new_rel_sec", "new_rel_sec_gate_pass"),
        ("overall_high", "overall_high_gate_pass"),
        ("ratchet_high", "ratchet_high_gate_pass"),
    ]

    out_rows = []
    n_repos = len(repos)

    for comp_label, r0, r1 in comparisons:
        repo_flips = {g[0]: np.zeros(n_repos, dtype=float) for g in gates}
        repo_totals = np.zeros(n_repos, dtype=float)
        flat_flips = {g[0]: [] for g in gates}

        for idx, repo in enumerate(repos):
            snaps = snaps_by_repo[repo]
            repo_totals[idx] = len(snaps)
            for s in snaps:
                d0 = rows_by_snap_regime[(s, r0)]
                d1 = rows_by_snap_regime[(s, r1)]
                for g_key, col in gates:
                    flipped = 1.0 if d0[col] != d1[col] else 0.0
                    repo_flips[g_key][idx] += flipped
                    flat_flips[g_key].append(flipped)

        total_trans = int(repo_totals.sum())
        row_out = {
            "comparison": comp_label,
            "r0": r0,
            "r1": r1,
            "n_repos": n_repos,
            "n_transitions": total_trans,
        }

        cluster_idx = rng.integers(0, n_repos, size=(B, n_repos))
        trans_idx = rng.integers(0, total_trans, size=(B, total_trans))

        for g_key, _ in gates:
            flips_arr = repo_flips[g_key]
            flat_arr = np.array(flat_flips[g_key], dtype=float)
            pooled_pct = 100.0 * flips_arr.sum() / total_trans

            t_rates = 100.0 * flat_arr[trans_idx].mean(axis=1)
            t_lo, t_hi = np.percentile(t_rates, [2.5, 97.5])

            c_num = flips_arr[cluster_idx].sum(axis=1)
            c_den = repo_totals[cluster_idx].sum(axis=1)
            c_rates = 100.0 * c_num / c_den
            c_lo, c_hi = np.percentile(c_rates, [2.5, 97.5])

            per_repo_rates = 100.0 * flips_arr / repo_totals
            med = float(np.median(per_repo_rates))
            q25, q75 = np.percentile(per_repo_rates, [25.0, 75.0])
            repos_affected = int((flips_arr > 0).sum())

            row_out[f"{g_key}_pooled_pct"] = round(pooled_pct, 2)
            row_out[f"{g_key}_trans_ci95"] = f"[{t_lo:.1f}, {t_hi:.1f}]"
            row_out[f"{g_key}_cluster_ci95"] = f"[{c_lo:.1f}, {c_hi:.1f}]"
            row_out[f"{g_key}_repo_median_iqr"] = f"{med:.1f} [{q25:.1f}, {q75:.1f}]"
            row_out[f"{g_key}_repos_affected"] = f"{repos_affected}/{n_repos}"

        out_rows.append(row_out)

    out_csv = FULL_DIR / "table1d_cluster_bootstrap_and_iqr.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[OK] Wrote {out_csv} ({len(out_rows)} rows)")
    return out_rows


def audit_n09_top17_rules():
    n09_rules = [
        ("python:S3776", "Cognitive Complexity of functions should not be too high", "55", "621,150", "Tables 4 & 5"),
        ("python:S1192", "String literals should not be duplicated", "54", "1,796,923", "Tables 4 & 5"),
        ("python:S1481", "Unused local variables should be removed", "52", "428,795", "Tables 4 & 5"),
        ("python:S117", "Local variable and function parameter names should comply with a naming convention", "44", "1,818,257", "Tables 4 & 5"),
        ("python:S1172", "Unused method/function parameters should be removed", "38", "255,584", "Tables 4 & 5"),
        ("python:S1135", "Track uses of \"TODO\" tags", "30", "166,245", "Tables 4 & 5"),
        ("python:S125", "Sections of code should not be commented out", "25", "252,414", "Tables 4 & 5"),
        ("python:S1542", "Function names should comply with a naming convention", "19", "135,195", "Tables 4 & 5"),
        ("python:S1186", "Functions and methods should not be empty", "18", "n/r (Table 4)", "Table 4"),
        ("python:S5806", "Builtins should not be shadowed by local variables", "16", "159,834", "Tables 4 & 5"),
        ("python:S6711", "numpy.random.Generator should be preferred to numpy.random.RandomState", "16", "n/r (Table 4)", "Table 4"),
        ("python:S100", "Method names should comply with a naming convention", "14", "n/r (Table 4)", "Table 4"),
        ("python:S5754", "\"SystemExit\" should be re-raised", "13", "n/r (Table 4)", "Table 4"),
        ("python:S112", "\"Exception\" and \"BaseException\" should not be raised", "11", "n/r (Table 4)", "Table 4"),
        ("python:S116", "Field names should comply with a naming convention", "10", "n/r (Table 4)", "Table 4"),
        ("python:S1940", "Boolean checks should not be inverted", "10", "106,172", "Tables 4 & 5"),
        ("python:PrintStatementUsage", "The \"print\" statement should not be used", "n/r (Table 5)", "121,893", "Table 5"),
    ]

    cat_dir = DATA_DIR / "sq_catalogs"
    sq99 = {r["key"]: r for r in json.loads((cat_dir / "sq99_std_rules.json").read_text(encoding="utf-8"))}
    sq103 = {r["key"]: r for r in json.loads((cat_dir / "sq103_n09_rules.json").read_text(encoding="utf-8"))}
    sq26_core = {r["key"]: r for r in json.loads((cat_dir / "sq26_mqr_rules.json").read_text(encoding="utf-8"))}
    sq26_comp = {r["key"]: r for r in json.loads((cat_dir / "sq26_comp_mqr_rules.json").read_text(encoding="utf-8"))}

    out = []
    deactivated_in_core = 0
    for key, desc_fallback, n09_repos, n09_issues, _ in n09_rules:
        r99 = sq99.get(key)
        r103 = sq103.get(key)
        rcore = sq26_core.get(key)
        rcomp = sq26_comp.get(key)

        ref = rcomp or rcore or r103 or r99 or {}
        std_type = ref.get("type", "N/A")
        std_sev = ref.get("severity", "N/A")
        cc_attr = ref.get("cleanCodeAttribute", "N/A")
        impacts = ref.get("impacts", [])
        imp_str = "; ".join(f"{i['softwareQuality']}:{i['severity']}" for i in impacts) if impacts else "N/A"

        in_core = rcore is not None
        in_comp = rcomp is not None
        if r103 is not None and not in_core:
            deactivated_in_core += 1

        fates = []
        if r103 is not None and not in_core and in_comp:
            fates.append("Silently deactivated in SQ 26.11 Core; retained in Comprehensive")
        elif r103 is not None and in_core:
            fates.append("Retained in SQ 26.11 Core & Comprehensive")
        elif r103 is not None and not in_core and not in_comp:
            fates.append("Removed in SQ 26.11")

        sev_map = {"INFO": 1, "MINOR": 2, "MAJOR": 3, "CRITICAL": 4, "BLOCKER": 5}
        mqr_map = {"INFO": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "BLOCKER": 5}
        if impacts and std_sev in sev_map:
            max_mqr = max(mqr_map.get(i["severity"], 0) for i in impacts)
            if max_mqr > sev_map[std_sev]:
                fates.append(f"MQR promoted ({std_sev} -> {imp_str})")
            elif max_mqr < sev_map[std_sev]:
                fates.append(f"MQR demoted ({std_sev} -> {imp_str})")
            pillars = {i["softwareQuality"] for i in impacts}
            if std_type == "CODE_SMELL" and ("RELIABILITY" in pillars or "SECURITY" in pillars):
                fates.append(f"Cross-pillar remapped ({std_type} -> {imp_str})")

        out.append({
            "rule_key": key,
            "rule_name": ref.get("name", desc_fallback),
            "n09_projects_of_57": n09_repos,
            "n09_reported_issues": n09_issues,
            "in_sq99_default": "Yes" if r99 else "No",
            "in_sq103_n09_default": "Yes" if r103 else "No",
            "in_sq26_core_default": "Yes" if in_core else "No",
            "in_sq26_comp_default": "Yes" if in_comp else "No",
            "legacy_std_type_sev": f"{std_type} / {std_sev}",
            "mqr_clean_code_attr": cc_attr,
            "mqr_impacts": imp_str,
            "migration_fate": "; ".join(fates) if fates else "Preserved",
        })

    out_csv = FULL_DIR / "table1b_n09_top17_rules_fate.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"[OK] Wrote {out_csv} ({len(out)} rules; deactivated in SQ 26.11 Core: {deactivated_in_core}/{len(out)})")
    return out


def plot_fig5_4d_and_scale():
    per_repo_4d = []
    with (FULL_DIR / "table3_per_repo_4d.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["is_active_repo"]) == 1:
                per_repo_4d.append(r)

    target_specs = [
        ("SQ 26.11 Comp: Std -> MQR (Rel/Sec)", "REL_SEC", "SQ 26.11 Comp.\nStd→MQR (Rel/Sec)"),
        ("SQ 10.3 [N09] -> SQ 26.11 Core MQR", "HIGH", "SQ 10.3→26.11\nCore MQR (High)"),
        ("SQ 10.3 [N09] -> SQ 26.11 Comp MQR", "HIGH", "SQ 10.3→26.11\nComp. MQR (High)"),
        ("Flake8 7.1 -> Ruff 0.17.5 (Extended)", "TOTAL", "Flake8 7.1→\nRuff 0.17.5 (Total)"),
        ("Pylint 3.3 -> Ruff 0.17.5 (Pylint-PL Aligned)", "HIGH", "Pylint 3.3→\nRuff 0.17.5 PL (High)"),
    ]

    scale_rows = []
    with (FULL_DIR / "table1c_scale_stratified_flips.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scale_rows.append(r)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.22, 1.0]})

    comps = [
        ("share_code_pct", r"$|E_{\mathrm{code}}|$ (Code Evolution)", "#1f77b4"),
        ("share_ruleset_pct", r"$|E_{\mathrm{rule\text{-}set}}|$ (Rule-Set Diff)", "#ff7f0e"),
        ("share_impl_pct", r"$|E_{\mathrm{impl\text{-}drift}}|$ (Shared-Rule Drift)", "#2ca02c"),
        ("share_tax_pct", r"$|E_{\mathrm{tax\text{-}remap}}|$ (Taxon./Severity Remap)", "#d62728"),
    ]

    rng = np.random.default_rng(42)
    x_centers = np.arange(len(target_specs))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    width = 0.15

    for c_idx, (col, label, color) in enumerate(comps):
        data_series = []
        for fam, mkind, _ in target_specs:
            vals = [
                float(r[col]) for r in per_repo_4d
                if r["transition_family"] == fam and r["metric_kind"] == mkind
            ]
            data_series.append(vals)

        pos = x_centers + offsets[c_idx]
        ax1.boxplot(
            data_series,
            positions=pos,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.3},
            boxprops={"facecolor": color, "alpha": 0.45, "edgecolor": color, "linewidth": 1.0},
            whiskerprops={"color": color, "linewidth": 0.9},
            capprops={"color": color, "linewidth": 0.9},
        )
        for g_i, vals in enumerate(data_series):
            if vals:
                jitter = rng.uniform(-0.04, 0.04, size=len(vals))
                ax1.scatter(
                    np.full(len(vals), pos[g_i]) + jitter,
                    vals,
                    s=11,
                    color=color,
                    alpha=0.72,
                    edgecolors="none",
                    zorder=3,
                )
        ax1.plot([], [], color=color, marker="s", markersize=7, linestyle="None", alpha=0.75, label=label)

    ax1.set_xticks(x_centers)
    ax1.set_xticklabels([t[2] for t in target_specs], fontsize=8.3)
    ax1.set_ylabel("Per-Repository Attribution Share (%)")
    ax1.set_ylim(-3, 122)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_title("(a) Active-Repository 4D Share Dispersion", fontweight="bold")
    ax1.legend(loc="upper center", fontsize=7.6, framealpha=0.95, ncol=2)

    scale_targets = [
        ("SQ 26.11 Core: Standard -> MQR (Pure Taxonomy, 232 rules)", "new_rel_sec_attr_flip_pct", "SQ 26.11 Core\nStd→MQR (Rel/Sec)"),
        ("SQ 10.3 [N09] -> SQ 26.11 Core MQR (Default Post-N09 Upgrade)", "new_high_flip_pct", "SQ 10.3→26.11\nCore (New High)"),
        ("SQ 10.3 [N09] -> SQ 26.11 Comprehensive MQR (Rule-Preserving Upgrade)", "ratchet_high_flip_pct", "SQ 10.3→26.11\nComp. (Ratchet)"),
        ("Flake8 7.1 -> Ruff 0.17.5 (Extended Profile)", "ratchet_high_flip_pct", "Flake8 7.1→\nRuff Ext. (Ratchet)"),
        ("Pylint 3.3 -> Ruff 0.17.5 (Pylint-PL Aligned Profile)", "ratchet_high_flip_pct", "Pylint 3.3→\nRuff PL (Ratchet)"),
    ]
    tiers = [
        ("Small (<25k SLOC)", "Small (<25k SLOC, n=17)", "#6baed6"),
        ("Medium (25k-150k SLOC)", "Medium (25k–150k SLOC, n=17)", "#2171b5"),
        ("Large/Monorepo (>150k SLOC)", "Large/Monorepo (>150k–972k, n=15)", "#08306b"),
    ]

    x2 = np.arange(len(scale_targets))
    bar_w = 0.25
    for t_i, (tier_key, tier_label, t_color) in enumerate(tiers):
        vals = []
        for comp_name, metric_col, _ in scale_targets:
            match = [
                float(r[metric_col]) for r in scale_rows
                if r["comparison"] == comp_name and r["scale_tier"] == tier_key
            ]
            vals.append(match[0] if match else 0.0)
        from decimal import Decimal, ROUND_HALF_UP
        bars = ax2.bar(x2 + (t_i - 1) * bar_w, vals, width=bar_w, color=t_color, edgecolor="white", linewidth=0.6, label=tier_label)
        for b in bars:
            h = b.get_height()
            h_lbl = Decimal(f"{h:.2f}").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 0.7, f"{h_lbl}", ha="center", va="bottom", fontsize=7.2)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([s[2] for s in scale_targets], fontsize=8.2)
    ax2.set_ylabel("Quality Gate Flip Rate (%)")
    ax2.set_ylim(0, 48)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_title("(b) Scale-Stratified Gate Flip Rates Across SLOC Tiers", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_4d_repo_distributions.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_4d_repo_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved fig5_4d_repo_distributions.{pdf,png}")


def plot_fig6_multi_anchor_and_debt():
    ma_rows = []
    with (FULL_DIR / "table4b_multi_anchor_sensitivity.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ma_rows.append(r)

    horizons_avail = sorted({r["anchor_horizon"] for r in ma_rows})
    tracka_rows = []
    with (FULL_DIR / "table4a_trackA_fingerprint_ci_calibration.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tracka_rows.append(r)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.3,
        "axes.labelsize": 9.3,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.1), gridspec_kw={"width_ratios": [1.12, 1.0]})

    hz_order = []
    for prefix in ("Early", "Mid", "Late"):
        matches = [h for h in horizons_avail if h.startswith(prefix)]
        if matches:
            hz_order.append(matches[0])

    strategies = [
        ("S0_Naive_Unadjusted", "S0: Unadjusted Switch (0 extra scans)", "#d62728", "-o"),
        ("S1_CommonRule_Catalog_Filter", r"S1: Catalog $\mathcal{R}_{\cap}$ Filter (0 extra)", "#ff7f0e", "--s"),
        ("S3_SingleAnchor_Safeguarded_Ratio", "S3: 1-Anchor Safeguarded Ratio (1 extra)", "#2ca02c", "-D"),
        ("S4_Window3_Moment_Affine", "S4: W=3 Moment-Affine (3 extra)", "#9467bd", ":v"),
        ("S2_SingleAnchor_Additive_Offset", "S2: 1-Anchor Additive Offset (1 extra)", "#1f77b4", "-^"),
    ]

    xh = np.arange(len(hz_order))
    label_offsets = {
        "S0_Naive_Unadjusted": [(0.0, 0.011), (0.0, 0.011), (0.0, 0.011)],
        "S1_CommonRule_Catalog_Filter": [(0.0, 0.015), (0.0, 0.012), (0.0, 0.012)],
        "S3_SingleAnchor_Safeguarded_Ratio": [(0.15, -0.004), (0.0, 0.010), (0.14, -0.008)],
        "S4_Window3_Moment_Affine": [(-0.15, -0.005), (0.13, -0.004), (-0.14, 0.008)],
        "S2_SingleAnchor_Additive_Offset": [(0.0, -0.025), (0.0, -0.025), (0.0, -0.024)],
    }
    for strat_key, strat_lbl, color, style in strategies:
        means = []
        for hz in hz_order:
            vals = [
                float(r["full_slope_mae_vs_historical"])
                for r in ma_rows
                if r["anchor_horizon"] == hz and r["strategy"] == strat_key
            ]
            means.append(float(np.mean(vals)))
        ax1.plot(xh, means, style, color=color, linewidth=2.1, markersize=6.5, label=strat_lbl)
        for idx_x, (xi, yi) in enumerate(zip(xh, means)):
            dx, dy = label_offsets[strat_key][idx_x]
            ax1.text(xi + dx, yi + dy, f"{yi:.3f}", ha="center", va="bottom", fontsize=7.4, color=color, fontweight="bold")

    ax1.set_xticks(xh)
    ax1.set_xticklabels(hz_order, fontsize=8.8)
    ax1.set_xlim(-0.32, 2.32)
    ax1.set_ylabel("5-Family Mean Full-Series Slope MAE (/kSLOC/rel)")
    ax1.set_ylim(0.01, 0.45)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.set_title("(a) 5-Family Mean Slope MAE Across Switch Horizons", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.94)

    short_labels = {
        "SQ 10.3 [N09] -> SQ 26.11 Core MQR": "SQ 10.3→\n26.11 Core",
        "SQ 10.3 [N09] -> SQ 26.11 Comp MQR": "SQ 10.3→\n26.11 Comp.",
        "SQ 9.9 LTA Std -> SQ 26.11 Core MQR": "SQ 9.9 LTA→\n26.11 Core",
        "Flake8 7.1 -> Ruff 0.17.5 (Extended)": "Flake8 7.1→\nRuff Ext.",
        "Pylint 3.3 -> Ruff 0.17.5 (Pylint-PL Aligned)": "Pylint 3.3→\nRuff PL",
    }
    x_ta = np.arange(len(tracka_rows))
    alarms = [float(r["mean_uncalibrated_high_alarms_at_rk1"]) for r in tracka_rows]
    unsupp = [float(r["mean_trackA_unsuppressed_new_alarms_at_rk1"]) for r in tracka_rows]
    retired = [float(r["mean_baseline_retired_monotonic_by_T_pct"]) for r in tracka_rows]
    strict_agree = [float(r["post_step_fp_strict_vs_newcode_agree_pct"]) for r in tracka_rows]
    lbls = [short_labels.get(r["transition"], r["transition"]) for r in tracka_rows]

    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x_ta - 0.22, alarms, width=0.22, color="#c51b7d", alpha=0.84, label=r"Uncalibrated Alarms at $r_{k+1}$")
    ax2.bar(x_ta, unsupp, width=0.22, color="#f1b6da", edgecolor="#8e0152", linewidth=0.8, label=r"Track A Net-New Alarms at $r_{k+1}$")
    bars2 = ax2_twin.bar(x_ta + 0.22, retired, width=0.22, color="#2b8cbe", alpha=0.85, label=r"Anchor Fingerprints Retired by $r_T$ (%)")
    ax2_twin.plot(x_ta + 0.22, strict_agree, "o--", color="#006d2c", linewidth=1.8, markersize=6, label="Step-FP Strict Gate Agree (%)")

    for b in bars1:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 16, f"{h:.0f}", ha="center", va="bottom", fontsize=7.3, color="#8e0152", fontweight="bold")
    for b in bars2:
        h = b.get_height()
        ax2_twin.text(b.get_x() + b.get_width() / 2.0, h + 1.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.2, color="#08519c")

    ax2.set_xticks(x_ta)
    ax2.set_xticklabels(lbls, fontsize=8.3)
    ax2.set_ylabel(r"Mean High-Severity Alarms at $r_{k+1}$", color="#8e0152")
    ax2_twin.set_ylabel("Percentage (%)", color="#08519c")
    ax2.set_ylim(0, 1080)
    ax2_twin.set_ylim(0, 136)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_title(r"(b) First Post-Switch ($r_{k+1}$) Alarm Suppression & Baseline Retirement", fontweight="bold")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.2, framealpha=0.94, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_multi_anchor_and_debt_decay.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_multi_anchor_and_debt_decay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved fig6_multi_anchor_and_debt_decay.{pdf,png}")


def sync_selected_code_listings():
    slices = json.loads((FULL_DIR / "table5_qualitative_100_slices.json").read_text(encoding="utf-8"))
    targets = [
        ("Case1_MQR_Upward_High", "aio-libs/aiohttp", "python:S7502"),
        ("Case1_MQR_RelSec_Remap", "ansible/ansible", "python:S8786"),
        ("Case2_SQ_NonHotspot_Birth", "celery/celery", "python:S2638"),
        ("Case2_SQ_Hotspot_Channel_Migration", "binux/pyspider", "python:S4790"),
        ("Case2_SQ_TestScope_Silencing", "aio-libs/aiohttp", "python:S100"),
        ("Case3_Flake8_Ruff_Pragma_B023", "aws/serverless-application-model", "B023"),
        ("Case3_Flake8_Ruff_Visitor_B008", "aio-libs/aiohttp", "B008"),
        ("Case4_Pylint_Ruff_Astroid_W0102_B006", "aws/serverless-application-model", "EQ:B006_MUTABLE_DEFAULT"),
        ("Case4_Pylint_Ruff_Syntactic_W0104_B018", "ansible/ansible", "EQ:B018_POINTLESS_STMT"),
    ]
    selected = {}
    for key_lbl, repo, c_rule in targets:
        matches = [s for s in slices if s["repository"] == repo and s["canonical_rule"] == c_rule]
        if matches:
            selected[key_lbl] = matches[0]
    (FULL_DIR / "selected_code_listings.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] Synced selected_code_listings.json ({len(selected)} exact manuscript listings)")


def main():
    comparisons = load_comparisons()
    rows_by_snap_regime, repos = load_unified_metrics()
    compute_cluster_bootstrap_table(rows_by_snap_regime, repos, comparisons)
    audit_n09_top17_rules()
    sync_selected_code_listings()
    plot_fig5_4d_and_scale()
    plot_fig6_multi_anchor_and_debt()


if __name__ == "__main__":
    main()
