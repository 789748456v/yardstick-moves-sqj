#!/usr/bin/env python3
"""
Generate publication-grade vector (PDF) and high-DPI raster (PNG) figures for D04-Plus Full Study (SQJ Revision).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))


def main() -> None:
    res_dir = BASE_DIR / "results" / "full"
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.3,
            "figure.dpi": 300,
        }
    )

    t1 = pd.read_csv(res_dir / "table1_multigate_flips_and_ranking.csv")
    snap = pd.read_csv(res_dir / "full_unified_metrics.csv")
    t2_repo = pd.read_csv(res_dir / "table2_per_repo_trends.csv")
    t3 = pd.read_csv(res_dir / "table3_4d_effect_decomposition_summary.csv")
    t4 = pd.read_csv(res_dir / "table4_dual_run_calibration_5strategies.csv")

    # =========================================================================
    # FIGURE 1 (Manuscript Fig. 2): Multi-Gate Flip Rate & Rank-Inversion Heatmap
    # =========================================================================
    cols = [
        "native_qg_flip_pct",
        "cayc_allnew_flip_pct",
        "new_high_flip_pct",
        "new_rel_sec_attr_flip_pct",
        "ratchet_high_flip_pct",
        "pairwise_rank_inversion_high_pct",
    ]
    col_labels = [
        "Native Gate\nFlip (%)",
        "CaYC All-New\nFlip (%)",
        "Zero New High\nFlip (%)",
        "Rel/Sec Attr\nGate Flip (%)",
        "High Ratchet\nFlip (%)",
        "Pairwise Rank\nInversion (%)",
    ]
    def clean_label(s: str) -> str:
        return (
            s.replace(" [N09]", "")
            .replace("Default Post-N09 Upgrade", "Fresh-Default Binding")
            .replace("Pure LTA Engine+Profile Upgrade", "LTA → Fresh-Default Std")
            .replace("LTA -> Modern Default MQR", "LTA → Fresh-Default MQR")
            .replace("Default Core (232)", "Core (232)")
            .replace(" (Pure ", " (")
            .replace(" Profile)", ")")
            .replace(" -> ", " → ")
        )

    mat = t1[cols].to_numpy(dtype=float)
    row_labels = [clean_label(c) for c in t1["comparison"]]
    is_cli = [not str(c).startswith("SQ ") for c in t1["comparison"]]

    # Mask CLI-inapplicable server/pillar mirror columns (Native Gate & Rel/Sec Attr Gate)
    mask = np.zeros_like(mat, dtype=bool)
    for i, cli_flag in enumerate(is_cli):
        if cli_flag:
            mask[i, 0] = True  # Native Gate (no server gate API in CLI linters)
            mask[i, 3] = True  # Rel/Sec Attr Gate (no multi-pillar taxonomy in CLI linters)

    mat_masked = np.ma.masked_array(mat, mask=mask)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#ECEFF1")

    fig, ax = plt.subplots(figsize=(11.8, 6.5))
    im = ax.imshow(mat_masked, cmap=cmap, aspect="auto", vmin=0, vmax=max(50.0, float(np.nanmax(mat))))
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Subtle horizontal divider between SonarQube rows (0..8) and Python CLI rows (9..14)
    first_cli = next((i for i, fl in enumerate(is_cli) if fl), None)
    if first_cli is not None and first_cli > 0:
        ax.axhline(first_cli - 0.5, color="#37474F", linewidth=1.2, linestyle="--")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mask[i, j]:
                ax.text(j, i, "N/A", ha="center", va="center", color="#607D8B", fontsize=7.4, style="italic")
            else:
                val = mat[i, j]
                txt_color = "white" if val >= 28.0 else "black"
                ax.text(j, i, f"{val:.2f}%", ha="center", va="center", color=txt_color, fontsize=7.6)
    ax.set_title("RQ1: Multi-Gate Flip Rates & Pairwise Project Rank Inversions Across Regimes (N/A = CLI-Inapplicable)", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Percentage (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_multigate_flip_heatmap.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig1_multigate_flip_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2 (Manuscript Fig. 3): Representative Longitudinal Hard Reversals
    # =========================================================================
    regime_display = {
        "sq99_std": "SQ 9.9 Std",
        "sq103_std": "SQ 10.3 Std",
        "sq103_cleancode": "SQ 10.3 CleanCode",
        "sq26_core_std": "SQ 26.11 Core Std",
        "sq26_core_mqr": "SQ 26.11 Core MQR",
        "sq26_comp_std": "SQ 26.11 Comp Std",
        "sq26_comp_mqr": "SQ 26.11 Comp MQR",
        "flake8_71": "Flake8 7.1",
        "ruff_modern_default": "Ruff 0.17.5 (Def)",
        "ruff_modern_extended": "Ruff 0.17.5 (Ext)",
        "ruff_modern_pylint": "Ruff 0.17.5 (PL)",
        "pylint_33": "Pylint 3.3",
    }
    metric_display = {
        "density_high_ksloc": "High-Severity Density",
        "density_total_ksloc": "Total Issue Density",
    }
    showcase = [
        (
            "encode/django-rest-framework",
            "density_high_ksloc",
            "(a) Rule-Preserving & Core SQ Upgrades",
            ["sq103_std", "sq26_comp_mqr", "sq26_core_mqr", "sq99_std"],
            "lower right",
        ),
        (
            "huggingface/pytorch-image-models",
            "density_high_ksloc",
            "(b) Built-in Profile Split (Comp 407 vs. Core 232)",
            ["sq103_std", "sq26_comp_mqr", "sq26_core_mqr", "sq99_std"],
            "center right",
        ),
        (
            "celery/celery",
            "density_total_ksloc",
            "(c) Flake8 → Ruff 0.17.5 Migration & SQ Upgrades",
            ["flake8_71", "ruff_modern_default", "sq103_std", "sq26_core_mqr"],
            "center right",
        ),
        (
            "python-poetry/poetry",
            "density_high_ksloc",
            "(d) Pylint 3.3 → Ruff 0.17.5 (PL-Aligned) Migration",
            ["pylint_33", "ruff_modern_pylint", "ruff_modern_extended", "sq26_comp_mqr"],
            "center right",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6))
    for ax, (repo, mcol, sub_lbl, regs, leg_loc) in zip(axes.flat, showcase):
        sub_r = snap[snap["repository"] == repo]
        if sub_r.empty:
            continue
        for reg in regs:
            g = sub_r[sub_r["regime"] == reg].sort_values("seq_idx")
            if g.empty:
                continue
            y = g[mcol].to_numpy(dtype=float)
            span = float(y.max() - y.min())
            y_norm = (y - float(y.min())) / span if span > 1e-9 else np.full_like(y, 0.5)
            tr_row = t2_repo[
                (t2_repo["repository"] == repo) & (t2_repo["regime"] == reg) & (t2_repo["metric"] == mcol)
            ]
            tau_txt = ""
            if not tr_row.empty:
                tau_v = float(tr_row["stat"].iloc[0])
                p_v = float(tr_row["p_val"].iloc[0])
                q_v = float(tr_row["q_fdr"].iloc[0]) if "q_fdr" in tr_row.columns else 1.0
                if q_v < 0.05:
                    sig_mark = "*†"
                elif p_v < 0.05:
                    sig_mark = "*"
                else:
                    sig_mark = " (ns)"
                tau_txt = f" [τ={tau_v:+.2f}{sig_mark}]"
            disp_reg = regime_display.get(reg, reg)
            ax.plot(g["seq_idx"], y_norm, marker="o", markersize=3.5, linewidth=1.6, label=f"{disp_reg}{tau_txt}")
        ax.set_ylim(-0.06, 1.08)
        ax.set_title(f"{sub_lbl}\n{repo} ({metric_display.get(mcol, mcol)})", fontsize=9.8)
        ax.set_xlabel("Release Sequence Index")
        ax.set_ylabel("Per-Series Min–Max Normalized Density [0, 1]")
        ax.legend(loc=leg_loc, frameon=True, framealpha=0.90, fontsize=7.8)
        ax.grid(alpha=0.25)
    fig.suptitle("RQ2: Opposite-Sign Longitudinal Slope Reversals (* raw p < 0.05; † BH-FDR q < 0.05)", y=1.02, fontsize=11.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_longitudinal_hard_reversals.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig2_longitudinal_hard_reversals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 3 (Manuscript Fig. 4): 4D Accounting Decomposition Stacked Bars
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10.9, 6.1))
    labels_d3 = [f"{clean_label(r['transition_family'])} [{r['metric_kind']}]" for _, r in t3.iterrows()]
    y_pos = np.arange(len(labels_d3))
    s_code = t3["mean_share_code_pct"].to_numpy(dtype=float)
    s_rule = t3["mean_share_ruleset_pct"].to_numpy(dtype=float)
    s_impl = t3["mean_share_impl_pct"].to_numpy(dtype=float)
    s_tax = t3["mean_share_tax_pct"].to_numpy(dtype=float)

    ax.barh(y_pos, s_code, color="#2b83ba", label=r"$|E_{\mathrm{code}}|$ (Baseline Code Evolution)")
    ax.barh(y_pos, s_rule, left=s_code, color="#fdae61", label=r"$|E_{\mathrm{rule\text{-}set}}|$ (Catalog Rule Add/Drop)")
    ax.barh(y_pos, s_impl, left=s_code + s_rule, color="#d7191c", label=r"$|E_{\mathrm{impl\text{-}drift}}|$ (Shared-Rule Drift)")
    ax.barh(y_pos, s_tax, left=s_code + s_rule + s_impl, color="#abdda4", label=r"$|E_{\mathrm{tax\text{-}remap}}|$ (Taxonomy / Severity Remap)")

    for idx_y in range(len(y_pos)):
        segs = [
            (0.0, s_code[idx_y], "white"),
            (s_code[idx_y], s_rule[idx_y], "black"),
            (s_code[idx_y] + s_rule[idx_y], s_impl[idx_y], "white"),
            (s_code[idx_y] + s_rule[idx_y] + s_impl[idx_y], s_tax[idx_y], "black"),
        ]
        for left_x, w_x, c_txt in segs:
            if w_x >= 9.0:
                ax.text(left_x + w_x / 2.0, idx_y, f"{w_x:.1f}%", ha="center", va="center", color=c_txt, fontsize=7.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_d3)
    ax.invert_yaxis()
    ax.set_xlim(0, 100.5)
    ax.set_xlabel("Active-Repository Mean Share of Stepwise Absolute Magnitude (%)")
    ax.set_title("RQ3: Zero-Residual 4D Accounting Decomposition Across Transitions")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_4d_effect_decomposition.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig3_4d_effect_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # FIGURE 4 (Manuscript Fig. 6): 5-Strategy Track B Calibration Benchmark
    # =========================================================================
    transitions = t4["transition"].unique().tolist()
    strats = [
        ("S0_Naive_Unadjusted", "S0: Naive (0 extra)", "#999999"),
        ("S1_CommonRule_Catalog_Filter", r"S1: Catalog $\mathcal{R}_{\cap}$ (0 extra)", "#e6ab02"),
        ("S2_SingleAnchor_Additive_Offset", "S2: 1-Anchor Offset (1 extra)", "#66a61e"),
        ("S3_SingleAnchor_Safeguarded_Ratio", "S3: 1-Anchor Ratio (1 extra)", "#1f78b4"),
        ("S4_Window3_Moment_Affine", "S4: W=3 Moment-Affine (3 extra)", "#e7298a"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))
    x = np.arange(len(transitions))
    width = 0.15
    short_tr = [
        clean_label(t).replace(" (Extended)", "").replace(" (Pylint-PL Aligned)", " (PL)")
        for t in transitions
    ]

    handles_f4, labels_f4 = [], []
    for idx_s, (scode, slbl, col) in enumerate(strats):
        sub_s = t4[t4["strategy"] == scode].set_index("transition")
        v_cliff = [float(sub_s.loc[t, "boundary_cliff_mae_ksloc"]) if t in sub_s.index else 0.0 for t in transitions]
        v_slope = [float(sub_s.loc[t, "full_slope_mae_vs_historical"]) if t in sub_s.index else 0.0 for t in transitions]
        offset_x = x + (idx_s - 2) * width
        b1 = ax1.bar(offset_x, v_cliff, width=width, label=slbl, color=col)
        ax2.bar(offset_x, v_slope, width=width, label=slbl, color=col)
        handles_f4.append(b1[0])
        labels_f4.append(slbl)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_tr, rotation=20, ha="right")
    ax1.set_ylabel("Boundary Step Cliff MAE vs. Target (High / kSLOC)")
    ax1.set_title(r"(a) Step Cliff Error at First Post-Switch Release $r_{k+1}$")
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_tr, rotation=20, ha="right")
    ax2.set_ylabel("Full-Series Sen's Slope MAE vs. Baseline (/kSLOC/rel)")
    ax2.set_title("(b) Full-Series Slope Continuity vs. Historical Baseline")
    ax2.grid(axis="y", alpha=0.3)

    n_repos_tot = int(t4["n_repos"].max()) if not t4.empty else 49
    fig.legend(handles_f4, labels_f4, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=True, fontsize=8.0)
    fig.suptitle(
        f"RQ4 Track B: Benchmark of 5 Density-Stitching Strategies ({n_repos_tot} Repositories, Mid-Horizon 0.50T)",
        y=1.09,
        fontsize=11.0,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_dual_run_calibration_benchmark.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig4_dual_run_calibration_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved 4 publication figures (PDF+PNG) to {fig_dir}")


if __name__ == "__main__":
    main()
