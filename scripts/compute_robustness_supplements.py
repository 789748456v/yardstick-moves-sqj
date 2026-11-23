#!/usr/bin/env python3
"""
Tier-1 Portable Robustness & Disentanglement Supplement for D04-Plus (SQJ).
Consumes only packaged CSV artifacts in results/full/ (zero external downloads).

Computes:
  1. Joint Hamed-Rao + BH-FDR (q_HR < 0.05) reversals and variance-floor sensitivity (0.10, 0.25, 0.50, 1.00).
  2. Window-vs-model disentanglement: S2 (W=1 Offset) vs S2-W3mean / S2-W3med (W=3 Offset) vs S4 (W=3 Affine),
     paired bootstrap 95% CIs, categorical 3-class MK recovery, non-negative clamping %, and offset drift |d_t - d_k|.
  3. Non-decreasing version-tuple subsequence sensitivity extended to RQ2 reversals and RQ4 S2 slope MAE.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, wilcoxon

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))
RES_DIR = BASE_DIR / "results" / "full"


def theil_sen(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    slopes = [(y[j] - y[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    return float(np.median(slopes)) if slopes else 0.0


def bh_fdr(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = min(prev, ranked[i] * n / (i + 1))
        q[i] = val
        prev = val
    out = np.empty(n)
    out[order] = q
    return out


def hr_mk_pval(y: np.ndarray, floor: float = 0.25) -> float:
    n = len(y)
    if n < 4:
        return 1.0
    x = np.arange(n, dtype=float)
    detrended = y - theil_sen(y) * x
    ranks = pd.Series(detrended).rank(method="average").to_numpy()
    r_mean = ranks.mean()
    denom = float(np.sum((ranks - r_mean) ** 2))
    S = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = y[j] - y[i]
            if diff > 0:
                S += 1.0
            elif diff < 0:
                S -= 1.0
    if S == 0.0:
        return 1.0
    _, counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var0 = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var0 <= 0.0:
        return 1.0
    corr_sum = 0.0
    if denom > 0.0:
        crit = 1.96 / np.sqrt(n)
        for lag in range(1, max(1, n - 2)):
            num = float(np.sum((ranks[:-lag] - r_mean) * (ranks[lag:] - r_mean)))
            rl = num / denom
            if abs(rl) > crit:
                corr_sum += (n - lag) * (n - lag - 1) * (n - lag - 2) * rl
    factor = max(floor, 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * corr_sum)
    var_hr = var0 * factor
    z = (abs(S) - 1.0) / np.sqrt(var_hr)
    return float(2.0 * (1.0 - norm.cdf(z)))


def bootstrap_mean_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diffs)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(diffs[idx])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def parse_ver_tuple(tag: str) -> Optional[Tuple[int, ...]]:
    nums = re.findall(r"\d+", str(tag))
    return tuple(int(x) for x in nums[:4]) if nums else None


def monotone_subseq_indices(tags: List[str]) -> List[int]:
    keep = [0]
    prev = parse_ver_tuple(tags[0])
    for i in range(1, len(tags)):
        cur = parse_ver_tuple(tags[i])
        if prev is None or cur is None or cur >= prev:
            keep.append(i)
            if cur is not None:
                prev = cur
    return keep


def mk_3class(arr: np.ndarray) -> int:
    tau, pv = kendalltau(np.arange(len(arr)), arr)
    if pd.isna(tau) or pv >= 0.05:
        return 0
    return 1 if tau > 0 else -1


def main() -> None:
    df = pd.read_csv(RES_DIR / "full_unified_metrics.csv")
    repos = sorted(df["repository"].unique())

    # 1. Joint HR + BH-FDR (q_HR < 0.05) and variance floor sensitivity
    rq2_pairs = [
        ("SQ Comp: Std->MQR", "sq26_comp_std", "sq26_comp_mqr", "density_rel_sec_ksloc"),
        ("SQ 10.3: Std->CC", "sq103_std", "sq103_cleancode", "density_rel_sec_ksloc"),
        ("SQ 26.11: Comp->Core (tot)", "sq26_comp_mqr", "sq26_core_mqr", "density_total_ksloc"),
        ("SQ 26.11: Comp->Core (high)", "sq26_comp_mqr", "sq26_core_mqr", "density_high_ksloc"),
        ("SQ 10.3->Core MQR (tot)", "sq103_std", "sq26_core_mqr", "density_total_ksloc"),
        ("SQ 10.3->Core MQR (high)", "sq103_std", "sq26_core_mqr", "density_high_ksloc"),
        ("SQ 10.3->Comp MQR (high)", "sq103_std", "sq26_comp_mqr", "density_high_ksloc"),
        ("SQ 10.3->Comp MQR (rel)", "sq103_std", "sq26_comp_mqr", "density_rel_sec_ksloc"),
        ("SQ 9.9->Core Std (high)", "sq99_std", "sq26_core_std", "density_high_ksloc"),
        ("Flake8->Ruff Def (tot)", "flake8_71", "ruff_modern_default", "density_total_ksloc"),
        ("Pylint->Ruff PL (high)", "pylint_33", "ruff_modern_pylint", "density_high_ksloc"),
        ("Pylint->Ruff Ext (tot)", "pylint_33", "ruff_modern_extended", "density_total_ksloc"),
    ]
    hr_rows = []
    for lbl, r0, r1, mcol in rq2_pairs:
        d0 = df[df["regime"] == r0].sort_values(["repository", "seq_idx"])
        d1 = df[df["regime"] == r1].sort_values(["repository", "seq_idx"])
        row = {"transition": lbl, "metric": mcol}
        for fl in [0.10, 0.25, 0.50, 1.00]:
            p0s, p1s, t0s, t1s = [], [], [], []
            for repo in repos:
                y0 = d0[d0["repository"] == repo][mcol].to_numpy(dtype=float)
                y1 = d1[d1["repository"] == repo][mcol].to_numpy(dtype=float)
                t0, _ = kendalltau(np.arange(len(y0)), y0)
                t1, _ = kendalltau(np.arange(len(y1)), y1)
                t0s.append(0.0 if pd.isna(t0) else float(t0))
                t1s.append(0.0 if pd.isna(t1) else float(t1))
                p0s.append(hr_mk_pval(y0, floor=fl))
                p1s.append(hr_mk_pval(y1, floor=fl))
            q0s, q1s = bh_fdr(p0s), bh_fdr(p1s)
            hr_rev = sum(1 for a, b, pa, pb in zip(t0s, t1s, p0s, p1s) if a * b < 0 and pa < 0.05 and pb < 0.05)
            qhr_rev = sum(1 for a, b, qa, qb in zip(t0s, t1s, q0s, q1s) if a * b < 0 and qa < 0.05 and qb < 0.05)
            tag = f"f{int(round(fl * 100)):03d}"
            row[f"hr_{tag}"] = hr_rev
            row[f"qhr_{tag}"] = qhr_rev
        hr_rows.append(row)
    hr_df = pd.DataFrame(hr_rows)
    hr_df.to_csv(RES_DIR / "table_robustness_hr_fdr_and_floors.csv", index=False)

    # 2. W=1 vs W=3 Offset vs W=3 Affine, Clamping, Offset Drift |d_t - d_k|, and 3-Class MK Recovery
    calib_pairs = [
        ("SQ 10.3 -> Core MQR", "sq103_std", "sq26_core_mqr"),
        ("SQ 10.3 -> Comp MQR", "sq103_std", "sq26_comp_mqr"),
        ("SQ 9.9 -> Core MQR", "sq99_std", "sq26_core_mqr"),
        ("Flake8 -> Ruff Ext", "flake8_71", "ruff_modern_extended"),
        ("Pylint -> Ruff PL", "pylint_33", "ruff_modern_pylint"),
    ]
    w_rows = []
    for lbl, r0, r1 in calib_pairs:
        d0 = df[df["regime"] == r0].sort_values(["repository", "seq_idx"])
        d1 = df[df["regime"] == r1].sort_values(["repository", "seq_idx"])
        e_s0, e_s2, e_w3m, e_w3med, e_s4 = [], [], [], [], []
        clamp_cnt, tot_steps = 0, 0
        drifts = {1: [], 3: [], 5: [], 10: []}
        bal_drifts = {1: [], 3: [], 5: [], 10: []}  # Balanced N=32 panel (T=20 repos covering all dt=1..10)
        mk_s0, mk_s2, mk_w3m, mk_s4 = 0, 0, 0, 0
        for repo in repos:
            y0 = d0[d0["repository"] == repo]["density_high_ksloc"].to_numpy(dtype=float)
            y1 = d1[d1["repository"] == repo]["density_high_ksloc"].to_numpy(dtype=float)
            T = len(y0)
            k = int(0.50 * T) - 1
            bs = theil_sen(y0)
            dk = y1[k] - y0[k]
            raw_s2 = y1[k + 1 :] - dk
            clamp_cnt += int(np.sum(raw_s2 < 0.0))
            tot_steps += len(raw_s2)

            yhat0 = np.concatenate([y0[: k + 1], y1[k + 1 :]])
            yhat2 = np.concatenate([y0[: k + 1], np.maximum(0.0, raw_s2)])
            dk3_m = float(np.mean(y1[k - 2 : k + 1] - y0[k - 2 : k + 1]))
            dk3_med = float(np.median(y1[k - 2 : k + 1] - y0[k - 2 : k + 1]))
            yhat_w3m = np.concatenate([y0[: k + 1], np.maximum(0.0, y1[k + 1 :] - dk3_m)])
            yhat_w3med = np.concatenate([y0[: k + 1], np.maximum(0.0, y1[k + 1 :] - dk3_med)])

            w0, w1 = y0[k - 2 : k + 1], y1[k - 2 : k + 1]
            mu0, mu1 = float(np.mean(w0)), float(np.mean(w1))
            sd0, sd1 = float(np.std(w0)), float(np.std(w1))
            if sd0 > 0.02 and sd1 > 0.02:
                beta = float(np.clip(sd0 / sd1, 0.08, 15.0))
            elif mu0 > 0.05 and mu1 > 0.05:
                beta = float(np.clip(mu0 / mu1, 0.08, 15.0))
            else:
                beta = 1.0
            yhat4 = np.concatenate([y0[: k + 1], np.maximum(0.0, mu0 + beta * (y1[k + 1 :] - mu1))])

            e_s0.append(abs(theil_sen(yhat0) - bs))
            e_s2.append(abs(theil_sen(yhat2) - bs))
            e_w3m.append(abs(theil_sen(yhat_w3m) - bs))
            e_w3med.append(abs(theil_sen(yhat_w3med) - bs))
            e_s4.append(abs(theil_sen(yhat4) - bs))

            for dt in drifts:
                if k + dt < T:
                    val_d = abs((y1[k + dt] - y0[k + dt]) - dk)
                    drifts[dt].append(val_d)
                    if T == 20:
                        bal_drifts[dt].append(val_d)

            c_base = mk_3class(y0)
            if mk_3class(yhat0) == c_base:
                mk_s0 += 1
            if mk_3class(yhat2) == c_base:
                mk_s2 += 1
            if mk_3class(yhat_w3m) == c_base:
                mk_w3m += 1
            if mk_3class(yhat4) == c_base:
                mk_s4 += 1

        d_s2_s0, lo_s2_s0, hi_s2_s0 = bootstrap_mean_diff_ci(e_s2, e_s0)
        p_w1_w3m = float(wilcoxon(e_s2, e_w3m).pvalue)
        p_w1_w3med = float(wilcoxon(e_s2, e_w3med).pvalue)
        p_w3m_s4 = float(wilcoxon(e_w3m, e_s4).pvalue)
        w_rows.append(
            {
                "transition": lbl,
                "s0_mae": round(float(np.mean(e_s0)), 4),
                "s2_w1_mae": round(float(np.mean(e_s2)), 4),
                "s2_w3mean_mae": round(float(np.mean(e_w3m)), 4),
                "s2_w3med_mae": round(float(np.mean(e_w3med)), 4),
                "s4_w3affine_mae": round(float(np.mean(e_s4)), 4),
                "p_w1_vs_w3mean": round(p_w1_w3m, 4),
                "p_w1_vs_w3med": round(p_w1_w3med, 4),
                "p_w3mean_vs_s4": round(p_w3m_s4, 4),
                "diff_s2_minus_s0": round(d_s2_s0, 4),
                "ci95_lo_s2_minus_s0": round(lo_s2_s0, 4),
                "ci95_hi_s2_minus_s0": round(hi_s2_s0, 4),
                "clamp_pct": round(100.0 * clamp_cnt / tot_steps, 2),
                "mk3_rec_s0_pct": round(100.0 * mk_s0 / len(repos), 1),
                "mk3_rec_s2_pct": round(100.0 * mk_s2 / len(repos), 1),
                "mk3_rec_w3m_pct": round(100.0 * mk_w3m / len(repos), 1),
                "mk3_rec_s4_pct": round(100.0 * mk_s4 / len(repos), 1),
                "n_dt1": len(drifts[1]),
                "n_dt5": len(drifts[5]),
                "n_dt10": len(drifts[10]),
                "bal_n": len(bal_drifts[10]),
                "bal_med_drift_dt1": round(float(np.median(bal_drifts[1])), 4),
                "bal_med_drift_dt3": round(float(np.median(bal_drifts[3])), 4),
                "bal_med_drift_dt5": round(float(np.median(bal_drifts[5])), 4),
                "bal_med_drift_dt10": round(float(np.median(bal_drifts[10])), 4),
            }
        )
    w_df = pd.DataFrame(w_rows)
    w_df.to_csv(RES_DIR / "table_robustness_w1_vs_w3_and_drift.csv", index=False)

    # 3. Monotone-version subsequence sensitivity for RQ2 and RQ4
    keep_map = {}
    for repo in repos:
        sub = df[(df["repository"] == repo) & (df["regime"] == "sq103_std")].sort_values("seq_idx")
        idxs = monotone_subseq_indices(sub["release_tag"].tolist())
        keep_map[repo] = set(sub["seq_idx"].iloc[idxs].tolist())

    mon_rows = []
    for lbl, r0, r1, mcol in rq2_pairs:
        rev_full, rev_mon = 0, 0
        for repo in repos:
            g0 = df[(df["repository"] == repo) & (df["regime"] == r0)].sort_values("seq_idx")
            g1 = df[(df["repository"] == repo) & (df["regime"] == r1)].sort_values("seq_idx")
            t0, p0 = kendalltau(np.arange(len(g0)), g0[mcol].to_numpy())
            t1, p1 = kendalltau(np.arange(len(g1)), g1[mcol].to_numpy())
            if not pd.isna(t0) and not pd.isna(t1) and t0 * t1 < 0 and p0 < 0.05 and p1 < 0.05:
                rev_full += 1
            km = keep_map[repo]
            g0m = g0[g0["seq_idx"].isin(km)]
            g1m = g1[g1["seq_idx"].isin(km)]
            t0m, p0m = kendalltau(np.arange(len(g0m)), g0m[mcol].to_numpy())
            t1m, p1m = kendalltau(np.arange(len(g1m)), g1m[mcol].to_numpy())
            if not pd.isna(t0m) and not pd.isna(t1m) and t0m * t1m < 0 and p0m < 0.05 and p1m < 0.05:
                rev_mon += 1
        mon_rows.append({"scope": "RQ2_Raw_Rev", "transition": lbl, "full_val": rev_full, "monotone_val": rev_mon})

    for lbl, r0, r1 in calib_pairs:
        e0_f, e2_f, e0_m, e2_m = [], [], [], []
        for repo in repos:
            g0 = df[(df["repository"] == repo) & (df["regime"] == r0)].sort_values("seq_idx")
            g1 = df[(df["repository"] == repo) & (df["regime"] == r1)].sort_values("seq_idx")
            y0, y1 = g0["density_high_ksloc"].to_numpy(dtype=float), g1["density_high_ksloc"].to_numpy(dtype=float)
            k = int(0.50 * len(y0)) - 1
            bs = theil_sen(y0)
            e0_f.append(abs(theil_sen(np.concatenate([y0[: k + 1], y1[k + 1 :]])) - bs))
            e2_f.append(abs(theil_sen(np.concatenate([y0[: k + 1], np.maximum(0.0, y1[k + 1 :] - (y1[k] - y0[k]))])) - bs))

            km = keep_map[repo]
            y0m = g0[g0["seq_idx"].isin(km)]["density_high_ksloc"].to_numpy(dtype=float)
            y1m = g1[g1["seq_idx"].isin(km)]["density_high_ksloc"].to_numpy(dtype=float)
            if len(y0m) >= 6:
                km_idx = int(0.50 * len(y0m)) - 1
                bsm = theil_sen(y0m)
                e0_m.append(abs(theil_sen(np.concatenate([y0m[: km_idx + 1], y1m[km_idx + 1 :]])) - bsm))
                e2_m.append(abs(theil_sen(np.concatenate([y0m[: km_idx + 1], np.maximum(0.0, y1m[km_idx + 1 :] - (y1m[km_idx] - y0m[km_idx]))])) - bsm))
        mon_rows.append(
            {
                "scope": "RQ4_Mid_S2_Reduction_Pct",
                "transition": lbl,
                "full_val": round(100.0 * (np.mean(e0_f) - np.mean(e2_f)) / np.mean(e0_f), 1),
                "monotone_val": round(100.0 * (np.mean(e0_m) - np.mean(e2_m)) / np.mean(e0_m), 1),
            }
        )
    mon_df = pd.DataFrame(mon_rows)
    mon_df.to_csv(RES_DIR / "table_robustness_monotone_rq2_rq4.csv", index=False)

    print("[OK] Exported 3 Tier-1 robustness tables:")
    print("\n--- Joint HR+FDR & Variance Floors ---")
    print(hr_df.to_string(index=False))
    print("\n--- W=1 vs W=3 Offset vs Affine, Drift, Clamping, MK3 Recovery ---")
    print(w_df.to_string(index=False))
    print("\n--- Monotone-Version RQ2 & RQ4 Sensitivity ---")
    print(mon_df.to_string(index=False))


if __name__ == "__main__":
    main()
