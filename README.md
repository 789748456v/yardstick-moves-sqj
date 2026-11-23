# When the Yardstick Moves: Quality Gate Flip Attribution and Dual-Run Transition Calibration under Rule Taxonomy Switching and Static Analyzer Migration

Official replication package for the manuscript submitted to *Software Quality Journal* (Springer Nature).

**Authors:** Zhijie Li¹ · Ketong Liu¹ · Shaoyang Xu¹ · Shibo Meng¹ · Feiran Gao² · Yuli Zhao¹\*  
¹ Software College, Northeastern University, Shenyang 110169, China  
² School of Computer Science and Technology, Beijing Institute of Technology, Beijing 100081, China  
\* Corresponding author: `zhaoyl@swc.neu.edu.cn`

---

## Study Overview

Automated static analysis tools (ASATs) act simultaneously as CI release quality gates and longitudinal measurement instruments in Mining Software Repositories (MSR) studies. This replication package accompanies our large-scale empirical study of measurement non-stationarity across two major transitions in the Python ecosystem:

1. **SonarQube rule taxonomy and profile tier transitions:** `Standard Experience` → `Multi-Quality Rule (MQR)` mode switching, `SonarQube 9.9.8` / `10.3.0` → `26.11.0`, and the built-in `Sonar way` split into `Core` (232 rules) vs. `Comprehensive` (407 rules).
2. **Python CLI linter migrations and upgrades:** `Flake8 7.1.1` / `Pylint 3.3.4` → `Ruff 0.17.5`, plus same-linter upgrades (`Pylint 2.17.7` [Python 3.11] → `3.3.4` [Python 3.12], `Ruff 0.1.15` → `0.17.5`).

Extending the 57-project Python benchmark of Berciu et al. (2026) without imposing a codebase size cap, we re-executed **49 eligible Python 3 repositories** (final SLOC `1,826–971,908`, cumulative `98.52M` SLOC) across **882 consecutive releases** (`833` transitions), **10,584 completed scans** (including `3,528` server-reconciled SonarQube scans), and **88.43M raw alerts** spanning **15 analysis regimes**.

### Key Headline Findings

- **Profile Tiering & Cross-Linter Instability (RQ1–RQ2):** Fresh-default `SonarQube 10.3 → 26.11 Core MQR` bindings deactivate `98/218` (`45.0%`) legacy rules—including `17/17` (`100%`) of Berciu et al.'s flagship Python rules—and flip `36.85%` of native quality gates ($\tau_b = 0.250$). Meanwhile, `Flake8`/`Pylint → Ruff 0.17.5` migrations flip `8.04%–27.13%` of operational gates and invert `8.2%–18.4%` of repository longitudinal trends ($p < 0.05$).
- **Pure `Standard → MQR` Switching (RQ1–RQ2):** Leaves all-new-issue gates invariant (`0.00%` flips) yet triggers `7.08%` unilateral `PASS → FAIL` blockages on Reliability/Security gates across `>60%` of repositories, and reverses the Reliability/Security trend of `huggingface/transformers` under BH-FDR control (`SQ 26.11`: $\tau_{\mathrm{MK}} = +0.43^* \to -0.43^*$, not significant at $\alpha = 0.05$ under Hamed–Rao; `SQ 10.3`: $+0.56^* \to -0.62^*$, surviving Hamed–Rao autocorrelation adjustment and joint $q_{\mathrm{HR}} < 0.05$).
- **Zero-Residual 4D Accounting & 100-Slice Audit (RQ3):** Shared-rule drift (`53.49%–60.49%` on rule-preserving upgrades; `51.74%` non-hotspot detector drift after isolating `16.67%` `SECURITY_HOTSPOT → ISSUE` channel migration under $D_5$ normalization) and taxonomy remapping (up to `85.79%`) dominate stepwise shifts, explaining why static common-rule filtering ($\mathcal{R}_{\cap}$) shows no statistically detectable improvement over naive stitching on drift- or taxonomy-affected transitions ($p \in [0.285, 0.874]$).
- **Single-Anchor Dual-Track Calibration (RQ4):** A single overlapping anchor scan ($W = 1$) guarantees exact count-ratchet continuity by algebraic identity, cuts switch-step legacy alarms by `95.6%–96.5%` via persistent-anchor suppression (while rolling step-fingerprint strict gating achieves `92.4%–97.1%` agreement with regime-specific `New High Gate` references), and reduces mid-horizon historical-baseline Sen's slope MAE by `64.9%–86.1%` ($p \le 10^{-5}$), though categorical trend recovery remains bounded under severe profile contraction.

---

## Manuscript-to-Artifact Crosswalk

Because Manuscript **Fig. 1** is an inline LaTeX/TikZ workflow diagram and several archival tables consolidate multiple analysis outputs, the table below maps each manuscript table and figure directly to its backing artifact file(s) and generator script.

| Manuscript Element | Backing Artifact File(s) in This Repository | Generator Script |
| :--- | :--- | :--- |
| **Table 1** (SonarQube Profile & Taxonomy Audit) | `data/sq_catalogs/*_{meta,rules}.json`, `results/full/execution_audit_and_versions.json` | `scripts/export_archival_evidence.py` |
| **Table 2** (Top-17 Flagship Rules Fate) | `results/full/table1b_n09_top17_rules_fate.csv` | `scripts/enrich_sqj_archival_analysis.py` |
| **Table 3** (CLI Linter Catalogs & Mapping) | `data/linter_catalogs.json`, `results/full/execution_audit_and_versions.json` | `scripts/export_archival_evidence.py` |
| **Table 4** (PRISMA Funnel & SLOC Tiers) | `data/prisma_cohort_funnel.csv`, `data/probe_all_53_stage1_candidates.csv`, `results/full/corpus_telemetry_49repos.json` | `scripts/build_full_cohort_manifest.py` |
| **Table 5** (RQ1 Multi-Gate Flip Rates & $\tau_b$) | `results/full/table1_multigate_flips_and_ranking.csv`, `results/full/table1_semver_monotone_sensitivity.csv` | `scripts/analyze_full_study.py` |
| **Table 6** (RQ1 Cluster Bootstrap 95% CIs & IQR) | `results/full/table1d_cluster_bootstrap_and_iqr.csv` | `scripts/enrich_sqj_archival_analysis.py` |
| **Table 7** (RQ1 Scale-Tier Flip Sensitivity) | `results/full/table1c_scale_stratified_flips.csv` | `scripts/analyze_full_study.py` |
| **Table 8** (RQ2 Trend Reversals: Raw / FDR / HR / $q_{\mathrm{HR}}$) | `results/full/table2_longitudinal_trend_reversals_summary.csv`, `results/full/table2_per_repo_trends.csv`, `results/full/table_robustness_hr_fdr_and_floors.csv` | `scripts/analyze_full_study.py`, `scripts/compute_robustness_supplements.py` |
| **Table 9** (RQ3 Zero-Residual 4D Accounting, $D_4$) | `results/full/table3_4d_effect_decomposition_summary.csv`, `results/full/table3_per_repo_4d.csv`, `results/full/table_mmdetection_truncation_sensitivity.csv` | `scripts/analyze_full_study.py` |
| **Table 10** (RQ3 $D_5$ Hotspot Channel Split) | `results/full/table3b_hotspot_sensitivity.csv` | `scripts/analyze_full_study.py` |
| **Table 11** & **Cases 1–4** (100-Slice Audit) | `results/full/table5_root_cause_taxonomy_summary.csv`, `results/full/table5_qualitative_100_slices.{csv,json}`, `results/full/selected_code_listings.json` | `scripts/analyze_full_study.py`, `scripts/enrich_sqj_archival_analysis.py` |
| **Table 12** (RQ4 Track A CI Calibration) | `results/full/table4a_trackA_fingerprint_ci_calibration.csv` | `scripts/analyze_full_study.py` |
| **Table 13** (RQ4 Track B 5-Strategy Benchmark) | `results/full/table4_dual_run_calibration_5strategies.csv` | `scripts/analyze_full_study.py` |
| **Table 14** (RQ4 Multi-Anchor Switch Horizons) | `results/full/table4b_multi_anchor_sensitivity.csv` | `scripts/analyze_full_study.py` |
| **Table 15** (RQ4 Consolidated Robustness Audit) | `results/full/table_robustness_w1_vs_w3_and_drift.csv`, `results/full/table_robustness_monotone_rq2_rq4.csv` | `scripts/compute_robustness_supplements.py` |
| **Table 16** (CI Governance Playbook) | Synthesized from Tables 5–15 | — |
| **Table A1** (Complete 49-Repo Cohort) | `results/full/table0_cohort_49repos.csv`, `data/full_releases_manifest.{csv,json}` | `scripts/analyze_full_study.py` |
| **Fig. 1** (Study Design & Dual-Track Workflow) | Inline LaTeX/TikZ diagram in manuscript | — |
| **Fig. 2** (Multi-Gate Flip Heatmap) | `results/full/figures/fig1_multigate_flip_heatmap.{pdf,png}` | `scripts/plot_full_figures.py` |
| **Fig. 3** (Longitudinal Hard Reversals) | `results/full/figures/fig2_longitudinal_hard_reversals.{pdf,png}` | `scripts/plot_full_figures.py` |
| **Fig. 4** (4D Accounting Stacked Bars) | `results/full/figures/fig3_4d_effect_decomposition.{pdf,png}` | `scripts/plot_full_figures.py` |
| **Fig. 5** (4D Dispersion & Scale Stratification) | `results/full/figures/fig5_4d_repo_distributions.{pdf,png}` | `scripts/enrich_sqj_archival_analysis.py` |
| **Fig. 6** (Track B Calibration Benchmark) | `results/full/figures/fig4_dual_run_calibration_benchmark.{pdf,png}` | `scripts/plot_full_figures.py` |
| **Fig. 7** (Multi-Anchor & Baseline Retirement) | `results/full/figures/fig6_multi_anchor_and_debt_decay.{pdf,png}` | `scripts/enrich_sqj_archival_analysis.py` |
| **References (60 Entries)** | `manuscript/bibliography_verification_ledger.csv` | — |

> **Artifact Naming Conventions:** To preserve exact backwards compatibility across all scripts and archived CSV schemas, the shorthand `n09` / `N09` appears in several archive filenames and column keys (e.g., `sq103_n09_rules.json`, `table1b_n09_top17_rules_fate.csv`, `figshare_n09/`) to denote the foundational benchmark of Berciu et al. (2026), *Software Quality Journal*, **Article 9**. Likewise, `D04_BASE_DIR` is the optional environment variable that overrides auto-detected repository root paths in `scripts/*.py`.

---

## Repository Structure

```text
yardstick-moves-sqj/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── prisma_cohort_funnel.csv               # Two-stage PRISMA screening ledger (57 -> 49 repos)
│   ├── probe_all_53_stage1_candidates.csv     # Python 3 ast.parse audit across all 53 Stage-1 candidates
│   ├── full_releases_manifest.{csv,json}      # 49 repositories x 882 consecutive releases manifest
│   ├── linter_catalogs.json                   # Exported static enabled-rule catalogs (8 CLI linter regimes)
│   └── sq_catalogs/                           # Exported SonarQube REST API rule catalogs & profile metadata (structural fields; HTML prose omitted)
├── results/
│   └── full/
│       ├── full_unified_metrics.csv           # 13,230-row regime-snapshot metrics table (server-reconciled)
│       ├── execution_audit_and_versions.json  # Binary version locks, scan counts & server reconciliation audit
│       ├── corpus_telemetry_49repos.json      # Corpus SLOC, release span, and transition telemetry
│       ├── table0_cohort_49repos.csv          # Appendix Table A1 cohort metadata
│       ├── table1_*.csv                       # RQ1 multi-gate flips, Top-17 fate, scale strata, cluster CI, monotone check
│       ├── table2_*.csv                       # RQ2 per-repo & summary trend reversals (raw p, BH-FDR q, Hamed-Rao)
│       ├── table3*.csv                        # RQ3 4D accounting decomposition, D5 hotspot sensitivity, per-repo shares
│       ├── table_mmdetection_truncation_sensitivity.csv  # N=48 sensitivity check for 10k shard cap
│       ├── table4*.csv                        # RQ4 Track A fingerprint calibration, Track B 5-strategy & multi-anchor benchmarks
│       ├── table5_*.{csv,json}                # 100 Git-verified code slices & 11-category taxonomy summary
│       ├── selected_code_listings.json        # Verbatim source snippets for manuscript Case 1-4 listings
│       ├── table_robustness_*.csv             # Joint HR+FDR variance-floor, W=1 vs W=3 drift, monotone RQ2/RQ4 checks
│       └── figures/                           # Vector PDF & high-DPI PNG figures (fig1-fig6)
├── scripts/
│   ├── plot_full_figures.py                   # [Tier 1] Regenerate manuscript empirical figures (Fig. 2-4, Fig. 6)
│   ├── enrich_sqj_archival_analysis.py        # [Tier 1] Regenerate cluster CIs, Top-17 fate, Fig. 5 & Fig. 7, code listings
│   ├── compute_robustness_supplements.py      # [Tier 1] Regenerate HR+FDR floor, W=1 vs W=3 drift & monotone checks
│   ├── analyze_full_study.py                  # [Tier 2] Streaming RQ1-RQ4 statistical engine & 100-slice sampler
│   ├── export_archival_evidence.py            # [Tier 3] Export static CLI/SQ catalogs & run server reconciliation audit
│   ├── build_full_cohort_manifest.py          # [Tier 3] Reproduce PRISMA funnel & 53-repo ast.parse screening
│   ├── git_diff_newcode.py                    # [Tier 4] Release-to-release git diff -U0 new-code line mapper
│   ├── harmonize_sympy_rubi_exclusion.py      # [Tier 4] Harmonize SymPy generated rubi_tests_web exclusion
│   ├── sonarqube_client.py                    # [Tier 4] Portable Docker SonarQube lifecycle & REST extraction client
│   ├── run_linters_full.py                    # [Tier 4] Isolated 8-configuration Python CLI linter runner
│   └── run_sonarqube_full.py                  # [Tier 4] Sequential SonarQube 9.9 / 10.3 / 26.11 scan orchestrator
└── manuscript/
    └── bibliography_verification_ledger.csv   # 60-entry publisher/DOI/arXiv bibliography verification ledger
```

---

## Reproduction Guide (Four Tiers)

All Python scripts automatically locate the repository root relative to `__file__` (or honor an optional `D04_BASE_DIR` environment variable) and contain zero hardcoded host paths.

### Tier 1 — Immediate Verification of Figures & Archival Robustness Tables (Zero External Downloads)

Requires only Python 3.10+ and the packages in `requirements.txt`. Runs directly against the structured CSV/JSON artifacts included in this repository:

```bash
pip install -r requirements.txt

python scripts/plot_full_figures.py
python scripts/enrich_sqj_archival_analysis.py
python scripts/compute_robustness_supplements.py
```

This regenerates all six empirical vector/PNG figures (`results/full/figures/fig1`–`fig6`), repository-cluster bootstrap confidence intervals, Top-17 rule fates, joint Hamed–Rao + BH-FDR variance-floor checks, $W=1$ vs. $W=3$ anchor drift sensitivity tables, and monotone version-order checks.

### Tier 2 — Full Streaming RQ1–RQ4 Recomputation & 100-Slice Audit

Recomputing `full_unified_metrics.csv`, Track A fingerprint matching, 4D per-alert decomposition, and the 100-slice Git source audit from raw alerts requires mounting the `19.7 GB` raw per-alert JSON archive (`data/sq_runs/`, `data/linter_runs/`, `data/full_newcode_lines.json`) and local Git checkouts (`repos/`):

```bash
python scripts/analyze_full_study.py
```

Because the raw per-alert JSON archive is `19.7 GB`, it is not hosted on GitHub and is available from the corresponding author (`zhaoyl@swc.neu.edu.cn`) upon reasonable request.

### Tier 3 — Static Catalog Re-Export & PRISMA Cohort Screening

Re-running `export_archival_evidence.py` and `build_full_cohort_manifest.py` requires:
- The three source files from Berciu et al. (2024) (`https://doi.org/10.6084/m9.figshare.26999296`) placed under `figshare_n09/`:
  - `figshare_n09/sonarqube_software_quality_categories_metrics.csv`
  - `figshare_n09/repositories_general_data.xlsx`
  - `figshare_n09/spearman_correlation_results.csv`
- Isolated pinned CLI virtual environments under `envs/`:
  - `envs/flake8_71`: `flake8==7.1.1`
  - `envs/pylint_217`: Python 3.11 + `pylint==2.17.7` (`astroid==2.15.8`)
  - `envs/pylint_33`: Python 3.12 + `pylint==3.3.4` (`astroid==3.3.11`)
  - `envs/ruff_01`: `ruff==0.1.15`
  - `envs/ruff_modern`: `ruff==0.17.5`

```bash
python scripts/export_archival_evidence.py
python scripts/build_full_cohort_manifest.py
```

### Tier 4 — End-to-End Physical Re-Execution (10,584 Scans)

Re-running all `10,584` physical scans from scratch requires Docker (`sonarqube:9.9.8-community`, `sonarqube:10.3.0-community`, `sonarqube:26.11.0.134175-community`), `sonar-scanner-cli`, Git, and the pinned virtual environments listed in Tier 3:

```bash
python scripts/run_linters_full.py
python scripts/run_sonarqube_full.py
```

Exact container digests, tool versions, CLI batching policy, and server-measure reconciliation checks are archived in `results/full/execution_audit_and_versions.json`.

---

## Citation

If you use this replication package, dataset, or Dual-Track calibration scripts in your work, please cite:

```bibtex
@article{li2026yardstick,
  title   = {When the Yardstick Moves: Quality Gate Flip Attribution and Dual-Run Transition Calibration under Rule Taxonomy Switching and Static Analyzer Migration},
  author  = {Li, Zhijie and Liu, Ketong and Xu, Shaoyang and Meng, Shibo and Gao, Feiran and Zhao, Yuli},
  journal = {Software Quality Journal},
  year    = {2026},
  note    = {Under review. Replication package: \url{https://github.com/789748456v/yardstick-moves-sqj}}
}
```

And please also cite the foundational benchmark dataset we extend:

```bibtex
@article{berciu2026longterm,
  title   = {A long-term exploratory study of source code quality issues in open-source Python projects},
  author  = {Berciu, L.-M. and Motogna, S.-C. and Molnar, A.-J.},
  journal = {Software Quality Journal},
  volume  = {34},
  number  = {1},
  pages   = {9},
  year    = {2026},
  doi     = {10.1007/s11219-026-09740-z}
}
```

---

## License

- **Code (`scripts/`):** Licensed under the [MIT License](LICENSE).
- **Structured Data, Tables, and Figures (`data/`, `results/`, `manuscript/`):** Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
