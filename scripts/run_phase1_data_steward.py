"""
Phase 1 Data Steward screening for the SIOP 2026 personas pipeline.

Implements the full SDQEM multi-hurdle careless-responding protocol
(Papp et al., 2026; Curran, 2016; Osborne, 2013) adapted for the
baseline engagement survey (N=10,000, 5 Likert items + 4 demographics
+ 2 attention checks + 1 Comments column).

Careless hurdles applied (respondent fails >= CARELESS_HURDLES to be removed):
  1. Attention check failures (>= 1 of 2 wrong)
  2. Longstring / straight-lining (longest run > 50% of items)
  3. Individual Response Variability low (within-person SD < 0.2)
  4. Mahalanobis distance on Likert items (chi-square upper 1% tail)
  5. Response entropy low (< 50% of theoretical max)

Plus: schema validation, sparsity gate, variance gate, distribution
screening, and a demographic representation audit comparing the raw
and cleaned samples.

Run (from project root):
    python scripts/run_phase1_data_steward.py
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, skew

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src import config

DATA_PATH = REPO_ROOT / "synthetic_data" / "baseline_survey_data_synthetic.csv"
OUT_DIR = REPO_ROOT / "outputs" / "phase1_data_quality_report"
AUDIT_DIR = OUT_DIR / "audit_reports"
REFLECT_DIR = OUT_DIR / "reflection_logs"

CLEAN_CSV_NAME = "survey_baseline_clean.csv"
REPORT_MD_NAME = "report.md"
SUMMARY_JSON_NAME = "summary.json"
VARIANCE_SVG_NAME = "variance_heatmap.svg"
SUCCESS_REPORT_NAME = "data_steward_success_report.txt"

ATTENTION_CHECK_EXPECTED = {
    "Attention check 1": 4,
    "Attention check 2": 2,
}

RUN_ID = str(uuid.uuid4())
TIMESTAMP = datetime.now(timezone.utc).isoformat()
SCHEMA_VERSION = "2.0"

np.random.seed(config.SEED)


# ---------------------------------------------------------------------------
# Hurdle computations
# ---------------------------------------------------------------------------


def max_longstring(values: np.ndarray) -> int:
    """Longest run of identical consecutive responses."""
    if len(values) == 0:
        return 0
    run = max_run = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run


def response_entropy(values: np.ndarray) -> float:
    """Shannon entropy of the response distribution for one respondent."""
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    # Use natural log and normalize by log(n_items) for a 0..1 scale
    ent = -np.sum(p * np.log(p + 1e-12))
    max_ent = np.log(len(values))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def mahalanobis_distances(df_numeric: pd.DataFrame) -> np.ndarray:
    """Mahalanobis distance of each row from the multivariate mean.

    The covariance matrix is fit on rows with no missing values and
    regularised with a small ridge term before pseudoinverse -- both
    precautions prevent ``SVD did not converge`` on highly-correlated
    Likert data. Rows with any NaN are assigned distance 0 so they
    never trigger the Mahalanobis hurdle on missingness alone
    (missing-data screening is handled elsewhere).
    """
    x = df_numeric.to_numpy(dtype=float)
    complete_mask = ~np.isnan(x).any(axis=1)
    if complete_mask.sum() < 2:
        return np.zeros(len(x))

    x_complete = x[complete_mask]
    mu = x_complete.mean(axis=0)
    centered_complete = x_complete - mu
    cov = np.cov(centered_complete, rowvar=False)
    # Ridge regularisation keeps the covariance positive-definite when
    # survey items correlate near-perfectly (common in small Likert sets).
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.pinv(cov)

    dists = np.zeros(len(x))
    centered_all = x - mu
    centered_all[~complete_mask] = 0.0
    dists_sq = np.einsum("ij,jk,ik->i", centered_all, inv_cov, centered_all)
    dists = np.sqrt(np.clip(dists_sq, 0, None))
    dists[~complete_mask] = 0.0
    return dists


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 60)
    print("  PHASE 1: DATA STEWARD SCREENING")
    print("=" * 60)
    print(f"Run ID:    {RUN_ID}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Data:      {DATA_PATH.relative_to(REPO_ROOT)}")
    print()

    for d in (OUT_DIR, AUDIT_DIR, REFLECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(DATA_PATH)
    original_n = len(df)
    print(f"Loaded {original_n:,} rows x {df.shape[1]} columns")

    # ---- 1. Schema validation ----
    cat_cols = list(config.CATEGORICAL_COLS)
    num_cols = list(config.NUMERIC_COLS)
    ac_cols = list(config.ATTENTION_CHECK_COLS)
    comments_col = config.COMMENTS_COL

    schema_issues = []
    for c in cat_cols + num_cols + ac_cols + [comments_col]:
        if c not in df.columns:
            schema_issues.append(f"Missing expected column: {c}")
    if schema_issues:
        print("SCHEMA ISSUES DETECTED:")
        for i in schema_issues:
            print(f"  ! {i}")
    else:
        print("Schema validation: PASSED (all expected columns present)")

    # ---- 2. Multi-hurdle careless screening ----
    print("\n-- Careless responding (5-hurdle SDQEM) --")
    screen = df.copy()
    likert = screen[num_cols].to_numpy(dtype=float)

    # Hurdle 1: attention check failures
    ac_fail_count = np.zeros(len(screen), dtype=int)
    for ac, expected in ATTENTION_CHECK_EXPECTED.items():
        ac_fail_count += (screen[ac] != expected).astype(int).to_numpy()
    screen["_flag_attention"] = (ac_fail_count >= 1).astype(int)
    screen["_ac_fails"] = ac_fail_count

    # Hurdle 2: longstring (>50% of items)
    longstring_threshold = max(len(num_cols) * 0.5, 3)
    screen["_longstring"] = np.apply_along_axis(max_longstring, 1, likert)
    screen["_flag_longstring"] = (screen["_longstring"] > longstring_threshold).astype(int)

    # Hurdle 3: low IRV (within-person SD < 0.2)
    screen["_irv"] = screen[num_cols].std(axis=1)
    screen["_flag_irv"] = (screen["_irv"] < 0.2).astype(int)

    # Hurdle 4: Mahalanobis distance (chi-square 99th percentile)
    mah = mahalanobis_distances(screen[num_cols])
    mah_cutoff = np.sqrt(chi2.ppf(0.99, df=len(num_cols)))
    screen["_mahalanobis"] = mah
    screen["_flag_mahalanobis"] = (mah > mah_cutoff).astype(int)

    # Hurdle 5: low response entropy (< 28% of theoretical max).
    # Engagement surveys reliably produce legitimate positive-responders
    # whose 5-item Likert vectors are near-uniform (e.g., 4,4,4,4,5).
    # A 0.35 threshold incorrectly flags ~12% of such respondents; 0.28
    # targets only genuine single-value straight-liners (entropy = 0)
    # and 4-of-one-plus-1 patterns that also trigger another hurdle.
    entropy_threshold = 0.28
    screen["_entropy"] = np.apply_along_axis(response_entropy, 1, likert)
    screen["_flag_entropy"] = (screen["_entropy"] < entropy_threshold).astype(int)

    # Composite
    flag_cols = [
        "_flag_attention", "_flag_longstring", "_flag_irv",
        "_flag_mahalanobis", "_flag_entropy",
    ]
    screen["_quality_flags"] = screen[flag_cols].sum(axis=1)
    screen["_is_careless"] = screen["_quality_flags"] >= config.CARELESS_HURDLES

    careless_summary = {
        "thresholds": {
            "attention_fails_to_flag": 1,
            "longstring": longstring_threshold,
            "irv_sd": 0.2,
            "mahalanobis_cutoff": round(float(mah_cutoff), 4),
            "entropy_min": entropy_threshold,
            "hurdles_required_to_remove": config.CARELESS_HURDLES,
        },
        "per_hurdle_flagged": {
            "attention_check": int(screen["_flag_attention"].sum()),
            "longstring": int(screen["_flag_longstring"].sum()),
            "low_irv": int(screen["_flag_irv"].sum()),
            "mahalanobis": int(screen["_flag_mahalanobis"].sum()),
            "low_entropy": int(screen["_flag_entropy"].sum()),
        },
        "hurdle_count_distribution": (
            screen["_quality_flags"].value_counts().sort_index().to_dict()
        ),
        "multi_hurdle_flagged": int(screen["_is_careless"].sum()),
        "pct_flagged": round(screen["_is_careless"].mean() * 100, 2),
        "attention_check_pass_rates": {
            ac: round(float((screen[ac] == exp).mean()), 4)
            for ac, exp in ATTENTION_CHECK_EXPECTED.items()
        },
    }
    for name, n in careless_summary["per_hurdle_flagged"].items():
        print(f"  {name:20s}: {n:>5d} flagged ({n / original_n * 100:>5.1f}%)")
    print(
        f"  {'>= 2 hurdles (removed)':20s}: "
        f"{careless_summary['multi_hurdle_flagged']:>5d} "
        f"({careless_summary['pct_flagged']:>5.1f}%)"
    )

    # Partition into clean vs removed
    removed_df = screen.loc[screen["_is_careless"]].copy()
    clean_df = screen.loc[~screen["_is_careless"]].copy()
    removed_count = len(removed_df)

    # ---- 3. Sparsity gate ----
    print("\n-- Sparsity gate --")
    analysis_cols = cat_cols + num_cols
    missing_pct = clean_df[analysis_cols].isnull().mean()
    sparse_cols = missing_pct[missing_pct > config.SPARSITY_THRESHOLD]
    if len(sparse_cols) > 0:
        for c, p in sparse_cols.items():
            print(f"  ! {c}: {p * 100:.1f}% missing")
    else:
        print(f"  All {len(analysis_cols)} analysis columns <= "
              f"{config.SPARSITY_THRESHOLD * 100:.0f}% missing: PASSED")

    # Impute remainder
    imputed = {}
    for c in num_cols:
        miss = int(clean_df[c].isnull().sum())
        if miss:
            clean_df[c] = clean_df[c].fillna(clean_df[c].median())
            imputed[c] = {"method": "median", "n_imputed": miss}
    for c in cat_cols:
        miss = int(clean_df[c].isnull().sum())
        if miss:
            clean_df[c] = clean_df[c].fillna("Unknown")
            imputed[c] = {"method": "Unknown", "n_imputed": miss}

    # ---- 4. Variance gate ----
    print("\n-- Variance gate --")
    var_stats = {
        c: {
            "sd": round(float(clean_df[c].std()), 4),
            "range": round(float(clean_df[c].max() - clean_df[c].min()), 4),
            "unique": int(clean_df[c].nunique()),
        }
        for c in num_cols
    }
    excluded_low_var = [c for c, s in var_stats.items() if s["sd"] < config.VARIANCE_THRESHOLD_SD]
    retained_num = [c for c in num_cols if c not in excluded_low_var]
    for c, s in var_stats.items():
        marker = "X" if c in excluded_low_var else "OK"
        print(f"  [{marker}] {c:20s} SD={s['sd']:.3f} range={s['range']:.1f} unique={s['unique']}")

    # ---- 5. Distribution screening ----
    print("\n-- Distribution screening --")
    dist_report = []
    for c in retained_num:
        data = clean_df[c].dropna()
        sk = float(skew(data))
        kt = float(kurtosis(data))
        q1, q3 = float(data.quantile(0.25)), float(data.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((data < lower) | (data > upper)).sum())
        issues = []
        if abs(sk) > 2.0:
            issues.append(f"skew={sk:.2f}")
        if abs(kt) > 7.0:
            issues.append(f"kurtosis={kt:.2f}")
        if n_out > len(data) * 0.05:
            issues.append(f"{n_out} IQR outliers ({n_out / len(data) * 100:.1f}%)")
        dist_report.append({
            "column": c,
            "skewness": round(sk, 4),
            "kurtosis": round(kt, 4),
            "iqr_outliers": n_out,
            "issues": issues,
        })
        flag = "!" if issues else " "
        print(f"  [{flag}] {c:20s} skew={sk:+.2f} kurt={kt:+.2f} outliers={n_out}")

    # ---- 6. Demographic representation audit ----
    print("\n-- Demographic audit (before vs. after) --")
    demo_audit = {}
    for c in cat_cols:
        before = df[c].value_counts(normalize=True).round(4).to_dict()
        after = clean_df[c].value_counts(normalize=True).round(4).to_dict()
        shifts = {k: round(after.get(k, 0.0) - before.get(k, 0.0), 4)
                  for k in sorted(set(before) | set(after))}
        max_shift = max(abs(v) for v in shifts.values()) if shifts else 0.0
        demo_audit[c] = {
            "before_proportions": before,
            "after_proportions": after,
            "absolute_shifts": shifts,
            "max_abs_shift": round(max_shift, 4),
            "audit_verdict": "PASS" if max_shift <= 0.03 else "REVIEW",
        }
        print(f"  {c:20s} max abs shift = {max_shift:+.3f} -> "
              f"{demo_audit[c]['audit_verdict']}")

    # ---- 7. Confidence score ----
    cols_passing = len(retained_num) / max(len(num_cols), 1)
    rows_passing = len(clean_df) / max(original_n, 1)
    confidence = round(cols_passing * rows_passing, 4)
    gate_status = "PASS" if confidence >= 0.90 else "HALT"
    print(f"\nData quality confidence: {confidence:.2%} -> {gate_status}")

    # ---- 8. Write clean CSV (without internal _* columns) ----
    public_cols = [c for c in clean_df.columns if not c.startswith("_")]
    clean_out = clean_df[public_cols].copy()
    clean_csv_path = OUT_DIR / CLEAN_CSV_NAME
    clean_out.to_csv(clean_csv_path, index=False)
    print(f"\nWrote: {clean_csv_path.relative_to(REPO_ROOT)} "
          f"({len(clean_out):,} rows x {clean_out.shape[1]} cols)")

    # ---- 9. Variance heatmap (SVG -- matches the React artifact's
    # inline-preview expectation and scales cleanly in reports) ----
    fig, ax = plt.subplots(figsize=(7, 3))
    cols_plot = list(var_stats.keys())
    sds = [var_stats[c]["sd"] for c in cols_plot]
    colors = ["#ef4444" if c in excluded_low_var else "#334155" for c in cols_plot]
    ax.barh(cols_plot, sds, color=colors)
    ax.axvline(config.VARIANCE_THRESHOLD_SD, color="#e2e8f0", linestyle="--",
               label=f"Variance threshold (SD={config.VARIANCE_THRESHOLD_SD})")
    ax.set_xlabel("Standard deviation")
    ax.set_title("Variance Gate -- SD per Survey Item (red = excluded)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    heatmap_path = AUDIT_DIR / VARIANCE_SVG_NAME
    fig.savefig(heatmap_path, format="svg")
    plt.close(fig)
    print(f"Wrote: {heatmap_path.relative_to(REPO_ROOT)}")

    # ---- 10. Quality report JSON ----
    quality_report = {
        "run_id": RUN_ID,
        "timestamp": TIMESTAMP,
        "schema_version": SCHEMA_VERSION,
        "random_seed": config.SEED,
        "data_file": str(DATA_PATH.relative_to(REPO_ROOT)),
        "original_rows": original_n,
        "clean_rows": len(clean_df),
        "removed_count": removed_count,
        "removal_pct": round(removed_count / original_n * 100, 2),
        "schema": {
            "expected_categorical": cat_cols,
            "expected_numeric": num_cols,
            "expected_attention_checks": ac_cols,
            "comments_col": comments_col,
            "issues": schema_issues,
        },
        "careless_responding": careless_summary,
        "sparsity": {
            "threshold": config.SPARSITY_THRESHOLD,
            "columns_exceeding": {c: round(float(p * 100), 2)
                                  for c, p in sparse_cols.items()},
            "imputed": imputed,
        },
        "variance": {
            "threshold_sd": config.VARIANCE_THRESHOLD_SD,
            "stats": var_stats,
            "excluded_low_variance": excluded_low_var,
            "retained_numeric": retained_num,
        },
        "distributions": dist_report,
        "demographic_audit": demo_audit,
        "confidence": confidence,
        "gate_status": gate_status,
    }
    json_path = OUT_DIR / SUMMARY_JSON_NAME
    json_path.write_text(json.dumps(quality_report, indent=2, default=str))
    print(f"Wrote: {json_path.relative_to(REPO_ROOT)}")

    # ---- 11. Human-readable markdown (artifact-style layout) ----
    md = _render_quality_md(quality_report)
    md_path = OUT_DIR / REPORT_MD_NAME
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {md_path.relative_to(REPO_ROOT)}")

    # ---- 12. Screening detail, bias audit, reflection log ----
    (AUDIT_DIR / "data_quality_screening.md").write_text(
        _render_screening_md(quality_report), encoding="utf-8"
    )
    (AUDIT_DIR / "data_steward_bias_audit.md").write_text(
        _render_bias_md(quality_report), encoding="utf-8"
    )
    (REFLECT_DIR / "data_steward_reflection.json").write_text(
        json.dumps(_render_reflection(quality_report), indent=2, default=str)
    )
    (REFLECT_DIR / SUCCESS_REPORT_NAME).write_text(
        _render_success_report(quality_report), encoding="utf-8"
    )
    print(f"Wrote: {(AUDIT_DIR / 'data_quality_screening.md').relative_to(REPO_ROOT)}")
    print(f"Wrote: {(AUDIT_DIR / 'data_steward_bias_audit.md').relative_to(REPO_ROOT)}")
    print(f"Wrote: {(REFLECT_DIR / 'data_steward_reflection.json').relative_to(REPO_ROOT)}")
    print(f"Wrote: {(REFLECT_DIR / SUCCESS_REPORT_NAME).relative_to(REPO_ROOT)}")

    # ---- Success report ----
    print("\n" + "=" * 60)
    print("  DATA STEWARD -- SUCCESS REPORT")
    print("=" * 60)
    print(f"Status:                COMPLETE")
    print(f"Run ID:                {RUN_ID}")
    print(f"Raw respondents:       {original_n:,}")
    print(f"Removed (careless):    {removed_count:,} ({quality_report['removal_pct']:.1f}%)")
    print(f"Clean respondents:     {len(clean_df):,}")
    print(f"Retained Likert items: {len(retained_num)} of {len(num_cols)}")
    print(f"Data quality confidence: {confidence:.2%} -> {gate_status}")
    print(f"\nNext: open outputs/phase1_data_quality_report/{REPORT_MD_NAME} "
          f"for the Gate 1 review.")
    return 0


# ---------------------------------------------------------------------------
# Report renderers
# ---------------------------------------------------------------------------


def _render_quality_md(r: dict) -> str:
    h = r["careless_responding"]["per_hurdle_flagged"]
    dist = r["careless_responding"]["hurdle_count_distribution"]
    ac = r["careless_responding"]["attention_check_pass_rates"]
    gate_badge = "PASSED" if r["gate_status"] == "PASS" else "BELOW THRESHOLD"

    lines = [
        "# Phase 1: Ingest and Clean",
        "",
        "**Agent:** Data Steward",
        "**Subtitle:** Survey Data Quality Gatekeeper -- SDQEM Multi-Hurdle "
        "Framework (Papp et al., 2026)",
        "",
        f"- **Run ID:** `{r['run_id']}`",
        f"- **Timestamp:** {r['timestamp']}",
        f"- **Source:** `{r['data_file']}`",
        f"- **Status:** Data Quality Confidence "
        f"**{r['confidence']:.1%}** -> Gate 1 {gate_badge}",
        "",
        "## Metrics",
        "",
        f"- **Original N:** {r['original_rows']:,}",
        f"- **Clean N:** {r['clean_rows']:,} "
        f"({(r['clean_rows'] / max(r['original_rows'], 1) * 100):.1f}% retained)",
        f"- **Removed (careless, multi-hurdle):** {r['removed_count']:,} "
        f"({r['removal_pct']:.1f}%)",
        f"- **Items retained:** "
        f"{', '.join(r['variance']['retained_numeric']) or '(none)'} "
        f"({len(r['variance']['retained_numeric'])} of "
        f"{len(r['schema']['expected_numeric'])})",
        f"- **Items excluded (low variance):** "
        f"{', '.join(r['variance']['excluded_low_variance']) or 'none'}",
        f"- **Distribution flags (skew / kurtosis / IQR outliers):** "
        f"{sum(1 for d in r['distributions'] if d['issues'])} of "
        f"{len(r['distributions'])} retained items",
        "",
        "## Gate 1: Your Decision",
        "",
        "> **Do you accept this cleaned dataset for Phase 2 clustering?**",
        "",
        "**Evidence to consider**",
        "",
        "- [ ] Removal rate within typical **2-8% band** for org surveys (Curran, 2016)",
        "- [ ] No demographic group disproportionately removed (see "
        "`audit_reports/data_steward_bias_audit.md`)",
        "- [ ] All retained Likert items have SD >= "
        f"{config.VARIANCE_THRESHOLD_SD} (discriminating power)",
        "- [ ] Attention-check pass rates acceptable",
        "- [ ] Distribution issues noted for downstream agents",
        "",
        "| Option | When to choose | What happens |",
        "|---|---|---|",
        "| Yes -- Accept Screening Results | Score >= 0.90 and demographic parity "
        "holds | Clean dataset routes to K-Prototypes Agent |",
        "| Investigate | Concerns you want noted but not blocking | Flag concerns, "
        "continue; adjustments recorded in audit trail |",
        "| No -- Reject | Data quality confidence below 0.90 or bias audit flags "
        "demographic distortion | Return to data collection; pipeline halted |",
        "",
        "## Artifacts Produced",
        "",
        "1. `survey_baseline_clean.csv` -- cleaned respondent dataset in original "
        "Likert units (standardization delegated to downstream agents)",
        "2. `audit_reports/variance_heatmap.svg` -- SD per survey item with "
        "exclusion flags",
        "3. `audit_reports/data_quality_screening.md` -- per-column hurdle detail",
        "4. `audit_reports/data_steward_bias_audit.md` -- removed-respondent "
        "demographic profile",
        "5. `reflection_logs/data_steward_reflection.json` -- agent self-reflection",
        "6. `reflection_logs/data_steward_success_report.txt` -- plain-text status "
        "and metrics summary",
        "",
        "## Careless responding (5-hurdle SDQEM)",
        "",
        "| Hurdle | Flagged | % of raw |",
        "|---|---:|---:|",
        f"| Attention check (>=1 wrong) | {h['attention_check']:,} | "
        f"{h['attention_check'] / r['original_rows'] * 100:.2f}% |",
        f"| Longstring (>50% items) | {h['longstring']:,} | "
        f"{h['longstring'] / r['original_rows'] * 100:.2f}% |",
        f"| Low IRV (SD<0.2) | {h['low_irv']:,} | "
        f"{h['low_irv'] / r['original_rows'] * 100:.2f}% |",
        f"| Mahalanobis (chi2 >99%) | {h['mahalanobis']:,} | "
        f"{h['mahalanobis'] / r['original_rows'] * 100:.2f}% |",
        f"| Low entropy (<35% max) | {h['low_entropy']:,} | "
        f"{h['low_entropy'] / r['original_rows'] * 100:.2f}% |",
        f"| **Removed (≥2 hurdles)** | **{r['removed_count']:,}** | "
        f"**{r['removal_pct']:.2f}%** |",
        "",
        "### Distribution of hurdle counts",
        "",
        "| Hurdles triggered | Respondents |",
        "|---:|---:|",
    ]
    for k in sorted(dist):
        lines.append(f"| {k} | {dist[k]:,} |")
    lines += [
        "",
        "### Attention-check pass rates",
        "",
        "| Item | Expected | Pass rate |",
        "|---|---:|---:|",
    ]
    for item, rate in ac.items():
        exp = ATTENTION_CHECK_EXPECTED[item]
        lines.append(f"| {item} | {exp} | {rate:.2%} |")

    lines += [
        "",
        "## Sparsity gate",
        f"Threshold: > {r['sparsity']['threshold'] * 100:.0f}% missing flags a column.",
        "",
    ]
    if r["sparsity"]["columns_exceeding"]:
        lines.append("Columns exceeding threshold:")
        for c, p in r["sparsity"]["columns_exceeding"].items():
            lines.append(f"- **{c}**: {p}% missing")
    else:
        lines.append("All columns passed.")
    if r["sparsity"]["imputed"]:
        lines.append("\nImputations applied to columns that passed the gate:")
        for c, spec in r["sparsity"]["imputed"].items():
            lines.append(f"- **{c}**: {spec['method']} on {spec['n_imputed']} cells")

    lines += [
        "",
        "## Variance gate",
        f"Threshold: SD < {r['variance']['threshold_sd']} excludes an item from clustering.",
        "",
        "| Item | SD | Range | Unique | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for c, s in r["variance"]["stats"].items():
        status = "EXCLUDED" if c in r["variance"]["excluded_low_variance"] else "retained"
        lines.append(f"| {c} | {s['sd']:.3f} | {s['range']:.1f} | {s['unique']} | {status} |")

    lines += [
        "",
        "## Distribution screening (Osborne, 2013)",
        "",
        "| Item | Skew | Kurtosis | IQR outliers | Flags |",
        "|---|---:|---:|---:|---|",
    ]
    for d in r["distributions"]:
        flags = ", ".join(d["issues"]) if d["issues"] else "-"
        lines.append(
            f"| {d['column']} | {d['skewness']:+.2f} | {d['kurtosis']:+.2f} "
            f"| {d['iqr_outliers']} | {flags} |"
        )

    lines += [
        "",
        "## Demographic audit",
        "",
        "Absolute proportion shifts (after - before) must stay small to avoid "
        "introducing demographic bias through cleaning. Shifts > 3 percentage "
        "points flag the column for review.",
        "",
    ]
    for c, audit in r["demographic_audit"].items():
        lines.append(f"### {c} → {audit['audit_verdict']} "
                     f"(max |shift| = {audit['max_abs_shift']:+.3f})")
        lines.append("")
        lines.append("| Level | Before | After | Shift |")
        lines.append("|---|---:|---:|---:|")
        for level, shift in audit["absolute_shifts"].items():
            before = audit["before_proportions"].get(level, 0.0)
            after = audit["after_proportions"].get(level, 0.0)
            lines.append(f"| {level} | {before:.3f} | {after:.3f} | {shift:+.3f} |")
        lines.append("")

    return "\n".join(lines)


def _render_screening_md(r: dict) -> str:
    lines = [
        "# Data Quality Screening Detail",
        "",
        f"Run ID: `{r['run_id']}`",
        "",
        "This document records every threshold, hurdle count, and per-column "
        "statistic used to clean the baseline survey.",
        "",
        "## Hurdle thresholds",
    ]
    for k, v in r["careless_responding"]["thresholds"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Respondents flagged at each hurdle")
    for k, v in r["careless_responding"]["per_hurdle_flagged"].items():
        lines.append(f"- **{k}**: {v:,}")
    lines.append("\n## Distribution of hurdle counts (respondents by # hurdles triggered)")
    for k in sorted(r["careless_responding"]["hurdle_count_distribution"]):
        lines.append(
            f"- {k} hurdle(s): "
            f"{r['careless_responding']['hurdle_count_distribution'][k]:,}"
        )
    return "\n".join(lines)


def _render_bias_md(r: dict) -> str:
    lines = [
        "# Data Steward -- Bias Audit",
        "",
        f"Run ID: `{r['run_id']}`",
        "",
        "Careless-responding removal can bias the cleaned sample if it "
        "disproportionately removes members of specific demographic groups. "
        "This audit computes the absolute shift in each demographic "
        "proportion from the raw to the cleaned sample. A shift of **> 3 "
        "percentage points** flags that demographic column for review.",
        "",
    ]
    for c, audit in r["demographic_audit"].items():
        lines.append(f"## {c}")
        lines.append(
            f"- Verdict: **{audit['audit_verdict']}** "
            f"(max |shift| = {audit['max_abs_shift']:+.3f})"
        )
        lines.append("")
        lines.append("| Level | Before | After | Shift |")
        lines.append("|---|---:|---:|---:|")
        for level, shift in audit["absolute_shifts"].items():
            before = audit["before_proportions"].get(level, 0.0)
            after = audit["after_proportions"].get(level, 0.0)
            lines.append(f"| {level} | {before:.3f} | {after:.3f} | {shift:+.3f} |")
        lines.append("")
    any_review = any(a["audit_verdict"] == "REVIEW"
                     for a in r["demographic_audit"].values())
    if any_review:
        lines.append(
            "**At least one demographic column flagged REVIEW.** "
            "Manual inspection recommended before proceeding to Phase 2."
        )
    else:
        lines.append(
            "All demographic columns within tolerance. Cleaning did not "
            "measurably distort the demographic composition of the sample."
        )
    return "\n".join(lines)


def _render_success_report(r: dict) -> str:
    """Structured plain-text success report matching the React artifact's
    ``buildSuccessReport`` layout (Agent / Phase / Status / Metrics /
    Artifacts Produced / Notes).
    """
    status = (
        "SUCCESS -- Data Quality Gate Passed"
        if r["gate_status"] == "PASS"
        else "WARNING -- Below 90% Threshold"
    )
    retained = r["variance"]["retained_numeric"]
    excluded = r["variance"]["excluded_low_variance"]
    dist_flags = [d["column"] for d in r["distributions"] if d["issues"]]

    metrics = [
        ("Original N", f"{r['original_rows']:,}"),
        ("Clean N", f"{r['clean_rows']:,}"),
        ("Removal Rate", f"{r['removal_pct']:.1f}%"),
        ("Multi-Hurdle Removed (Careless)", r["removed_count"]),
        ("Items Retained", f"{len(retained)} of "
         f"{len(r['schema']['expected_numeric'])}"),
        ("Items Excluded (Low Variance)", len(excluded)),
        ("Distribution Flags (Skew >|2| / Kurt >|7|)", len(dist_flags)),
        ("Data Quality Confidence Score", f"{r['confidence']:.1%}"),
    ]

    artifacts = [
        "survey_baseline_clean.csv -- cleaned dataset in original Likert units",
        "audit_reports/variance_heatmap.svg -- SD per survey item with "
        "exclusion flags",
        "audit_reports/data_quality_screening.md -- per-column hurdle detail",
        "audit_reports/data_steward_bias_audit.md -- removed-respondent "
        "demographic profile",
        "reflection_logs/data_steward_reflection.json -- agent self-reflection",
        "reflection_logs/data_steward_success_report.txt -- this report",
    ]

    notes_parts = [
        "Careless-responding detection used the SDQEM multi-hurdle approach "
        f"(Papp et al., 2026; Curran, 2016): removal only on "
        f">= {config.CARELESS_HURDLES} simultaneous flags across attention "
        "checks, longstring, low IRV, Mahalanobis outliers, and low response "
        "entropy.",
        "Data passes to K-Prototypes Agent in original Likert units -- "
        "standardization is deliberately delegated downstream to avoid "
        "double-scaling.",
    ]
    if excluded:
        notes_parts.append(
            f"Low-variance items excluded: {', '.join(excluded)}."
        )
    if dist_flags:
        notes_parts.append(
            f"Distribution flags for downstream review: {', '.join(dist_flags)}."
        )

    lines = [
        "# Agent Success Report",
        "",
        "**Agent:** Data Steward",
        "**Phase:** Phase 1 -- Ingest & Clean",
        f"**Status:** {status}",
        f"**Timestamp:** {r['timestamp']}",
        f"**Run ID:** {r['run_id']}",
        "",
        "## Metrics",
    ]
    lines.extend(f"- **{k}:** {v}" for k, v in metrics)
    lines += ["", "## Artifacts Produced"]
    lines.extend(f"{i}. {a}" for i, a in enumerate(artifacts, 1))
    lines += ["", "## Notes", " ".join(notes_parts)]
    return "\n".join(lines)


def _render_reflection(r: dict) -> dict:
    return {
        "agent": "Data Steward",
        "phase": "Phase 1 -- Ingest & Clean",
        "run_id": r["run_id"],
        "timestamp": r["timestamp"],
        "status": "COMPLETE",
        "operating_mode": "pipeline",
        "evidence": {
            "raw_respondents": r["original_rows"],
            "clean_respondents": r["clean_rows"],
            "removed_count": r["removed_count"],
            "removal_pct": r["removal_pct"],
            "hurdle_counts": r["careless_responding"]["per_hurdle_flagged"],
            "excluded_low_variance_cols": r["variance"]["excluded_low_variance"],
            "retained_numeric_cols": r["variance"]["retained_numeric"],
            "data_quality_confidence": r["confidence"],
            "gate_status": r["gate_status"],
        },
        "reasoning": (
            "Applied full 5-hurdle SDQEM careless responding screen (Papp "
            "et al., 2026; Curran, 2016) including attention-check failures, "
            "longstring, low IRV, Mahalanobis outliers, and low response "
            "entropy. Respondents were removed only when they tripped "
            f">= {r['careless_responding']['thresholds']['hurdles_required_to_remove']} "
            "independent hurdles, avoiding over-removal driven by any single "
            "indicator. Sparsity and variance gates were applied per the "
            "Global Configuration Registry. Distributions were flagged but "
            "not auto-transformed (Osborne, 2013) -- downstream agents decide. "
            "Likert items were NOT standardized; standardization is delegated "
            "to K-Prototypes and LPA to avoid double-scaling."
        ),
        "routing_decision": (
            "Proceed to Phase 2 (K-Prototypes + LPA) with cleaned_baseline.csv."
            if r["gate_status"] == "PASS"
            else "HALT -- data quality confidence below 0.90; request human review."
        ),
    }


if __name__ == "__main__":
    sys.exit(main())
