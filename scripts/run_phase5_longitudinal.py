"""
Phase 5 Longitudinal Alignment for the SIOP 2026 pipeline.

Applies the Continuity Agent to map follow-up respondents onto the
baseline K-Prototypes centroids and flags weak-fit individuals who
do not clearly belong to any baseline segment.  If enough weak-fit
respondents are available, the Emergence Agent runs a K+1 test on
that pool to detect genuinely new sub-structure.

References
----------
Lu, Z. (2025). Clustering longitudinal data: A review. ISR, 93.
Moore, Quartiroli, & Little (2025). Best practice recommendations
    for longitudinal latent transition analysis.
Bakac, Zyberaj, & Barber (2022). Latent transition analysis in
    organizational psychology. Frontiers in Psychology.

Phases 1 and 2 must have run first.

Run (from project root):
    python scripts/run_phase5_longitudinal.py
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.p1_ingest.data_steward import run_data_steward
from src.p3_ground.emergence import test_new_segments
from src.p5_longitudinal.continuity import align_to_baseline

PHASE1_DIR = REPO_ROOT / "outputs" / "phase1_data_quality_report"
PHASE2_DIR = REPO_ROOT / "outputs" / "phase2_cluster_validation"
FOLLOWUP_CSV = REPO_ROOT / "synthetic_data" / "survey_followup.csv"

BASELINE_CLEAN = PHASE1_DIR / "survey_baseline_clean.csv"
BASELINE_LABELS = PHASE2_DIR / "cluster_labels.csv"
BASELINE_CENTROIDS = PHASE2_DIR / "kproto_centroids.json"

OUT_DIR = REPO_ROOT / "outputs" / "phase5_longitudinal"
AUDIT_DIR = OUT_DIR / "audit_reports"
REFLECT_DIR = OUT_DIR / "reflection_logs"

RUN_ID = str(uuid.uuid4())
TIMESTAMP = datetime.now(timezone.utc).isoformat()

# Known schema drift: the follow-up file names the construct
# "Trust_Leaders", but the baseline and codebook use
# "Trust_Leadership".  Harmonise on load so the Continuity Agent
# can align on identical column names.
COLUMN_ALIASES = {"Trust_Leaders": "Trust_Leadership"}


def _harmonise_followup(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_ALIASES)


def _render_report(summary: dict) -> str:
    fu = summary["followup"]
    al = summary["alignment"]
    ks = summary["k_plus_one"]
    strong_n = al["aligned_n"] - al["weak_fit_count"]
    strong_pct = (
        round(strong_n / max(al["aligned_n"], 1) * 100, 1)
        if al["aligned_n"] else 0.0
    )
    lines = [
        "# Phase 5: Longitudinal Alignment (Bonus)",
        "",
        "**Agents:** Continuity + Emergence (Longitudinal Mode)",
        "**Subtitle:** Following the workforce across time.",
        "",
        f"- **Run ID:** `{summary['run_id']}`",
        f"- **Timestamp:** {summary['timestamp']}",
        f"- **Baseline N:** {summary['baseline_n']:,}",
        f"- **Follow-up N (raw):** {fu['raw_n']:,}",
        f"- **Follow-up N (clean):** {fu['clean_n']:,}",
        "",
        "## Metrics",
        "",
        f"- **Follow-up screening removed:** {fu['removed_count']:,} "
        f"({fu['removal_pct']:.1f}%)",
        f"- **Strongly aligned (follow-up respondents fitting baseline "
        f"clusters):** {strong_n:,} ({strong_pct:.1f}%)",
        f"- **Weak-fit (min distance > {config.WEAK_FIT_DISTANCE}):** "
        f"{al['weak_fit_count']:,} ({al['weak_fit_pct']:.1f}%)",
        f"- **Schema warnings raised:** "
        f"{len(summary.get('schema_warnings') or [])}",
        "",
    ]
    if summary.get("schema_warnings"):
        lines.append("### Schema warnings (see "
                     "`audit_reports/schema_drift_audit.md`)")
        lines.append("")
        for w in summary["schema_warnings"]:
            lines.append(f"- {w}")
        lines.append("")
        lines.append(
            "> **Note:** a 100% weak-fit rate almost always indicates "
            "categorical-level vocabulary drift between waves, not "
            "a workforce revolution. Harmonise the follow-up CSV's "
            "categorical levels with the baseline before drawing "
            "conclusions."
        )
        lines.append("")

    lines.append("## Migration Patterns (baseline -> follow-up; aggregate "
                 "proportions)")
    lines.append("")
    if al.get("transition_matrix"):
        lines.append("| Cluster | Baseline % | Follow-up % | Change (ppt) |")
        lines.append("|---:|---:|---:|---:|")
        for cid, row in al["transition_matrix"].items():
            lines.append(
                f"| {cid} | {row['Baseline_pct']:.1f} | "
                f"{row['Followup_pct']:.1f} | {row['Change_ppt']:+.1f} |"
            )
        lines.append("")

    lines.append("## K+1 Emergence Test on the Weak-Fit Pool")
    lines.append("")
    kp = ks.get("k_plus_1_results", {})
    if kp.get("skipped"):
        lines.append(
            f"Skipped: {kp.get('reason', 'insufficient sample')}. "
            "A weak-fit pool below n=30 cannot support a meaningful "
            "K+1 clustering test."
        )
    else:
        lines.append("| K | BIC | Silhouette | Cluster sizes |")
        lines.append("|---:|---:|---:|---|")
        for k, stats in kp.items():
            if not isinstance(stats, dict):
                continue
            sil = stats.get("silhouette")
            sil_str = f"{sil:.3f}" if isinstance(sil, (int, float)) else "-"
            sizes = stats.get("sizes", [])
            lines.append(
                f"| {k} | {stats.get('bic', '-')} | {sil_str} | {sizes} |"
            )
        lines.append("")
        cands = ks.get("new_segment_candidates") or []
        if cands:
            lines.append("**Candidate new segments flagged for review:**")
            lines.append("")
            for c in cands:
                lines.append(
                    f"- K={c['k']}, silhouette={c['silhouette']:.3f}: "
                    f"{c['interpretation']}"
                )
            lines.append("")
        else:
            lines.append(
                "No coherent new segment detected in the weak-fit pool. "
                "Baseline K structure holds."
            )
            lines.append("")

    lines += [
        "## Gate 5: Your Decision (Continue or Conclude)",
        "",
        "Unlike Gates 1-4, Phase 5 has no hard accept/reject; it produces "
        "a findings + recommendations + next-steps summary.",
        "",
        "**Findings to review**",
        "",
        "- [ ] Does the baseline cluster structure hold "
        "(>= 90% strongly aligned, weak-fit rate <= 10%)?",
        "- [ ] Are migration patterns consistent with known workforce "
        "events (layoffs, reorgs, acquisitions)?",
        "- [ ] Does the K+1 test suggest a new segment? If so, is its "
        "size and silhouette large enough to warrant follow-up?",
        "- [ ] Are any schema warnings addressed before drawing conclusions?",
        "",
        "| Recommendation | What it means |",
        "|---|---|",
        "| Continue longitudinal analysis | When the next follow-up arrives, "
        "re-run Continuity + Emergence. Track the weak-fit pool. |",
        "| Conclude analysis | Baseline personas remain valid. Use them for "
        "the current decision cycle. |",
        "",
        "## Artifacts Produced",
        "",
        "1. `followup_cleaned.csv` -- cleaned follow-up dataset",
        "2. `aligned_labels.csv` -- per-respondent: aligned cluster, "
        "weak-fit flag, min distance",
        "3. `transition_matrix.csv` -- aggregate baseline -> follow-up "
        "proportions",
        "4. `k_plus_one_test.json` -- BIC / silhouette / sizes for K=1,2,3 "
        "on the weak-fit pool",
        "5. `audit_reports/schema_drift_audit.md` -- categorical level "
        "vocabulary drift between waves",
        "6. `audit_reports/audit_trail.json` -- every agent action",
        "7. `reflection_logs/phase5_success_report.txt` -- status, metrics, "
        "artifacts produced",
        "",
    ]
    return "\n".join(lines)


def _render_schema_drift_audit(schema_warnings: list) -> str:
    lines = [
        "# Schema Drift Audit (Phase 5)",
        "",
        "Surveys rarely use identical categorical level vocabulary across "
        "waves. This audit surfaces level-name mismatches that would "
        "otherwise inflate Hamming distance against baseline centroids and "
        "produce an artifactually high weak-fit rate. Harmonise follow-up "
        "levels to match baseline *before* interpreting migration.",
        "",
        "## Warnings raised this run",
        "",
    ]
    if not schema_warnings:
        lines.append("*No warnings raised. Follow-up schema matches "
                     "baseline.*")
    else:
        for w in schema_warnings:
            lines.append(f"- {w}")
    lines += [
        "",
        "## How to remediate",
        "",
        "1. Open the follow-up CSV.",
        "2. For each warning above, identify the baseline spelling "
        "(e.g., `Senior`) and the follow-up spelling (e.g., `Sr`).",
        "3. Map every follow-up value to the corresponding baseline value.",
        "4. Re-run Phase 5.",
        "",
        "For a tutorial demo you can skip this step and note the caveat in "
        "the Gate 5 decision; for production use, always harmonise before "
        "publishing migration claims.",
    ]
    return "\n".join(lines)


def _render_success_report(summary: dict) -> str:
    fu = summary["followup"]
    al = summary["alignment"]
    ks = summary["k_plus_one"]
    strong_n = al["aligned_n"] - al["weak_fit_count"]
    strong_pct = (
        round(strong_n / max(al["aligned_n"], 1) * 100, 1)
        if al["aligned_n"] else 0.0
    )
    has_drift = bool(summary.get("schema_warnings"))
    status = (
        "WARNING -- Schema drift detected; interpret weak-fit with care"
        if has_drift
        else "SUCCESS -- Longitudinal alignment complete"
    )
    metrics = [
        ("Baseline N", f"{summary['baseline_n']:,}"),
        ("Follow-up N (raw)", f"{fu['raw_n']:,}"),
        ("Follow-up N (clean)", f"{fu['clean_n']:,}"),
        ("Follow-up removal rate", f"{fu['removal_pct']:.1f}%"),
        ("Strongly aligned", f"{strong_n:,} ({strong_pct:.1f}%)"),
        ("Weak-fit", f"{al['weak_fit_count']:,} ({al['weak_fit_pct']:.1f}%)"),
        ("Weak-fit threshold", config.WEAK_FIT_DISTANCE),
        ("Schema warnings", len(summary.get("schema_warnings") or [])),
        ("New-segment candidates (K+1 test)",
         len(ks.get("new_segment_candidates") or [])),
    ]
    artifacts = [
        "followup_cleaned.csv -- cleaned follow-up data",
        "aligned_labels.csv -- per-respondent cluster, weak-fit flag, "
        "min distance",
        "transition_matrix.csv -- baseline -> follow-up proportions",
        "k_plus_one_test.json -- K+1 emergence test results",
        "audit_reports/schema_drift_audit.md -- level vocabulary drift",
        "audit_reports/audit_trail.json",
        "reflection_logs/phase5_success_report.txt -- this report",
    ]
    notes = (
        "Continuity alignment uses Euclidean distance on Z-scored numeric "
        "features (standardised against baseline mean/SD, not follow-up's "
        "own -- self-standardisation would mask genuine workforce change) "
        "and Hamming distance on categorical features, weighted by feature "
        "count. K+1 emergence testing (Lu, 2025; Moore, Quartiroli, & "
        "Little, 2025) fits GMMs at K=1,2,3 on the weak-fit pool and "
        "compares BIC + silhouette. A Δ silhouette > 0.05 is required to "
        "flag a genuine new segment."
    )
    lines = [
        "# Agent Success Report",
        "",
        "**Agents:** Continuity + Emergence (Longitudinal Mode)",
        "**Phase:** Phase 5 -- Longitudinal Alignment (Bonus)",
        f"**Status:** {status}",
        f"**Timestamp:** {summary['timestamp']}",
        f"**Run ID:** {summary['run_id']}",
        "",
        "## Metrics",
    ]
    lines.extend(f"- **{k}:** {v}" for k, v in metrics)
    lines += ["", "## Artifacts Produced"]
    lines.extend(f"{i}. {a}" for i, a in enumerate(artifacts, 1))
    lines += ["", "## Notes", notes]
    return "\n".join(lines)


def main() -> int:
    print("=" * 60)
    print("  PHASE 5: LONGITUDINAL ALIGNMENT")
    print("=" * 60)
    print(f"Run ID:    {RUN_ID}")
    print(f"Timestamp: {TIMESTAMP}")

    for path, name in (
        (BASELINE_CLEAN, "Phase 1 cleaned baseline"),
        (BASELINE_LABELS, "Phase 2 cluster labels"),
        (BASELINE_CENTROIDS, "Phase 2 K-Proto centroids"),
        (FOLLOWUP_CSV, "Follow-up survey CSV"),
    ):
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            return 2

    for d in (OUT_DIR, AUDIT_DIR, REFLECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load baseline artefacts ----
    baseline_df = pd.read_csv(BASELINE_CLEAN)
    baseline_labels_df = pd.read_csv(BASELINE_LABELS)
    baseline_centroids = json.loads(BASELINE_CENTROIDS.read_text(encoding="utf-8"))
    baseline_labels = baseline_labels_df["kproto_cluster"].to_numpy()
    print(f"Loaded {len(baseline_df):,} baseline respondents and "
          f"{len(baseline_labels):,} cluster labels")

    # ---- Load and harmonise follow-up ----
    followup_raw = pd.read_csv(FOLLOWUP_CSV)
    original_n = len(followup_raw)
    followup_harm = _harmonise_followup(followup_raw)

    schema_warnings = []
    applied = [c for c in COLUMN_ALIASES if c in followup_raw.columns]
    if applied:
        schema_warnings.append(
            "Renamed follow-up columns to match baseline schema: "
            + ", ".join(f"{c} -> {COLUMN_ALIASES[c]}" for c in applied)
        )

    missing = [c for c in config.NUMERIC_COLS + config.CATEGORICAL_COLS
               if c not in followup_harm.columns]
    if missing:
        print(f"ERROR: follow-up CSV is missing required columns: {missing}")
        return 2

    # Compare categorical level vocabularies to surface silent drift.
    for col in config.CATEGORICAL_COLS:
        base_levels = set(baseline_df[col].dropna().unique())
        fu_levels = set(followup_harm[col].dropna().unique())
        only_fu = fu_levels - base_levels
        if only_fu:
            schema_warnings.append(
                f"Follow-up '{col}' contains levels absent from baseline: "
                + ", ".join(sorted(map(str, only_fu)))
            )

    # ---- Data Steward on follow-up ----
    print("\n-- Data Steward screening (follow-up) --")
    ds_out = run_data_steward(
        followup_harm,
        survey_cols=list(config.NUMERIC_COLS),
        cat_cols=list(config.CATEGORICAL_COLS),
    )
    followup_clean = ds_out["clean_df"]
    removed = ds_out["removed_count"]
    print(f"  Raw respondents : {original_n:,}")
    print(f"  Removed         : {removed:,} "
          f"({removed / max(original_n, 1) * 100:.1f}%)")
    print(f"  Clean           : {len(followup_clean):,}")

    # ---- Continuity alignment ----
    print("\n-- Aligning follow-up to baseline centroids --")
    al_out = align_to_baseline(
        followup_clean, baseline_centroids,
        cat_cols=list(config.CATEGORICAL_COLS),
        num_cols=list(config.NUMERIC_COLS),
        baseline_labels=baseline_labels,
        baseline_df=baseline_df,
    )
    n_aligned = len(al_out["aligned_labels"])
    weak_count = al_out["weak_fit_count"]
    weak_pct = round(weak_count / max(n_aligned, 1) * 100, 1)
    print(f"  Aligned respondents : {n_aligned:,}")
    print(f"  Weak-fit flagged    : {weak_count} ({weak_pct}%)")

    # ---- K+1 emergence test on weak-fit pool ----
    followup_aligned = followup_clean.iloc[:n_aligned].reset_index(drop=True).copy()
    followup_aligned["aligned_cluster"] = al_out["aligned_labels"]
    followup_aligned["weak_fit"] = al_out["weak_fit_mask"]
    followup_aligned["min_distance"] = al_out["distance_matrix"].min(axis=1)

    weak_fit_df = followup_aligned.loc[followup_aligned["weak_fit"]]
    print(f"\n-- K+1 emergence test on weak-fit pool "
          f"(n={len(weak_fit_df)}) --")
    ks_out = test_new_segments(
        weak_fit_df,
        cat_cols=list(config.CATEGORICAL_COLS),
        num_cols=list(config.NUMERIC_COLS),
    )
    if ks_out["k_plus_1_results"].get("skipped"):
        print(f"  Skipped: {ks_out['k_plus_1_results'].get('reason')}")
    else:
        for k, stats in ks_out["k_plus_1_results"].items():
            sil = stats.get("silhouette")
            sil_str = f"{sil:.3f}" if isinstance(sil, (int, float)) else "n/a"
            print(f"  K={k}: BIC={stats['bic']:.1f}  silhouette={sil_str}  "
                  f"sizes={stats['sizes']}")

    # ---- Persist ----
    clean_path = OUT_DIR / "followup_cleaned.csv"
    followup_clean.to_csv(clean_path, index=False)
    print(f"\nWrote: {clean_path.relative_to(REPO_ROOT)}")

    aligned_path = OUT_DIR / "aligned_labels.csv"
    followup_aligned[[
        "aligned_cluster", "weak_fit", "min_distance"
    ]].to_csv(aligned_path, index=False)
    print(f"Wrote: {aligned_path.relative_to(REPO_ROOT)}")

    trans_path = OUT_DIR / "transition_matrix.csv"
    transition_dict = None
    if al_out["transition_matrix"] is not None:
        al_out["transition_matrix"].to_csv(trans_path)
        transition_dict = al_out["transition_matrix"].to_dict(orient="index")
        print(f"Wrote: {trans_path.relative_to(REPO_ROOT)}")

    ks_path = OUT_DIR / "k_plus_one_test.json"
    ks_path.write_text(json.dumps({
        "k_plus_1_results": ks_out["k_plus_1_results"],
        "new_segment_candidates": ks_out["new_segment_candidates"],
    }, indent=2, default=str))
    print(f"Wrote: {ks_path.relative_to(REPO_ROOT)}")

    audit = (
        ds_out["audit_entries"]
        + al_out["audit_entries"]
        + ks_out["audit_entries"]
    )
    audit_path = AUDIT_DIR / "audit_trail.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    print(f"Wrote: {audit_path.relative_to(REPO_ROOT)}")

    drift_path = AUDIT_DIR / "schema_drift_audit.md"
    drift_path.write_text(
        _render_schema_drift_audit(schema_warnings), encoding="utf-8"
    )
    print(f"Wrote: {drift_path.relative_to(REPO_ROOT)}")

    summary = {
        "run_id": RUN_ID,
        "timestamp": TIMESTAMP,
        "baseline_n": int(len(baseline_df)),
        "schema_warnings": schema_warnings,
        "followup": {
            "raw_n": int(original_n),
            "clean_n": int(len(followup_clean)),
            "removed_count": int(removed),
            "removal_pct": round(removed / max(original_n, 1) * 100, 2),
        },
        "alignment": {
            "aligned_n": int(n_aligned),
            "weak_fit_count": int(weak_count),
            "weak_fit_pct": float(weak_pct),
            "transition_matrix": transition_dict,
        },
        "k_plus_one": {
            "k_plus_1_results": ks_out["k_plus_1_results"],
            "new_segment_candidates": ks_out["new_segment_candidates"],
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote: {summary_path.relative_to(REPO_ROOT)}")

    md_path = OUT_DIR / "report.md"
    md_path.write_text(_render_report(summary), encoding="utf-8")
    print(f"Wrote: {md_path.relative_to(REPO_ROOT)}")

    success_path = REFLECT_DIR / "phase5_success_report.txt"
    success_path.write_text(_render_success_report(summary), encoding="utf-8")
    print(f"Wrote: {success_path.relative_to(REPO_ROOT)}")

    # ---- Success report ----
    print("\n" + "=" * 60)
    print("  PHASE 5 -- SUCCESS REPORT")
    print("=" * 60)
    print(f"Status:              COMPLETE")
    print(f"Follow-up raw:       {original_n:,}")
    print(f"Follow-up clean:     {len(followup_clean):,}")
    print(f"Weak-fit flagged:    {weak_count} ({weak_pct}%)")
    if ks_out["new_segment_candidates"]:
        print(f"New-segment candidates: "
              f"{len(ks_out['new_segment_candidates'])}")
    print(f"\nNext: open outputs/phase5_longitudinal/report.md "
          f"for the Gate 5 review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
