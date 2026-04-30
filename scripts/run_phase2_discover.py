"""
Phase 2 Workforce Segment Discovery for the SIOP 2026 personas pipeline.

Runs two independent clustering methods on the cleaned baseline and
then validates both with the Psychometrician.

    K-Prototypes  -- Huang (1998): demographics + survey items together.
    LPA / GMM     -- Spurk et al. (2020): survey items only.
    Psychometrician -- Rousseeuw (1987), Hubert & Arabie (1985):
                       silhouette, outlier flagging, ARI.

Phase 1 must have run first; this script reads
``outputs/phase1_data_quality_report/cleaned_baseline.csv`` and, when
present, respects the ``variance.retained_numeric`` list from
``quality_report.json`` so that Likert items the Data Steward excluded
for low variance are not handed to the clustering algorithms.

Run (from project root):
    python scripts/run_phase2_discover.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.p2_discover.k_prototypes import run_k_prototypes
from src.p2_discover.lpa import run_lpa
from src.p2_discover.psychometrician import run_validation

PHASE1_DIR = REPO_ROOT / "outputs" / "phase1_data_quality_report"
CLEAN_CSV = PHASE1_DIR / "survey_baseline_clean.csv"
P1_QUALITY = PHASE1_DIR / "summary.json"

OUT_DIR = REPO_ROOT / "outputs" / "phase2_cluster_validation"
FIG_DIR = OUT_DIR / "audit_reports" / "figures"
AUDIT_DIR = OUT_DIR / "audit_reports"
REFLECT_DIR = OUT_DIR / "reflection_logs"

RUN_ID = str(uuid.uuid4())
TIMESTAMP = datetime.now(timezone.utc).isoformat()

np.random.seed(config.SEED)


def _load_retained_numeric() -> list[str]:
    """Use the variance-gated indicator list from Phase 1 when available."""
    if not P1_QUALITY.exists():
        return list(config.NUMERIC_COLS)
    try:
        report = json.loads(P1_QUALITY.read_text(encoding="utf-8"))
        retained = report.get("variance", {}).get("retained_numeric")
        if retained:
            return list(retained)
    except (json.JSONDecodeError, OSError):
        pass
    return list(config.NUMERIC_COLS)


def _plot_elbow(elbow_data: dict, k_selected: int, path: Path) -> None:
    """Two-panel elbow + silhouette diagnostic (emitted as SVG)."""
    ks = list(elbow_data["k_values"])
    costs = elbow_data["costs"]
    sils = [elbow_data["silhouette_scores"][int(k)] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(ks, costs, marker="o", color="#3b82f6")
    ax1.axvline(k_selected, color="#ef4444", linestyle="--",
                label=f"K selected = {k_selected}")
    ax1.set_xlabel("K")
    ax1.set_ylabel("K-Prototypes cost")
    ax1.set_title("Elbow curve")
    ax1.legend(fontsize=8)

    ax2.plot(ks, sils, marker="s", color="#22c55e")
    ax2.axvline(k_selected, color="#ef4444", linestyle="--")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette by K")

    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _df_to_md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Avoids the pandas ``to_markdown`` dependency on ``tabulate``,
    which is not part of the tutorial's required-packages set.
    """
    cols = [str(c) for c in df.columns]
    rows = [cols, ["---"] * len(cols)]
    for _, r in df.iterrows():
        rows.append([
            f"{v:.3f}" if isinstance(v, float) else str(v)
            for v in r.values
        ])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _render_report(summary: dict) -> str:
    kp = summary["k_prototypes"]
    lp = summary["lpa"]
    val = summary["validation"]
    ari = val.get("ari")
    ari_str = f"{ari:.3f}" if isinstance(ari, (int, float)) else "n/a"
    ari_band = val.get("ari_quality") or "n/a"
    lines = [
        "# Phase 2: Discover Workforce Segments",
        "",
        "**Agents:** K-Prototypes, LPA, Psychometrician",
        "**Subtitle:** Two independent clustering methods. One validated truth.",
        "",
        f"- **Run ID:** `{summary['run_id']}`",
        f"- **Timestamp:** {summary['timestamp']}",
        f"- **Respondents clustered:** {summary['n_respondents']:,}",
        f"- **Numeric indicators used:** {', '.join(summary['numeric_cols'])}",
        f"- **Categorical features used:** {', '.join(summary['categorical_cols'])}",
        "",
        "## Metrics",
        "",
        f"- **K-Prototypes K selected:** {kp['k_selected']} "
        f"(gamma = {kp['gamma']:.3f}, silhouette = "
        f"{kp['silhouette_at_selected']:.3f})",
        f"- **LPA profiles selected:** {lp['k_selected']} "
        f"({lp['cov_type']} covariance, BIC = {lp['bic']:.1f}, entropy = "
        f"{lp['entropy']:.3f})",
        f"- **Global silhouette (Psychometrician):** "
        f"{val['silhouette_overall']:.3f} -> {val['quality']}",
        f"- **ARI (K-Prototypes vs. LPA):** {ari_str} -> {ari_band}",
        f"- **Outliers flagged (top {config.OUTLIER_PERCENTILE}th pctile):** "
        f"{val['n_outliers']} ({val['pct_outliers']:.1f}%)",
        f"- **LPA ambiguous respondents (posterior < "
        f"{config.LPA_AMBIGUITY_POSTERIOR}):** {lp['ambiguous_count']} "
        f"({lp['ambiguous_pct']:.1f}%)",
        "",
        "## Gate 2: Your Decision",
        "",
        "> **Do you accept this cluster solution "
        "(K-Prototypes + LPA consensus)?**",
        "",
        "**Evidence Summary**",
        "",
        "- [ ] K-Prototypes K selection is defensible (elbow + silhouette)",
        "- [ ] LPA model selection is defensible (BIC + entropy)",
        "- [ ] Silhouette quality band is acceptable for downstream personas",
        "- [ ] ARI between methods agrees with substantive expectation "
        "(STRONG > 0.65 / MODERATE 0.30-0.65 / WEAK < 0.30)",
        "- [ ] Outlier rate does not mask a legitimate sub-group",
        "- [ ] LPA ambiguous rate does not exceed 15% without explanation",
        "",
        "| Option | When to choose | What happens |",
        "|---|---|---|",
        "| Yes -- Accept Cluster Solution | Both methods defensible; validation "
        "bands acceptable | Clusters route to RAG + Emergence (Phase 3) |",
        "| Investigate | Want to note a discrepancy (e.g., ARI < 0.30) but "
        "proceed with K-Prototypes as primary | Flag in audit trail; continue |",
        "| No -- Reject | Silhouette POOR or solution unstable in ways you "
        "cannot defend to leadership | Return to parameter tuning; halted |",
        "",
        "## K-Prototypes (Huang, 1998)",
        "",
        f"- **K selected:** {kp['k_selected']}",
        f"- **Gamma (categorical weight):** {kp['gamma']:.3f}",
        f"- **Silhouette at selected K:** {kp['silhouette_at_selected']:.3f}",
        "",
        "| K | Cost | Silhouette |",
        "|---:|---:|---:|",
    ]
    for k, cost, sil in zip(
        kp["elbow"]["k_values"],
        kp["elbow"]["costs"],
        [kp["elbow"]["silhouette_scores"][int(k)] for k in kp["elbow"]["k_values"]],
    ):
        lines.append(f"| {k} | {cost:.2f} | {sil:.3f} |")

    lines += [
        "",
        "### Cluster profiles (means on Z-scored indicators; modal demographics)",
        "",
    ]
    profile_df = pd.DataFrame(kp["profiles"])
    lines.append(_df_to_md_table(profile_df))

    lines += [
        "",
        "## Latent Profile Analysis (Spurk et al., 2020)",
        "",
        f"- **Profiles selected:** {lp['k_selected']} ({lp['cov_type']} covariance)",
        f"- **BIC:** {lp['bic']:.1f}",
        f"- **Entropy:** {lp['entropy']:.3f}",
        f"- **Ambiguous respondents (max posterior < "
        f"{config.LPA_AMBIGUITY_POSTERIOR}):** "
        f"{lp['ambiguous_count']} ({lp['ambiguous_pct']:.1f}%)",
        "",
        "### Psychological Fingerprints",
        "",
        "| Profile | n | % | Label |",
        "|---:|---:|---:|---|",
    ]
    for prof_row, (pid, fp) in zip(lp["profiles"], lp["fingerprints"].items()):
        lines.append(
            f"| {pid} | {prof_row['n']:,} | {prof_row['pct']:.1f}% | "
            f"{fp['label']} |"
        )

    lines += [
        "",
        "## Cross-model validation (Psychometrician)",
        "",
        f"- **Global silhouette (K-Prototypes):** {val['silhouette_overall']:.3f} "
        f"-> {val['quality']}",
        f"- **Outliers flagged (top {config.OUTLIER_PERCENTILE}th pctile):** "
        f"{val['n_outliers']} ({val['pct_outliers']:.1f}%)",
    ]
    if val.get("ari") is not None:
        lines.append(
            f"- **ARI (K-Proto vs. LPA):** {val['ari']:.3f} -> "
            f"{val['ari_quality']}"
        )
        lines.append(f"  - {val['ari_interpretation']}")
    lines += [
        "",
        "### Per-cluster silhouette",
        "",
        "| K-Prototypes cluster | Mean silhouette | % negative |",
        "|---:|---:|---:|",
    ]
    for cid, stats in val["silhouette_per_cluster"].items():
        lines.append(f"| {cid} | {stats['mean']:.3f} | {stats['pct_negative']:.1f}% |")

    lines += [
        "",
        "## Agent reasoning",
        "",
        f"**K-Prototypes:** {kp['reasoning']}",
        "",
        f"**LPA:** {lp['reasoning']}",
        "",
        f"**Psychometrician:** {val['reasoning']}",
        "",
        "## Artifacts Produced",
        "",
        "1. `cluster_labels.csv` -- per-respondent: K-Prototypes cluster, "
        "LPA profile, posterior probability, outlier flag, centroid distance",
        "2. `kproto_profiles.csv` -- one row per K-Prototypes cluster (means "
        "on Z-scored indicators plus modal demographics)",
        "3. `kproto_centroids.json` -- centroids for Phase 5 alignment",
        "4. `lpa_profiles.csv` -- one row per LPA profile",
        "5. `lpa_fingerprints.json` -- Psychological Fingerprints "
        "(High / Moderate / Low per indicator)",
        "6. `audit_reports/figures/kproto_elbow.svg` -- elbow + silhouette "
        "diagnostic",
        "7. `audit_reports/audit_trail.json` -- every agent action with "
        "timestamp",
        "8. `reflection_logs/phase2_success_report.txt` -- status, metrics, "
        "artifacts produced",
        "",
    ]
    return "\n".join(lines)


def _render_success_report(summary: dict) -> str:
    """``buildSuccessReport``-style plain-text status summary."""
    kp = summary["k_prototypes"]
    lp = summary["lpa"]
    val = summary["validation"]
    ari = val.get("ari")
    ari_str = f"{ari:.3f}" if isinstance(ari, (int, float)) else "n/a"
    quality = val["quality"]
    status = (
        f"SUCCESS -- Cluster solution validated (silhouette {quality})"
        if quality in ("EXCELLENT", "GOOD")
        else f"WARNING -- Silhouette {quality}; interpret with care"
    )

    metrics = [
        ("Respondents clustered", f"{summary['n_respondents']:,}"),
        ("K-Prototypes K", kp["k_selected"]),
        ("K-Prototypes gamma", f"{kp['gamma']:.3f}"),
        ("K-Prototypes silhouette (at K)",
         f"{kp['silhouette_at_selected']:.3f}"),
        ("LPA profiles (selected)", f"{lp['k_selected']} ({lp['cov_type']})"),
        ("LPA BIC", f"{lp['bic']:.1f}"),
        ("LPA entropy", f"{lp['entropy']:.3f}"),
        ("LPA ambiguous (posterior < threshold)",
         f"{lp['ambiguous_count']} ({lp['ambiguous_pct']:.1f}%)"),
        ("Global silhouette (Gower)",
         f"{val['silhouette_overall']:.3f} -> {quality}"),
        ("Outliers (top 10th pctile)",
         f"{val['n_outliers']} ({val['pct_outliers']:.1f}%)"),
        ("ARI (K-Proto vs. LPA)", f"{ari_str} -> {val.get('ari_quality', 'n/a')}"),
    ]

    artifacts = [
        "cluster_labels.csv -- per-respondent cluster, profile, posterior, "
        "outlier, distance",
        "kproto_profiles.csv -- one row per K-Prototypes cluster",
        "kproto_centroids.json -- needed by Phase 5",
        "lpa_profiles.csv -- one row per LPA profile",
        "lpa_fingerprints.json -- Psychological Fingerprints (High/Mod/Low)",
        "audit_reports/figures/kproto_elbow.svg",
        "audit_reports/audit_trail.json",
        "reflection_logs/phase2_success_report.txt -- this report",
    ]

    notes = (
        "K-Prototypes uses Huang (1998) with Cao initialisation; gamma is set "
        "from the mean SD of standardised numeric indicators. LPA fits "
        "Gaussian Mixture Models across K=2-6 with diagonal and full "
        "covariance; BIC selects the optimal model (Nylund et al., 2007). "
        "Psychometrician validation uses Rousseeuw (1987) silhouette and "
        "Hubert & Arabie (1985) ARI. Respondents are not standardised upstream "
        "in Phase 1; standardisation is performed here and deliberately not "
        "repeated. Ambiguous respondents and outliers are flagged but not "
        "removed -- the I-O psychologist decides at Gate 2."
    )

    lines = [
        "# Agent Success Report",
        "",
        "**Agents:** K-Prototypes, LPA, Psychometrician",
        "**Phase:** Phase 2 -- Discover Workforce Segments",
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
    print("  PHASE 2: WORKFORCE SEGMENT DISCOVERY")
    print("=" * 60)
    print(f"Run ID:    {RUN_ID}")
    print(f"Timestamp: {TIMESTAMP}")

    if not CLEAN_CSV.exists():
        print(f"ERROR: Phase 1 output not found at {CLEAN_CSV}")
        print("Run: python scripts/run_phase1_data_steward.py first.")
        return 2

    for d in (OUT_DIR, FIG_DIR, AUDIT_DIR, REFLECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN_CSV)
    numeric_cols = _load_retained_numeric()
    cat_cols = list(config.CATEGORICAL_COLS)
    print(f"Loaded {len(df):,} clean respondents from Phase 1")
    print(f"Numeric indicators : {numeric_cols}")
    print(f"Categorical cols   : {cat_cols}")
    print()

    # ---- K-Prototypes ----
    # K range is deliberately constrained to 2-4 as a theoretical prior
    # for the tutorial: prior I-O segmentation work on disruption-era
    # engagement surveys typically identifies 2-4 distinct workforce
    # archetypes (Meyer & Allen 1991; Spurk et al. 2020). Attendees are
    # told up front that the range is theory-bounded, not data-fit.
    print("-- Running K-Prototypes (Huang, 1998) --")
    kp_out = run_k_prototypes(df, cat_cols=cat_cols, num_cols=numeric_cols,
                              k_range=range(2, 5))
    print(f"  K selected     : {kp_out['k_selected']}")
    print(f"  Gamma          : {kp_out['centroids']['gamma']:.3f}")
    print(f"  Cluster sizes  : "
          + ", ".join(f"{r['cluster']}: n={r['n']}" for r in
                      kp_out['profiles'].to_dict('records')))

    # ---- LPA ----
    # Same theory-bounded K range (2-4) as K-Prototypes. Letting LPA
    # over-split to K=5-6 on integer-rounded Likert data produces
    # ultra-tight sub-modes that reflect posterior discreteness, not
    # meaningful psychological structure. It is acceptable for LPA and
    # K-Prototypes to disagree within 2-4; disagreement is itself a
    # Gate 2 judgment call.
    print("\n-- Running Latent Profile Analysis --")
    lpa_out = run_lpa(df, indicator_cols=numeric_cols, k_range=range(2, 5))
    selected = lpa_out["fit_indices"]["selected"]
    print(f"  K selected     : {selected['K']} ({selected['cov_type']} covariance)")
    print(f"  BIC            : {selected['BIC']:.1f}")
    print(f"  Entropy        : {selected['entropy']:.3f}")
    print(f"  Ambiguous (posterior < "
          f"{config.LPA_AMBIGUITY_POSTERIOR}): {lpa_out['ambiguous_count']}")

    # ---- Psychometrician ----
    print("\n-- Running Psychometrician validation --")
    val_out = run_validation(
        df, labels_kproto=kp_out["labels"],
        labels_lpa=lpa_out["labels"], feature_cols=numeric_cols,
    )
    print(f"  Silhouette     : {val_out['silhouette_overall']:.3f} "
          f"({val_out['quality']})")
    print(f"  Outliers       : {int(val_out['outlier_flags'].sum())} "
          f"(top {config.OUTLIER_PERCENTILE}th pctile)")
    if val_out["ari"] is not None:
        print(f"  ARI (vs. LPA)  : {val_out['ari']:.3f} "
              f"({val_out['ari_interpretation']})")

    # ---- Persist per-respondent labels ----
    n = len(kp_out["labels"])
    labels_df = pd.DataFrame({
        "respondent_index": np.arange(n),
        "kproto_cluster": kp_out["labels"],
        "lpa_profile": lpa_out["labels"],
        "lpa_posterior": lpa_out["posteriors"].max(axis=1),
        "lpa_is_ambiguous": (lpa_out["posteriors"].max(axis=1)
                             < config.LPA_AMBIGUITY_POSTERIOR),
        "centroid_distance": val_out["centroid_distances"],
        "is_outlier": val_out["outlier_flags"],
    })
    labels_path = OUT_DIR / "cluster_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"\nWrote: {labels_path.relative_to(REPO_ROOT)}")

    # ---- Persist K-Proto profiles and centroids ----
    kp_profiles_path = OUT_DIR / "kproto_profiles.csv"
    kp_out["profiles"].to_csv(kp_profiles_path, index=False)
    print(f"Wrote: {kp_profiles_path.relative_to(REPO_ROOT)}")

    centroids_path = OUT_DIR / "kproto_centroids.json"
    centroids_path.write_text(
        json.dumps(kp_out["centroids"], indent=2, default=str)
    )
    print(f"Wrote: {centroids_path.relative_to(REPO_ROOT)}")

    # ---- Persist LPA profiles and fingerprints ----
    lpa_profiles_path = OUT_DIR / "lpa_profiles.csv"
    lpa_out["profiles"].to_csv(lpa_profiles_path, index=False)
    print(f"Wrote: {lpa_profiles_path.relative_to(REPO_ROOT)}")

    fingerprints_path = OUT_DIR / "lpa_fingerprints.json"
    fingerprints_path.write_text(
        json.dumps(lpa_out["fingerprints"], indent=2, default=str)
    )
    print(f"Wrote: {fingerprints_path.relative_to(REPO_ROOT)}")

    # ---- Elbow plot (SVG, matches artifact preview expectation) ----
    elbow_path = FIG_DIR / "kproto_elbow.svg"
    _plot_elbow(kp_out["elbow_data"], kp_out["k_selected"], elbow_path)
    print(f"Wrote: {elbow_path.relative_to(REPO_ROOT)}")

    # ---- Audit trail ----
    audit = (
        kp_out["audit_entries"]
        + lpa_out["audit_entries"]
        + val_out["audit_entries"]
    )
    audit_path = AUDIT_DIR / "audit_trail.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    print(f"Wrote: {audit_path.relative_to(REPO_ROOT)}")

    # ---- Structured summary ----
    sil_at_selected = kp_out["elbow_data"]["silhouette_scores"][kp_out["k_selected"]]
    summary = {
        "run_id": RUN_ID,
        "timestamp": TIMESTAMP,
        "n_respondents": int(n),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "k_prototypes": {
            "k_selected": kp_out["k_selected"],
            "gamma": kp_out["centroids"]["gamma"],
            "silhouette_at_selected": sil_at_selected,
            "elbow": kp_out["elbow_data"],
            "profiles": kp_out["profiles"].to_dict(orient="records"),
            "reasoning": kp_out["reasoning"],
        },
        "lpa": {
            "k_selected": selected["K"],
            "cov_type": selected["cov_type"],
            "bic": selected["BIC"],
            "entropy": selected["entropy"],
            "ambiguous_count": int(lpa_out["ambiguous_count"]),
            "ambiguous_pct": round(
                lpa_out["ambiguous_count"] / max(n, 1) * 100, 2,
            ),
            "profiles": lpa_out["profiles"].to_dict(orient="records"),
            "fingerprints": {
                int(k): {"label": v["label"], "dims": v["dims"]}
                for k, v in lpa_out["fingerprints"].items()
            },
            "reasoning": lpa_out["reasoning"],
        },
        "validation": {
            "silhouette_overall": val_out["silhouette_overall"],
            "quality": val_out["quality"],
            "silhouette_per_cluster": val_out["silhouette_per_cluster"],
            "n_outliers": int(val_out["outlier_flags"].sum()),
            "pct_outliers": round(
                float(val_out["outlier_flags"].sum()) / max(n, 1) * 100, 2,
            ),
            "ari": val_out["ari"],
            "ari_quality": (
                "STRONG" if val_out["ari"] is not None
                and val_out["ari"] > config.ARI_STRONG
                else "MODERATE" if val_out["ari"] is not None
                and val_out["ari"] > config.ARI_MODERATE
                else "WEAK" if val_out["ari"] is not None
                else None
            ),
            "ari_interpretation": val_out["ari_interpretation"],
            "reasoning": val_out["reasoning"],
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote: {summary_path.relative_to(REPO_ROOT)}")

    md_path = OUT_DIR / "report.md"
    md_path.write_text(_render_report(summary), encoding="utf-8")
    print(f"Wrote: {md_path.relative_to(REPO_ROOT)}")

    success_path = REFLECT_DIR / "phase2_success_report.txt"
    success_path.write_text(_render_success_report(summary), encoding="utf-8")
    print(f"Wrote: {success_path.relative_to(REPO_ROOT)}")

    # ---- Success report ----
    print("\n" + "=" * 60)
    print("  PHASE 2 -- SUCCESS REPORT")
    print("=" * 60)
    print(f"Status:             COMPLETE")
    print(f"Run ID:             {RUN_ID}")
    print(f"K-Prototypes K:     {kp_out['k_selected']}")
    print(f"LPA profiles:       {selected['K']} ({selected['cov_type']} covariance)")
    print(f"Silhouette (K-P):   {val_out['silhouette_overall']:.3f} ({val_out['quality']})")
    if val_out["ari"] is not None:
        print(f"ARI cross-model:    {val_out['ari']:.3f}")
    print(f"\nNext: open outputs/phase2_cluster_validation/report.md "
          f"for the Gate 2 review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
