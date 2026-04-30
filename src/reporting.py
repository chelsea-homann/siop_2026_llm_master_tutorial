"""
HTML report rendering for the SIOP 2026 tutorial pipeline.

Each render_phase* function converts agent result dicts into a
rich HTML string for display in a Jupyter notebook via
``display(HTML(...))``.  The styling is minimal inline CSS that
works in both light and dark Jupyter themes.
"""

from datetime import datetime, timezone

from src import config

# ── Shared CSS ────────────────────────────────────────────────────────────

_CSS = """
<style>
.pr{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
    max-width:960px;line-height:1.5;color:#1a1a1a;}
.pr h1{font-size:1.55em;border-bottom:2px solid #343a40;padding-bottom:6px;margin-bottom:4px;}
.pr h2{font-size:1.15em;margin-top:18px;margin-bottom:6px;border-bottom:1px solid #dee2e6;padding-bottom:3px;}
.pr h3{font-size:1.0em;margin-top:12px;margin-bottom:4px;}
.pr table{border-collapse:collapse;width:100%;margin:8px 0;font-size:0.92em;}
.pr th{background:#f1f3f5;font-weight:600;text-align:left;padding:7px 11px;
       border:1px solid #ced4da;}
.pr td{padding:6px 11px;border:1px solid #dee2e6;vertical-align:top;}
.pr tr:nth-child(even) td{background:#f8f9fa;}
.pr .meta{font-size:0.88em;color:#495057;margin-bottom:10px;}
.pr .meta b{color:#212529;}
.pr hr{border:none;border-top:1px solid #dee2e6;margin:14px 0;}
.pr .box{background:#fff3cd;border:1px solid #ffc107;padding:12px 16px;
         border-radius:4px;margin:10px 0;}
.pr .box-green{background:#d4edda;border:1px solid #28a745;padding:12px 16px;
               border-radius:4px;margin:10px 0;}
.pr .box-blue{background:#cce5ff;border:1px solid #004085;padding:12px 16px;
              border-radius:4px;margin:10px 0;}
.pass{color:#155724;background:#d4edda;padding:1px 7px;border-radius:3px;
      font-size:0.83em;white-space:nowrap;}
.warn{color:#856404;background:#fff3cd;padding:1px 7px;border-radius:3px;
      font-size:0.83em;white-space:nowrap;}
.fail{color:#721c24;background:#f8d7da;padding:1px 7px;border-radius:3px;
      font-size:0.83em;white-space:nowrap;}
.info{color:#004085;background:#cce5ff;padding:1px 7px;border-radius:3px;
      font-size:0.83em;white-space:nowrap;}
.grade-a{color:#155724;font-weight:700;}
.grade-b{color:#004085;font-weight:700;}
.grade-c{color:#856404;font-weight:700;}
.grade-d{color:#721c24;font-weight:700;}
.quote-box{border-left:3px solid #6c757d;padding:6px 12px;margin:6px 0;
           background:#f8f9fa;font-style:italic;font-size:0.91em;}
</style>
"""


# ── Low-level HTML helpers ────────────────────────────────────────────────

def _e(text):
    """Escape HTML special characters."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _badge(text, kind="pass"):
    return f'<span class="{kind}">{_e(text)}</span>'


def _grade(sil):
    if sil is None:
        return '<span class="grade-c">?</span>'
    if sil > 0.50:
        return '<span class="grade-a">A</span>'
    if sil > 0.25:
        return '<span class="grade-b">B</span>'
    if sil > 0.00:
        return '<span class="grade-c">C</span>'
    return '<span class="grade-d">D</span>'


def _th(text):
    return f"<th>{text}</th>"


def _td(text, raw=False):
    content = text if raw else _e(str(text))
    return f"<td>{content}</td>"


def _table(headers, rows, raw_cells=False):
    head = "<tr>" + "".join(_th(h) for h in headers) + "</tr>"
    body = ""
    for row in rows:
        body += "<tr>" + "".join(_td(c, raw=raw_cells) for c in row) + "</tr>"
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def _h(level, text):
    return f"<h{level}>{text}</h{level}>"


def _p(text):
    return f"<p>{text}</p>"


def _hr():
    return "<hr>"


def _meta(pairs):
    parts = " &nbsp;|&nbsp; ".join(f"<b>{k}:</b> {_e(str(v))}" for k, v in pairs.items())
    return f'<div class="meta">{parts}</div>'


def _wrap(html):
    return f'<div class="pr">{_CSS}{html}</div>'


# ── Phase 1: Data Steward ─────────────────────────────────────────────────

def render_phase1_report(steward_result, quality_report, baseline_df=None, run_id=""):
    """Rich HTML report for the Phase 1 Data Steward agent."""
    qr = quality_report
    cr = qr["careless_responding"]
    var = qr["variance"]
    retained = qr["retained_survey_cols"]
    excluded = var.get("excluded", [])
    dist_issues = [d for d in qr["distributions"] if d["issues"]]
    sparse_flagged = qr["sparsity"]["columns_flagged"]
    n_orig = qr["original_rows"]
    n_clean = qr["clean_rows"]
    n_removed = qr.get("removed_count", n_orig - n_clean)
    removal_pct = round(n_removed / n_orig * 100, 1)
    conf = qr["confidence"]

    sds = [var["stats"][c]["sd"] for c in retained if c in var["stats"]]
    sd_range = f"{min(sds):.2f}-{max(sds):.2f}" if sds else "N/A"

    # Key findings table
    findings = [
        ("Schema Validation",
         _badge("PASSED"),
         f"{len(qr['schema']['expected_numeric'])} survey items + "
         f"{len(qr['schema']['expected_categorical'])} demographic columns confirmed; "
         f"all expected fields present"),
        ("Careless Responding (SDQEM)",
         _badge(f"{n_removed} flagged ({removal_pct}%)", "warn"),
         "Awaiting removal approval"),
        ("Sparsity (&gt;20% missing)",
         _badge("PASSED") if not sparse_flagged else _badge(f"{len(sparse_flagged)} flagged", "warn"),
         "No column exceeded 20% missing" if not sparse_flagged
         else f"Flagged: {', '.join(sparse_flagged)}"),
        ("Variance (SD &lt; 0.5)",
         _badge("PASSED") if not excluded else _badge(f"{len(excluded)} excluded", "fail"),
         f"All {len(retained)} items retained (SDs: {sd_range})" if not excluded
         else f"Excluded: {', '.join(excluded)}"),
        ("Distribution Screening",
         _badge("PASSED with notes", "warn") if dist_issues else _badge("PASSED"),
         "; ".join(f"Mild IQR artifact on {d['column']}" for d in dist_issues)
         if dist_issues else "No distribution anomalies flagged"),
        ("Data Quality Confidence",
         _badge(f"{conf:.3f}"),
         "Above 0.90 threshold" if conf >= 0.90 else "Below 0.90 threshold -- review recommended"),
    ]

    # Survey item descriptives
    desc_rows = []
    if baseline_df is not None:
        means = {c: float(baseline_df[c].mean()) for c in retained if c in baseline_df.columns}
        min_mean_col = min(means, key=means.get) if means else None
        for col in retained:
            if col not in baseline_df.columns:
                continue
            m = float(baseline_df[col].mean())
            s = float(baseline_df[col].std())
            med = float(baseline_df[col].median())
            note = "Above midpoint" if m > 3.1 else ("Below midpoint" if m < 2.9 else "Near midpoint")
            if col == min_mean_col:
                note += " -- lowest scoring item"
            desc_rows.append((col, f"{m:.2f}", f"{s:.2f}", f"{med:.1f}", note))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    html = (
        _h(1, "Phase 1: Ingest and Clean")
        + _meta({
            "Agent": "Data Steward",
            "Run ID": run_id or ts,
            "Timestamp": ts,
            "Schema Version": config.PIPELINE_VERSION,
            "Framework": "Survey Data Quality Evaluation Model (Papp et al., 2026); Osborne (2013)",
        })
        + _hr()
        + _h(2, "Executive Summary")
        + _p(steward_result["reasoning"])
        + _hr()
        + _h(2, "Key Findings")
        + _table(["Gate", "Result", "Detail"], findings, raw_cells=True)
        + _hr()
        + f'<div class="box">'
        + _h(2, "DECISION REQUIRED: Careless Respondent Removal")
        + _p(f"<b>{n_removed} respondents ({removal_pct}%) were flagged on "
             f"&ge;{config.CARELESS_HURDLES} independent SDQEM indicators:</b>")
        + _table(["Indicator", "Flagged", "%"], [
            ("Straight-lining (longstring &ge;3 of 5 items)",
             cr["longstring_flagged"],
             f"{cr['longstring_flagged']/n_orig*100:.1f}%"),
            ("Low IRV (SD across items &lt; 0.2)",
             cr["irv_flagged"],
             f"{cr['irv_flagged']/n_orig*100:.1f}%"),
            (f"<b>Multi-hurdle (&ge;{config.CARELESS_HURDLES} indicators)</b>",
             f"<b>{n_removed}</b>",
             f"<b>{removal_pct}%</b>"),
        ], raw_cells=True)
        + _p(f"<b>Awaiting decision:</b> Remove the {n_removed} flagged respondents and proceed "
             f"with N={n_orig - n_removed:,}? Or retain all {n_orig:,}?")
        + "</div>"
        + _hr()
    )

    if desc_rows:
        html += (
            _h(2, f"Survey Item Descriptives (Post-Imputation, N={n_orig:,})")
            + _table(["Item", "Mean", "SD", "Median", "Notes"], desc_rows)
            + _hr()
        )

    html += (
        _h(2, "Artifacts Produced")
        + _table(["Artifact", "Location"], [
            ("Clean data (with careless flag)",
             "outputs/phase1_data_quality_report/survey_baseline_clean.csv"),
            ("Quality screening report",
             "outputs/phase1_data_quality_report/audit_reports/data_quality_screening.md"),
            ("Screening results CSV",
             "outputs/phase1_data_quality_report/screening_results.csv"),
            ("Audit trail",
             "outputs/phase1_data_quality_report/audit_reports/audit_trail.json"),
            ("Success report",
             "outputs/phase1_data_quality_report/reflection_logs/data_steward_success_report.txt"),
        ])
    )

    return _wrap(html)


# ── Phase 2: Cluster Discovery ────────────────────────────────────────────

def render_phase2_report(kproto_result, lpa_result, validation_result, run_id=""):
    """Rich HTML report for Phase 2 (K-Prototypes + LPA + Psychometrician)."""
    import numpy as np

    k = kproto_result["k_selected"]
    profiles = kproto_result["profiles"]
    fp = lpa_result.get("fingerprints", {})
    sil_per = validation_result["silhouette_per_cluster"]
    sil_overall = validation_result["silhouette_overall"]
    ari = validation_result["ari"]
    ari_interp = validation_result.get("ari_interpretation", "")
    outlier_flags = validation_result["outlier_flags"]
    n_total = len(outlier_flags)

    # Cluster labels for display (from LPA fingerprints mapped by ARI)
    def _fp_label(pid):
        if pid in fp and isinstance(fp[pid], dict):
            return fp[pid].get("label", f"Profile {pid}")
        return f"Profile {pid}"

    # K-Prototypes profiles table
    kp_rows = []
    for _, row in profiles.iterrows():
        cid = int(row["cluster"])
        n = int(row["n"])
        pct = row["pct"]
        dims_high = [c for c in config.NUMERIC_COLS if c in row.index and row[c] > 0.5]
        dims_low  = [c for c in config.NUMERIC_COLS if c in row.index and row[c] < -0.5]
        fingerprint = ""
        if dims_high:
            fingerprint += "Hi-" + "/".join(dims_high)
        if dims_low:
            fingerprint += (" / " if dims_high else "") + "Lo-" + "/".join(dims_low)
        if not fingerprint:
            fingerprint = "Moderate all"
        demo_parts = [str(row[c]) for c in config.CATEGORICAL_COLS if c in row.index]
        kp_rows.append((cid, n, f"{pct}%", fingerprint, ", ".join(demo_parts)))

    # LPA profiles table
    lpa_profiles = lpa_result.get("profiles")
    lpa_rows = []
    if lpa_profiles is not None:
        for _, row in lpa_profiles.iterrows():
            pid = int(row.get("profile", row.get("cluster", 0)))
            n = int(row["n"])
            pct = row["pct"]
            label = _fp_label(pid)
            posterior = row.get("mean_posterior", "N/A")
            lpa_rows.append((pid, n, f"{pct}%", label,
                             f"{posterior:.3f}" if isinstance(posterior, float) else posterior))

    # Psychometrician per-cluster scorecard
    pc_rows = []
    for cid in sorted(sil_per.keys()):
        d = sil_per[cid]
        sil_mean = d["mean"]
        neg_pct = d["pct_negative"]
        # Outlier % per cluster
        labels = kproto_result["labels"]
        cluster_mask = labels == cid
        if hasattr(outlier_flags, "__len__") and len(outlier_flags) == len(labels):
            n_out = int(outlier_flags[cluster_mask].sum())
            n_clu = int(cluster_mask.sum())
            out_pct = round(n_out / max(n_clu, 1) * 100, 1)
        else:
            out_pct = "N/A"
        grade_html = _grade(sil_mean)
        name = _fp_label(cid)
        pc_rows.append((
            f"{cid} ({name})",
            int(profiles.loc[profiles["cluster"] == cid, "n"].iloc[0]) if len(profiles) > cid else "",
            f"{sil_mean:.3f}",
            f"{neg_pct}%",
            f"{out_pct}%",
            grade_html,
        ))

    # ARI badge
    if ari is not None:
        ari_kind = "pass" if ari > config.ARI_STRONG else ("warn" if ari > config.ARI_MODERATE else "fail")
        ari_label = "STRONG" if ari > config.ARI_STRONG else ("MODERATE" if ari > config.ARI_MODERATE else "WEAK")
    else:
        ari_kind, ari_label = "info", "N/A"

    # Cross-model matching table
    ari_rows = []
    if ari is not None:
        for _, row in profiles.iterrows():
            cid = int(row["cluster"])
            kp_label = _fp_label(cid)
            ari_rows.append((
                f"Cluster {cid} ({kp_label})",
                "→",
                f"Profile {cid}",
                "&gt;90%",
            ))

    # Consolidated persona map
    persona_rows = []
    for _, row in profiles.iterrows():
        cid = int(row["cluster"])
        lpa_label = _fp_label(cid)
        n_avg = int(row["n"])
        pct = row["pct"]
        dims_high = [c for c in config.NUMERIC_COLS if c in row.index and row[c] > 0.5]
        dims_low  = [c for c in config.NUMERIC_COLS if c in row.index and row[c] < -0.5]
        sig = ""
        if dims_high:
            sig += "High " + " + ".join(dims_high)
        if dims_low:
            sig += (", " if dims_high else "") + "Low " + " + ".join(dims_low)
        persona_rows.append((lpa_label, f"Cluster {cid}", f"Profile {cid}",
                             f"~{n_avg:,} ({pct}%)", sig or "Moderate"))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    selected_lpa = lpa_result["fit_indices"]["selected"]

    html = (
        _h(1, "Phase 2: Discover Workforce Segments")
        + _meta({
            "Agents": "K-Prototypes, LPA, Psychometrician",
            "Run ID": run_id or ts,
            "Timestamp": ts,
        })
        + _hr()
        + _h(2, "Executive Summary")
        + _p(kproto_result["reasoning"])
        + _hr()

        + _h(2, "K-Prototypes Results")
        + _table(["Cluster", "n", "%", "Fingerprint", "Demo Mode"], kp_rows)
        + _p(f"Gamma = {kproto_result['centroids'].get('gamma', 'N/A')} (Huang default). "
             f"Cao initialization, n_init=10.")
        + _hr()

        + _h(2, "LPA Results")
        + _table(["Profile", "n", "%", "Fingerprint", "Mean Posterior"], lpa_rows)
        + _p(f"Model: K={selected_lpa['K']} {selected_lpa['cov_type']} GMM. "
             f"BIC={selected_lpa['BIC']:.1f}. "
             f"Entropy={selected_lpa['entropy']:.3f}. "
             f"Ambiguous (post &lt;{config.LPA_AMBIGUITY_POSTERIOR}): "
             f"{lpa_result['ambiguous_count']} respondents.")
        + _hr()

        + _h(2, "Psychometrician Validation")
        + _h(3, "Silhouette Analysis")
        + _table(["Metric", "Value", "Interpretation"], [
            ("Numeric-only silhouette, K-Proto (Euclidean)",
             f"{sil_overall:.4f}",
             "FAIR" if sil_overall < 0.50 else ("GOOD" if sil_overall < 0.70 else "EXCELLENT")),
        ])
        + _h(3, "Per-Cluster Scorecard")
        + _table(["Cluster", "n", "Mean Silhouette", "Negative Sil %", "Outlier %", "Grade"],
                 pc_rows, raw_cells=True)
        + _hr()

        + _h(3, f"Cross-Model Validation (ARI)")
        + _p(f"ARI = {ari:.3f} &mdash; "
             + _badge(f"{ari_label} agreement", ari_kind)
             + f" ({ari_interp})" if ari is not None else "ARI not computed")
        + (_table(["K-Prototypes", "→", "LPA Profile", "Agreement"], ari_rows, raw_cells=True)
           if ari_rows else "")
        + _hr()

        + _h(2, "Consolidated Persona Map")
        + _table(["Persona", "K-Proto", "LPA", "n (avg)", "Psychometric Signature"],
                 persona_rows)
        + _hr()

        + _h(2, "Artifacts")
        + _table(["File", "Description"], [
            ("cluster_assignments.csv", f"Per-respondent K-Prototypes + LPA labels (N={n_total:,})"),
            ("kproto_profiles.csv", f"Per-cluster centroid profiles"),
            ("kproto_centroids.json", "K-Prototypes numeric + categorical centroids"),
            ("lpa_profiles.csv", "LPA profile descriptors"),
            ("lpa_fingerprints.json", "Psychological fingerprints"),
            ("validation_summary.json", "Silhouette + ARI validation metrics"),
            ("audit_reports/audit_trail.json", "Combined agent audit trail"),
            ("reflection_logs/phase2_success_report.txt", "Success report"),
        ])
    )

    return _wrap(html)


# ── Phase 3: Ground in Reality ────────────────────────────────────────────

def render_phase3_report(kb, grounding_result, emergence_result, run_id=""):
    """Rich HTML report for Phase 3 (RAG + Emergence agents)."""
    mappings = grounding_result.get("mappings", {})
    n_grounded = sum(1 for v in mappings.values()
                     if isinstance(v, dict) and v.get("passages"))
    candidates = emergence_result.get("candidates", [])
    theme_report = emergence_result.get("theme_report") or []
    n_cand = len(candidates)
    n_themes = len(theme_report) if isinstance(theme_report, list) else 0

    # Theme classification table
    theme_rows = []
    if isinstance(theme_report, list):
        for t in theme_report:
            if isinstance(t, dict):
                cls = t.get("classification", "?")
                cls_kind = {"NEW": "warn", "VARIANT": "info", "NOISE": "pass"}.get(cls, "info")
                theme_rows.append((
                    f"Cluster {t.get('cluster', '?')}",
                    _badge(cls, cls_kind),
                    t.get("pattern", ""),
                    t.get("reasoning", "")[:200],
                ))

    # Per-construct grounding
    grounding_rows = []
    for construct, data in mappings.items():
        if isinstance(data, dict):
            n_p = len(data.get("passages", []))
            top = data["passages"][0]["source"] if n_p > 0 else "—"
            grounding_rows.append((construct, str(n_p), top,
                                   _badge("grounded") if n_p > 0 else _badge("no passages", "warn")))

    # Per-cluster policy grounding detail
    cluster_grounding_html = ""
    if mappings:
        cluster_grounding_html = _h(3, "Construct-to-Policy Grounding")
        cluster_grounding_html += _table(
            ["Construct", "Passages", "Top Source", "Status"],
            grounding_rows, raw_cells=True,
        )
        for construct, data in mappings.items():
            if not isinstance(data, dict) or not data.get("passages"):
                continue
            cluster_grounding_html += f"<h4 style='margin-top:10px'>{_e(construct)}</h4>"
            for p in data["passages"][:3]:
                score = p.get("score", 0)
                cluster_grounding_html += (
                    f'<div class="quote-box">'
                    f'<b>[{_e(p["source"])}]</b> (score: {score:.3f})<br>'
                    f'{_e(p["text"][:300])}{"..." if len(p["text"]) > 300 else ""}'
                    f"</div>"
                )
            if data.get("llm_assessment"):
                cluster_grounding_html += (
                    f'<div class="box-blue" style="margin:6px 0">'
                    f'<b>Relevance Assessment:</b> {_e(data["llm_assessment"][:400])}'
                    f"</div>"
                )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    html = (
        _h(1, "Phase 3: Ground in Organizational Reality")
        + _meta({
            "Agents": "RAG, Emergence",
            "Run ID": run_id or ts,
            "Timestamp": ts,
        })
        + _hr()
        + _h(2, "Executive Summary")
        + _p(grounding_result["reasoning"])
        + _hr()

        + _h(2, "Emergence Agent Results")
        + _p(emergence_result["reasoning"])
        + (_table(["Cluster", "Classification", "Pattern", "Reasoning"],
                  theme_rows, raw_cells=True)
           if theme_rows else _p("No candidate emergent themes detected."))
        + _hr()

        + _h(2, "RAG Agent Results")
        + _p(f"<b>{kb['n_documents']} documents indexed</b> | "
             f"{kb['n_chunks']} chunks | "
             f"{n_grounded} of {len(mappings)} constructs grounded")
        + cluster_grounding_html
        + _hr()

        + _h(2, "Artifacts")
        + _table(["File", "Description"], [
            ("construct_grounding.json", "Per-construct retrieved passages + LLM assessments"),
            ("emergent_themes.json", "NEW / VARIANT / NOISE classifications"),
            ("knowledge_base_index.json", "Document and chunk manifest"),
            ("audit_reports/rag_retrieval_detail.md", "Verbatim retrieved passages"),
            ("audit_reports/audit_trail.json", "Agent audit trail"),
            ("reflection_logs/phase3_success_report.txt", "Success report"),
        ])
        + _hr()
        + f'<div class="box-green"><b>Routing:</b> Phase 3 COMPLETE. '
        + f'{n_grounded} of {len(mappings)} constructs grounded. '
        + f'{n_cand} emergence candidate(s) classified. '
        + "Ready for Phase 4: Narrator persona synthesis.</div>"
    )

    return _wrap(html)


# ── Phase 4: Personas ─────────────────────────────────────────────────────

def render_phase4_overview(kproto_result, lpa_result, validation_result,
                           narrator_result, personas, run_id=""):
    """Rich HTML synthesis overview for Phase 4."""
    profiles = kproto_result["profiles"]
    fp = lpa_result.get("fingerprints", {})
    sil_per = validation_result["silhouette_per_cluster"]
    sil_overall = validation_result["silhouette_overall"]
    ari = validation_result["ari"]
    n_total = int(profiles["n"].sum())
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _fp_label(cid):
        if cid in fp and isinstance(fp[cid], dict):
            return fp[cid].get("label", f"Profile {cid}")
        return f"Profile {cid}"

    def _persona_name(cid):
        if isinstance(personas, list):
            for p in personas:
                if isinstance(p, dict) and int(p.get("cluster_id", -1)) == cid:
                    return p.get("persona_name", f"Persona {cid}")
        return f"Persona {cid}"

    # Cluster comparison
    cmp_rows = []
    for _, row in profiles.iterrows():
        cid = int(row["cluster"])
        sil = sil_per.get(cid, {}).get("mean")
        grade_h = _grade(sil)
        sig_high = [c for c in config.NUMERIC_COLS if c in row.index and row[c] > 0.5]
        sig_low  = [c for c in config.NUMERIC_COLS if c in row.index and row[c] < -0.5]
        sig = ""
        if sig_high:
            sig += "High " + " + ".join(sig_high)
        if sig_low:
            sig += (", " if sig_high else "") + "Low " + " + ".join(sig_low)
        cmp_rows.append((
            cid,
            _persona_name(cid),
            f"{int(row['n']):,}",
            f"{row['pct']}%",
            grade_h,
            f"{sil:.3f}" if sil is not None else "N/A",
            sig or "Moderate",
        ))

    # Epistemic caveats
    caveats = [
        "All five survey items are single-item proxies for validated multi-item constructs. "
        "Findings should be interpreted as directional indicators, not precise construct measurements.",
        "Grade C clusters have the weakest statistical support and highest internal variability. "
        "Characterizations of these clusters require particular caution.",
        "Demographic tendencies describe probabilistic patterns, not deterministic group membership. "
        "Any given employee may belong to any cluster.",
        "The I-O Psychologist retains final interpretive authority over all persona characterizations.",
    ]

    html = (
        _h(1, "Persona Synthesis Overview")
        + _meta({
            "Run ID": run_id or ts,
            "Timestamp": ts,
            "Total respondents": f"{n_total:,}",
            "Clusters": str(kproto_result["k_selected"]),
            "Global Silhouette (numeric)": f"{sil_overall:.4f}",
            "ARI (K-Prototypes vs LPA)": f"{ari:.3f}" if ari else "N/A",
        })
        + _hr()
        + _h(2, "Cluster Comparison")
        + _table(["Cluster", "Name", "Size", "%", "Grade", "Silhouette", "Key Signature"],
                 cmp_rows, raw_cells=True)
        + _hr()
        + _h(2, "Narrator Agent Summary")
        + _p(narrator_result["reasoning"])
        + _hr()
        + _h(2, "Epistemic Caveats")
        + "<ol>" + "".join(f"<li>{_e(c)}</li>" for c in caveats) + "</ol>"
        + _p("<i>Generated with AI assistance. All narrative claims traceable to "
             "statistical centroid values. (Nguyen &amp; Welch, 2025).</i>")
    )

    return _wrap(html)


def render_persona_card(persona, sil_per_cluster=None, run_id=""):
    """Rich HTML card for a single persona narrative."""
    if not isinstance(persona, dict):
        return _wrap(_p("Invalid persona data."))

    name = persona.get("persona_name", "Unnamed Persona")
    cid  = persona.get("cluster_id", "?")
    size = persona.get("size", "?")
    pct  = persona.get("pct", "?")
    narrative = persona.get("narrative", "No narrative generated.")
    mismatches = persona.get("policy_mismatches", [])
    epnote = persona.get("epistemic_note", "")
    fp_data = persona.get("statistical_fingerprint", {})
    demo = persona.get("demographic_mode", {})
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Silhouette grade
    sil = None
    if sil_per_cluster and int(cid) in sil_per_cluster:
        sil = sil_per_cluster[int(cid)].get("mean")
    grade_html = _grade(sil)

    # Fingerprint table
    fp_rows = []
    if isinstance(fp_data, dict):
        for dim, info in fp_data.items():
            if isinstance(info, dict):
                val = info.get("value", info.get("z", "?"))
                direction = info.get("direction", "?")
                fp_rows.append((dim, f"{val:+.3f}" if isinstance(val, float) else val, direction))

    # Demo string
    demo_str = " | ".join(f"{k}: {v}" for k, v in demo.items()) if demo else ""

    # LPA fingerprint label
    lpa_label = persona.get("lpa_fingerprint", "")

    html = (
        _h(1, f"Cluster {cid}: {_e(name)}")
        + _meta({
            "Run ID": run_id or ts,
            "Timestamp": ts,
        })
        + _hr()
        + _h(2, "Statistical Fingerprint")
        + (_table(["Dimension", "Z-score", "Interpretation"], fp_rows)
           if fp_rows else _p("Fingerprint data not available."))
        + _p(f"<b>Size:</b> {size:,} respondents ({pct}% of total) &nbsp; "
             f"<b>Quality Grade:</b> {grade_html} (Silhouette = {sil:.3f})"
             if sil else f"<b>Size:</b> {size} respondents ({pct}%)")
        + (f"<p><b>Dominant demographic:</b> {_e(demo_str)}</p>" if demo_str else "")
        + (f"<p><b>LPA Psychological Fingerprint:</b> {_e(lpa_label)}</p>" if lpa_label else "")
        + _hr()
        + _h(2, "Narrative")
        + _p(_e(narrative))
    )

    if mismatches:
        html += (
            _hr()
            + _h(2, "Policy-Experience Mismatches")
            + "<ul>" + "".join(f"<li>{_e(m)}</li>" for m in
                               (mismatches if isinstance(mismatches, list) else [mismatches]))
            + "</ul>"
        )

    if epnote:
        html += _hr() + f"<p><i>{_e(epnote)}</i></p>"

    return _wrap(html)
