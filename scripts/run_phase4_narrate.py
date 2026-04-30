"""
Phase 4 Persona Narrative Generation for the SIOP 2026 pipeline.

Synthesises cluster profiles, LPA fingerprints, and RAG policy
context into evidence-grounded persona narratives.  Every claim in
a generated narrative must trace to a centroid value or retrieved
passage; the Narrator applies the epistemic risk mitigation
protocol from Nguyen & Welch (2025) to guard against anthropomorphic
interpretation, fabricated quotations, and the Oracle Effect.

Phases 2 and 3 must have run first.  This script reads:
    * kproto_profiles.csv        (cluster centroids + modes)
    * lpa_fingerprints.json      (profile signatures)
    * construct_grounding.json   (per-construct policy context)

Run (from project root):
    python scripts/run_phase4_narrate.py
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.p4_narrate.narrator import generate_personas

PHASE2_DIR = REPO_ROOT / "outputs" / "phase2_cluster_validation"
PHASE3_DIR = REPO_ROOT / "outputs" / "phase3_emergent_themes"

KPROTO_PROFILES = PHASE2_DIR / "kproto_profiles.csv"
LPA_FINGERPRINTS = PHASE2_DIR / "lpa_fingerprints.json"
GROUNDING = PHASE3_DIR / "construct_grounding.json"

OUT_DIR = REPO_ROOT / "outputs" / "phase4_persona_narratives"
AUDIT_DIR = OUT_DIR / "audit_reports"
REFLECT_DIR = OUT_DIR / "reflection_logs"

RUN_ID = str(uuid.uuid4())
TIMESTAMP = datetime.now(timezone.utc).isoformat()


def _render_persona_md(personas: list, mock_mode: bool,
                       has_policy: bool) -> str:
    mode = "MOCK" if mock_mode else "LIVE"
    lines = [
        "# Phase 4: Write and Validate Personas",
        "",
        "**Agents:** Narrator + Ethics Checkpoint + Project Manager Governance",
        "**Subtitle:** Evidence-grounded narratives. Ethics checkpoint "
        "required before approval.",
        "",
        f"- **Run ID:** `{RUN_ID}`",
        f"- **Timestamp:** {TIMESTAMP}",
        f"- **LLM mode:** {mode}",
        f"- **Personas generated:** {len(personas)}",
        f"- **Policy context (RAG grounding):** "
        f"{'attached' if has_policy else 'not attached'}",
        "",
        "## Narrative Principles (Braun & Clarke, 2006)",
        "",
        "- Every claim traces to a statistical centroid or a retrieved "
        "policy passage",
        "- Quotes are verbatim from respondent data (no paraphrasing)",
        "- Quotes span different dimensions (not just the most extreme or "
        "eloquent)",
        "- Confidence levels stated (z-scores indicate evidence strength)",
        "- Anthropomorphic language flagged for human review",
        "- The I-O psychologist retains final interpretive authority",
        "",
        "## Epistemic Risk Mitigation (Nguyen & Welch, 2025)",
        "",
        "The Narrator follows a strict protocol to prevent three common "
        "failure modes:",
        "",
        "1. **Anthropomorphic interpretation** -- treating a cluster as a "
        "person with desires and intentions rather than a statistical "
        "abstraction.",
        "2. **Fabricated quotations** -- inventing representative quotes that "
        "sound right. The Narrator quotes only from respondent data.",
        "3. **The Oracle Effect** -- masking uncertainty with confident "
        "language. The narrative must state its uncertainty explicitly.",
        "",
        "## Gate 4: Your Decision",
        "",
        "> **Do you approve personas for presentation?**",
        "> Do narratives match statistical fingerprints? Are they "
        "evidence-based or over-interpreted? Would an employee be comfortable "
        "if they recognized their group?",
        "",
        "**Prerequisite:** The ethics checkpoint in "
        "`audit_reports/ethics_checkpoint.md` must be completed (all six "
        "bias audits acknowledged) before Gate 4 unlocks.",
        "",
        "**Evidence Summary**",
        "",
        "- [ ] Every claim in every narrative traces to a centroid value or "
        "retrieved passage",
        "- [ ] Persona names are neutral and non-stigmatising",
        "- [ ] Epistemic notes are present on every persona",
        "- [ ] Policy-experience mismatches reflect real organizational "
        "context, not LLM speculation",
        "- [ ] No fabricated quotes, no imputed emotions beyond what scores "
        "support",
        "- [ ] You would be comfortable standing behind each narrative in "
        "front of leadership",
        "",
        "| Option | When to choose | What happens |",
        "|---|---|---|",
        "| Approve | Ethics checkpoint clean; narratives defensible | "
        "Personas ready for leadership presentation |",
        "| Revise | One or two narratives overreach or misname | Revise the "
        "specific narratives; re-approve |",
        "| Reject | A bias audit failed; the narrative frame is unsound | "
        "Return to Phase 3 to reconsider grounding |",
        "",
        "## Persona Narratives",
        "",
    ]
    for p in personas:
        if not isinstance(p, dict):
            continue
        name = p.get("persona_name", f"Cluster {p.get('cluster_id', '?')}")
        cid = p.get("cluster_id", "?")
        size = p.get("size", "?")
        pct = p.get("pct", "?")
        pct_str = f"{pct:.1f}%" if isinstance(pct, (int, float)) else str(pct)
        lines.append(f"### {name}  (Cluster {cid}, n={size}, {pct_str})")
        lines.append("")

        fingerprint = p.get("statistical_fingerprint")
        if fingerprint:
            lines.append("**Psychometric Fingerprint**")
            lines.append("")
            if isinstance(fingerprint, dict):
                lines.append("| Indicator | Value | Direction |")
                lines.append("|---|---:|---|")
                for ind, v in fingerprint.items():
                    if isinstance(v, dict):
                        val = v.get("value", "")
                        dirn = v.get("direction", "")
                        val_str = (
                            f"{val:+.2f}"
                            if isinstance(val, (int, float))
                            else str(val)
                        )
                        lines.append(f"| {ind} | {val_str} | {dirn} |")
                    else:
                        lines.append(f"| {ind} | {v} |  |")
            else:
                # Some LLM responses return the fingerprint as a single
                # prose sentence rather than a per-indicator dict.
                lines.append(str(fingerprint))
            lines.append("")

        demo = p.get("demographic_mode")
        if demo:
            demo_text = ", ".join(f"{k}={v}" for k, v in demo.items())
            lines.append(f"**Modal demographics:** {demo_text}")
            lines.append("")

        narrative = p.get("narrative")
        if narrative:
            lines.append("**Narrative**")
            lines.append("")
            lines.append(narrative)
            lines.append("")

        mismatches = p.get("policy_mismatches") or []
        if mismatches:
            lines.append("**Policy-experience mismatches**")
            lines.append("")
            for m in mismatches:
                lines.append(f"- {m}")
            lines.append("")

        epi = p.get("epistemic_note")
        if epi:
            lines.append(f"> *{epi}*")
            lines.append("")

    lines += [
        "## Artifacts Produced",
        "",
        "1. `personas.md` -- this report (Gate 4 deliverable)",
        "2. `personas.json` -- structured personas for downstream use",
        "3. `personas.csv` -- one row per persona with fingerprint columns",
        "4. `audit_reports/ethics_checkpoint.md` -- six-dimension bias audit "
        "worksheet (must be acknowledged before Gate 4 is approved)",
        "5. `audit_reports/audit_trail.json` -- Narrator audit entries",
        "6. `reflection_logs/phase4_success_report.txt` -- status, metrics, "
        "artifacts produced",
        "",
    ]
    return "\n".join(lines)


def _render_ethics_checkpoint() -> str:
    """Six-dimension ethics audit worksheet (matches the React artifact).

    Gate 4 approval in the artifact is locked until the reviewing
    I-O psychologist has acknowledged each of these six bias types.
    The markdown version here is a pre-review worksheet with the
    same structure and questions.
    """
    bias_types = [
        ("INPUT BIAS",
         "Survey representation & demographic balance",
         [
             "Response rate by demographic (all groups >= 70%?)",
             "No demographic group oversampled in any cluster?",
             "Removal rates during Phase 1 were unbiased?",
         ]),
        ("CLUSTERING BIAS",
         "Do clusters map onto protected characteristics?",
         [
             "Cluster membership independent of demographics? (chi-square)",
             'No cluster = demographic group (e.g., Cluster 1 != "women")?',
             "Silhouette analysis shows fair separation across all groups?",
         ]),
        ("NARRATIVE BIAS",
         "Stereotyping or flattening nuance",
         [
             'No stereotypical language ("Millennials are...")?',
             "Claims grounded in data (z-scores, percentages)?",
             "Would employees recognize themselves fairly in narratives?",
         ]),
        ("RETRIEVAL BIAS",
         "Organizational documents themselves biased",
         [
             "RAG corpus includes exec AND frontline voices?",
             "Missing perspectives noted (e.g., union, critics)?",
             "Policy language checked for inclusive vs. exclusive phrasing?",
         ]),
        ("EPISTEMIC RISK",
         "Overconfident or uncertain claims",
         [
             'Low-confidence claims (z < 1.0) flagged as "tentative"?',
             "Oracle Effect present (treating AI output as ground truth)?",
             "Recommendations marked with confidence levels?",
         ]),
        ("ANTHROPOMORPHISM",
         "Treating groups as intentional agents",
         [
             'No "want," "feel," "desire" without evidence?',
             "Observable behaviors described, not inferred intentions?",
             "Correlation vs. causation clearly marked?",
         ]),
    ]

    lines = [
        "# Ethics Checkpoint (Required)",
        "",
        "Before approving personas for leadership, audit for bias across six "
        "dimensions. Check each box as you review. Gate 4 approval requires "
        "all six dimensions acknowledged.",
        "",
    ]
    for i, (title, subtitle, checks) in enumerate(bias_types, 1):
        lines.append(f"## {i}. {title}")
        lines.append(f"*{subtitle}*")
        lines.append("")
        for c in checks:
            lines.append(f"- [ ] {c}")
        lines.append("")
    lines += [
        "## Acknowledgement",
        "",
        "- [ ] I have reviewed all six bias dimensions above",
        "- [ ] All issues raised have been resolved or explicitly noted in "
        "the audit trail",
        "- [ ] I am comfortable presenting these personas to leadership",
        "",
        "Reviewer: ____________________________    Date: _______________",
    ]
    return "\n".join(lines)


def _render_success_report(personas: list, mock_mode: bool,
                           has_policy: bool,
                           has_fingerprints: bool) -> str:
    mode = "MOCK" if mock_mode else "LIVE"
    status = (
        f"PENDING GATE 4 -- Ethics checkpoint required "
        f"({mode} mode, {len(personas)} personas generated)"
    )
    metrics = [
        ("LLM mode", mode),
        ("Personas generated", len(personas)),
        ("Policy context (RAG grounding)", "attached" if has_policy else "not attached"),
        ("LPA fingerprints attached", "yes" if has_fingerprints else "no"),
        ("Narrative principles applied",
         "Braun & Clarke (2006); Nguyen & Welch (2025) epistemic risk mitigation"),
    ]
    artifacts = [
        "personas.md -- Gate 4 deliverable",
        "personas.json -- structured personas",
        "personas.csv -- flat table",
        "audit_reports/ethics_checkpoint.md -- six-dimension bias audit",
        "audit_reports/audit_trail.json",
        "reflection_logs/phase4_success_report.txt -- this report",
    ]
    notes = (
        "Each persona pairs a Psychometric Fingerprint (z-scored cluster "
        "statistics from Phase 2), verbatim quotes, policy citations from "
        "Phase 3 RAG retrieval, and risk flags for epistemic uncertainty "
        "and anthropomorphic language. Gate 4 approval in the tutorial "
        "artifact is locked until all six bias audits are acknowledged by "
        "the I-O psychologist. The narrative frame deliberately does not "
        "attribute motivations or emotions to clusters beyond what the "
        "z-scores and quotes support."
    )
    lines = [
        "# Agent Success Report",
        "",
        "**Agents:** Narrator + Ethics Checkpoint + Project Manager Governance",
        "**Phase:** Phase 4 -- Write and Validate Personas",
        f"**Status:** {status}",
        f"**Timestamp:** {TIMESTAMP}",
        f"**Run ID:** {RUN_ID}",
        "",
        "## Metrics",
    ]
    lines.extend(f"- **{k}:** {v}" for k, v in metrics)
    lines += ["", "## Artifacts Produced"]
    lines.extend(f"{i}. {a}" for i, a in enumerate(artifacts, 1))
    lines += ["", "## Notes", notes]
    return "\n".join(lines)


def _flatten_personas_csv(personas: list) -> pd.DataFrame:
    rows = []
    for p in personas:
        if not isinstance(p, dict):
            continue
        row = {
            "cluster_id": p.get("cluster_id"),
            "persona_name": p.get("persona_name"),
            "size": p.get("size"),
            "pct": p.get("pct"),
            "narrative": p.get("narrative"),
            "policy_mismatches": " | ".join(p.get("policy_mismatches") or []),
            "epistemic_note": p.get("epistemic_note"),
        }
        fp = p.get("statistical_fingerprint")
        if isinstance(fp, dict):
            for ind, v in fp.items():
                if isinstance(v, dict):
                    row[f"{ind}_value"] = v.get("value")
                    row[f"{ind}_direction"] = v.get("direction")
        elif fp is not None:
            # LLM returned the fingerprint as a single prose string --
            # preserve it verbatim so the CSV still reflects what the
            # model produced.
            row["fingerprint_text"] = str(fp)
        demo = p.get("demographic_mode")
        if isinstance(demo, dict):
            for k, v in demo.items():
                row[f"mode_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 60)
    print("  PHASE 4: PERSONA NARRATIVE GENERATION")
    print("=" * 60)
    print(f"Run ID:    {RUN_ID}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"LLM mode:  {'MOCK' if config.MOCK_MODE else 'LIVE'}")

    if not KPROTO_PROFILES.exists():
        print(f"ERROR: Phase 2 profiles not found at {KPROTO_PROFILES}")
        print("Run: python scripts/run_phase2_discover.py first.")
        return 2

    for d in (OUT_DIR, AUDIT_DIR, REFLECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    cluster_profiles = pd.read_csv(KPROTO_PROFILES)
    print(f"Loaded {len(cluster_profiles)} cluster profiles")

    construct_scores = None
    if LPA_FINGERPRINTS.exists():
        construct_scores = json.loads(LPA_FINGERPRINTS.read_text(encoding="utf-8"))
        print(f"Loaded LPA fingerprints from Phase 2")

    policy_context = None
    if GROUNDING.exists():
        policy_context = json.loads(GROUNDING.read_text(encoding="utf-8"))
        print(f"Loaded construct grounding from Phase 3")
    else:
        print("Grounding not found -- narratives will lack policy context.")

    # ---- Narrator ----
    print("\n-- Generating personas --")
    narrator_out = generate_personas(
        cluster_profiles,
        construct_scores=construct_scores,
        policy_context=policy_context,
        codebook=list(config.NUMERIC_COLS),
    )
    personas = narrator_out["personas"]
    if not isinstance(personas, list):
        personas = [personas]
    print(f"  Personas generated: {len(personas)}")

    # ---- Persist ----
    personas_json = OUT_DIR / "personas.json"
    personas_json.write_text(json.dumps(personas, indent=2, default=str))
    print(f"\nWrote: {personas_json.relative_to(REPO_ROOT)}")

    personas_csv = OUT_DIR / "personas.csv"
    _flatten_personas_csv(personas).to_csv(personas_csv, index=False)
    print(f"Wrote: {personas_csv.relative_to(REPO_ROOT)}")

    personas_md = OUT_DIR / "personas.md"
    personas_md.write_text(
        _render_persona_md(personas, config.MOCK_MODE, bool(policy_context)),
        encoding="utf-8",
    )
    print(f"Wrote: {personas_md.relative_to(REPO_ROOT)}")

    audit_path = AUDIT_DIR / "audit_trail.json"
    audit_path.write_text(
        json.dumps(narrator_out["audit_entries"], indent=2, default=str)
    )
    print(f"Wrote: {audit_path.relative_to(REPO_ROOT)}")

    ethics_path = AUDIT_DIR / "ethics_checkpoint.md"
    ethics_path.write_text(_render_ethics_checkpoint(), encoding="utf-8")
    print(f"Wrote: {ethics_path.relative_to(REPO_ROOT)}")

    success_path = REFLECT_DIR / "phase4_success_report.txt"
    success_path.write_text(
        _render_success_report(
            personas, mock_mode=config.MOCK_MODE,
            has_policy=bool(policy_context), has_fingerprints=bool(construct_scores),
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {success_path.relative_to(REPO_ROOT)}")

    # ---- Success report ----
    print("\n" + "=" * 60)
    print("  PHASE 4 -- SUCCESS REPORT")
    print("=" * 60)
    print(f"Status:              COMPLETE")
    print(f"LLM mode:            {'MOCK' if config.MOCK_MODE else 'LIVE'}")
    print(f"Personas generated:  {len(personas)}")
    print(f"Policy context:      {'yes' if policy_context else 'no'}")
    print(f"LPA fingerprints:    {'yes' if construct_scores else 'no'}")
    print(f"\nNext: open outputs/phase4_persona_narratives/personas.md "
          f"for the Gate 4 review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
