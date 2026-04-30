"""
Phase 3 Grounding in Organizational Reality for the SIOP 2026 pipeline.

Runs the two Grounding agents in sequence:

    RAG Agent       -- Lewis et al. (2020): TF-IDF knowledge base
                       over synthetic_data/org_documents/, plus
                       per-construct passage retrieval and (in live
                       mode) LLM relevance assessment.
    Emergence Agent -- Glaser & Strauss (2017); Braun & Clarke (2006):
                       cross-sectional scan of cluster profiles for
                       patterns outside the codebook, classified as
                       NEW / VARIANT / NOISE.

Phase 2 must have run first; this script reads the K-Prototypes
profiles from ``outputs/phase2_cluster_validation/kproto_profiles.csv``.

Run (from project root):
    python scripts/run_phase3_ground.py
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
from src.p3_ground.rag import build_knowledge_base, ground_constructs
from src.p3_ground.emergence import detect_emergent_themes

PHASE2_DIR = REPO_ROOT / "outputs" / "phase2_cluster_validation"
KPROTO_PROFILES = PHASE2_DIR / "kproto_profiles.csv"

OUT_DIR = REPO_ROOT / "outputs" / "phase3_emergent_themes"
AUDIT_DIR = OUT_DIR / "audit_reports"
REFLECT_DIR = OUT_DIR / "reflection_logs"

RUN_ID = str(uuid.uuid4())
TIMESTAMP = datetime.now(timezone.utc).isoformat()


def _render_report(summary: dict) -> str:
    rag = summary["rag"]
    em = summary["emergence"]
    mode = "LIVE"

    n_with = sum(
        1 for v in rag["mappings"].values()
        if isinstance(v, dict) and v.get("passages")
    )

    lines = [
        "# Phase 3: Ground in Organizational Reality",
        "",
        "**Agents:** RAG, Emergence",
        "**Subtitle:** Clusters are statistical abstractions. To become "
        "useful to leadership, they must be grounded in organizational context.",
        "",
        f"- **Run ID:** `{summary['run_id']}`",
        f"- **Timestamp:** {summary['timestamp']}",
        f"- **LLM mode:** {mode}",
        "",
        "## Metrics",
        "",
        f"- **Documents indexed:** {rag['n_documents']}",
        f"- **Text chunks:** {rag['n_chunks']}",
        f"- **Constructs grounded (with passages):** {n_with} of "
        f"{len(rag['mappings'])}",
        f"- **Emergence candidates identified:** {em['n_candidates']}",
        "",
        "## Gate 3: Your Decision",
        "",
        "> **Do you accept the emergent themes and codebook expansion?**",
        "",
        "**Evidence Summary**",
        "",
        "- [ ] Every construct has at least one policy passage with "
        "defensible relevance (or missing constructs are flagged)",
        "- [ ] Policy-experience mismatches are surfaced for Phase 4 use",
        "- [ ] NEW themes have evidence of > 15% cluster frequency",
        "- [ ] VARIANT themes map to an existing construct",
        "- [ ] NOISE themes are genuinely idiosyncratic, not a misclassification",
        "- [ ] Any NEW theme is evidence-based, not stereotype",
        "",
        "| Option | When to choose | What happens |",
        "|---|---|---|",
        "| Accept & Expand Codebook | NEW themes defensible and worth "
        "tracking in future waves | NEW themes added; personas route to "
        "Narrator |",
        "| Request Review | You want another I-O psychologist's read before "
        "adding a construct | Flag for review; proceed with existing codebook |",
        "| Revise | A classification is wrong (e.g., NOISE should be VARIANT) "
        "| Re-run Phase 3 after fixing inputs |",
        "",
        "## RAG Construct Grounding",
        "",
    ]
    for construct, entry in rag["mappings"].items():
        lines.append(f"### {construct}")
        passages = entry.get("passages", []) if isinstance(entry, dict) else []
        if not passages:
            lines.append("- *No relevant passages retrieved.*")
            lines.append("")
            continue
        for p in passages:
            src = p.get("source", "?")
            score = p.get("score", 0.0)
            relevance = p.get("relevance", "")
            snippet = (p.get("text", "") or "")[:240].replace("\n", " ")
            tag = f" ({relevance})" if relevance else ""
            lines.append(f"- **{src}** [score={score:.3f}]{tag}: {snippet}...")
        assessment = entry.get("llm_assessment")
        if assessment:
            lines.append("")
            lines.append(f"*Assessment:* {assessment}")
        lines.append("")

    lines += [
        "## Emergence scan",
        "",
        f"- **Candidate patterns identified:** {em['n_candidates']}",
        "",
    ]
    if em["candidates"]:
        lines.append("| Cluster | Pattern |")
        lines.append("|---:|---|")
        for c in em["candidates"]:
            lines.append(f"| {c.get('cluster', '?')} | {c.get('pattern', '')} |")
        lines.append("")

    theme_report = em.get("theme_report")
    if theme_report:
        lines.append("### Theme classifications")
        lines.append("")
        if isinstance(theme_report, list):
            lines.append(
                "| Cluster | Classification | Pattern | Confidence | Reasoning |"
            )
            lines.append("|---:|---|---|---:|---|")
            for t in theme_report:
                cid = t.get("cluster", "?")
                clas = t.get("classification", "?")
                pat = t.get("pattern", "")
                conf = t.get("confidence", "")
                conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
                reason = (t.get("reasoning", "") or "").replace("\n", " ")
                lines.append(f"| {cid} | {clas} | {pat} | {conf_str} | {reason} |")
        else:
            lines.append("```")
            lines.append(json.dumps(theme_report, indent=2, default=str))
            lines.append("```")
        lines.append("")

    lines += [
        "## Agent reasoning",
        "",
        f"**RAG:** {rag['reasoning']}",
        "",
        f"**Emergence:** {em['reasoning']}",
        "",
        "## Artifacts Produced",
        "",
        "1. `construct_grounding.json` -- per-construct retrieved passages "
        "+ LLM relevance assessment",
        "2. `emergent_themes.json` -- candidate patterns + NEW/VARIANT/NOISE "
        "classification",
        "3. `knowledge_base_index.json` -- document / chunk manifest",
        "4. `audit_reports/rag_retrieval_detail.md` -- full retrieved "
        "passages for every construct",
        "5. `audit_reports/audit_trail.json` -- every agent action with "
        "timestamp",
        "6. `reflection_logs/phase3_success_report.txt` -- status, metrics, "
        "artifacts produced",
        "",
    ]
    return "\n".join(lines)


def _render_retrieval_detail(mappings: dict) -> str:
    """Verbatim RAG passages, one section per construct."""
    lines = ["# RAG Retrieval Detail", "",
             "Verbatim retrieved passages for each codebook construct, "
             "with cosine-similarity scores and (when available) the LLM's "
             "relevance rating. Used during Gate 3 review to verify that "
             "grounding is defensible and not the product of false-positive "
             "term matches.", ""]
    for construct, entry in mappings.items():
        lines.append(f"## {construct}")
        lines.append("")
        if not isinstance(entry, dict):
            lines.append(f"*{entry}*")
            lines.append("")
            continue
        passages = entry.get("passages") or []
        if not passages:
            lines.append("*No passages retrieved above threshold.*")
            lines.append("")
            continue
        for p in passages:
            src = p.get("source", "?")
            score = p.get("score", 0.0)
            relevance = p.get("relevance", "")
            text = (p.get("text") or "").strip()
            tag = f" [{relevance}]" if relevance else ""
            lines.append(f"**`{src}`** (cosine = {score:.3f}){tag}")
            lines.append("")
            lines.append(f"> {text}")
            lines.append("")
        assessment = entry.get("llm_assessment")
        if assessment:
            lines.append(f"*LLM assessment:* {assessment}")
            lines.append("")
    return "\n".join(lines)


def _render_success_report(summary: dict) -> str:
    rag = summary["rag"]
    em = summary["emergence"]
    mode = "LIVE"
    n_with = sum(
        1 for v in rag["mappings"].values()
        if isinstance(v, dict) and v.get("passages")
    )

    metrics = [
        ("LLM mode", mode),
        ("Documents indexed", rag["n_documents"]),
        ("Text chunks", rag["n_chunks"]),
        ("Constructs grounded (with passages)",
         f"{n_with} of {len(rag['mappings'])}"),
        ("Emergence candidates identified", em["n_candidates"]),
    ]
    if isinstance(em.get("theme_report"), list):
        by_class = {}
        for t in em["theme_report"]:
            c = t.get("classification", "?")
            by_class[c] = by_class.get(c, 0) + 1
        if by_class:
            metrics.append(
                ("Theme classification",
                 ", ".join(f"{k}: {v}" for k, v in sorted(by_class.items())))
            )

    artifacts = [
        "construct_grounding.json -- per-construct retrieved passages + "
        "LLM relevance assessment",
        "emergent_themes.json -- NEW / VARIANT / NOISE classifications",
        "knowledge_base_index.json -- document / chunk manifest",
        "audit_reports/rag_retrieval_detail.md -- verbatim retrieved passages",
        "audit_reports/audit_trail.json",
        "reflection_logs/phase3_success_report.txt -- this report",
    ]
    notes = (
        "TF-IDF + cosine similarity is used in place of dense embeddings to "
        "keep tutorial dependencies light (Lewis et al., 2020, describes the "
        "dense-embedding production pattern). Emergence classification follows "
        "Braun & Clarke (2006) thematic analysis: NEW themes warrant codebook "
        "addition, VARIANT themes warrant sub-labels, NOISE themes are "
        "statistical artifacts. LLM relevance ratings and theme classifications "
        "are produced live by the Anthropic API."
    )
    status = f"SUCCESS -- Grounding complete ({mode} mode)"

    lines = [
        "# Agent Success Report",
        "",
        "**Agents:** RAG, Emergence",
        "**Phase:** Phase 3 -- Ground in Organizational Reality",
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
    print("  PHASE 3: GROUNDING IN ORGANIZATIONAL REALITY")
    print("=" * 60)
    print(f"Run ID:    {RUN_ID}")
    print(f"Timestamp: {TIMESTAMP}")
    print("LLM mode:  LIVE")

    if not KPROTO_PROFILES.exists():
        print(f"ERROR: Phase 2 profiles not found at {KPROTO_PROFILES}")
        print("Run: python scripts/run_phase2_discover.py first.")
        return 2

    for d in (OUT_DIR, AUDIT_DIR, REFLECT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    kproto_profiles = pd.read_csv(KPROTO_PROFILES)
    print(f"Loaded {len(kproto_profiles)} cluster profiles from Phase 2")

    # ---- RAG: build KB and ground constructs ----
    print("\n-- Building knowledge base from org_documents/ --")
    kb = build_knowledge_base(str(REPO_ROOT / config.ORG_DOCS_DIR))
    print(f"  Documents     : {kb['n_documents']}")
    print(f"  Chunks        : {kb['n_chunks']}")

    print("\n-- Grounding codebook constructs --")
    rag_out = ground_constructs(kb, codebook=list(config.NUMERIC_COLS))
    print(f"  Constructs grounded: {len(rag_out['mappings'])}")

    # ---- Emergence: cross-sectional scan ----
    print("\n-- Emergence scan over cluster profiles --")
    em_out = detect_emergent_themes(
        kproto_profiles, codebook_constructs=list(config.NUMERIC_COLS),
    )
    print(f"  Candidates identified: {len(em_out['candidates'])}")

    # ---- Persist ----
    grounding_path = OUT_DIR / "construct_grounding.json"
    grounding_path.write_text(
        json.dumps(rag_out["mappings"], indent=2, default=str)
    )
    print(f"\nWrote: {grounding_path.relative_to(REPO_ROOT)}")

    themes_path = OUT_DIR / "emergent_themes.json"
    themes_path.write_text(json.dumps({
        "candidates": em_out["candidates"],
        "theme_report": em_out["theme_report"],
    }, indent=2, default=str))
    print(f"Wrote: {themes_path.relative_to(REPO_ROOT)}")

    kb_index_path = OUT_DIR / "knowledge_base_index.json"
    kb_index_path.write_text(json.dumps({
        "n_documents": kb["n_documents"],
        "n_chunks": kb["n_chunks"],
        "sources": sorted({c["source"] for c in kb["chunks"]}),
    }, indent=2, default=str))
    print(f"Wrote: {kb_index_path.relative_to(REPO_ROOT)}")

    audit = rag_out["audit_entries"] + em_out["audit_entries"]
    audit_path = AUDIT_DIR / "audit_trail.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    print(f"Wrote: {audit_path.relative_to(REPO_ROOT)}")

    retrieval_detail_path = AUDIT_DIR / "rag_retrieval_detail.md"
    retrieval_detail_path.write_text(
        _render_retrieval_detail(rag_out["mappings"]), encoding="utf-8"
    )
    print(f"Wrote: {retrieval_detail_path.relative_to(REPO_ROOT)}")

    summary = {
        "run_id": RUN_ID,
        "timestamp": TIMESTAMP,
        "mock_mode": False,
        "rag": {
            "n_documents": kb["n_documents"],
            "n_chunks": kb["n_chunks"],
            "mappings": rag_out["mappings"],
            "reasoning": rag_out["reasoning"],
        },
        "emergence": {
            "n_candidates": len(em_out["candidates"]),
            "candidates": em_out["candidates"],
            "theme_report": em_out["theme_report"],
            "reasoning": em_out["reasoning"],
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote: {summary_path.relative_to(REPO_ROOT)}")

    md_path = OUT_DIR / "report.md"
    md_path.write_text(_render_report(summary), encoding="utf-8")
    print(f"Wrote: {md_path.relative_to(REPO_ROOT)}")

    success_path = REFLECT_DIR / "phase3_success_report.txt"
    success_path.write_text(_render_success_report(summary), encoding="utf-8")
    print(f"Wrote: {success_path.relative_to(REPO_ROOT)}")

    # ---- Success report ----
    print("\n" + "=" * 60)
    print("  PHASE 3 -- SUCCESS REPORT")
    print("=" * 60)
    print(f"Status:              COMPLETE")
    print("LLM mode:            LIVE")
    print(f"Documents indexed:   {kb['n_documents']}")
    print(f"Constructs grounded: {len(rag_out['mappings'])}")
    print(f"Emergence candidates:{len(em_out['candidates'])}")
    print(f"\nNext: open outputs/phase3_emergent_themes/report.md "
          f"for the Gate 3 review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
