"""
Narrator Agent -- persona narrative generation via LLM.

Translates statistical cluster profiles into evidence-grounded
persona narratives.  Every claim in a generated narrative must be
traceable to a specific centroid value or empirical observation;
the agent applies the epistemic risk mitigation protocol from
Nguyen & Welch (2025) to guard against anthropomorphic
interpretation, fabricated quotations, and the Oracle Effect.

In mock mode the agent returns realistic pre-generated personas
stored in ``narrator_output.json``.

References
----------
Nguyen, D. C. & Welch, C. (2025). Generative artificial
    intelligence in qualitative data analysis. Organizational
    Research Methods (advance online).
"""

import json

from src import config
from src.utils import audit_entry, call_llm, load_mock


# ── Private helpers ──────────────────────────────────────────────────────


def _build_prompt(cluster_profiles, construct_scores, policy_context,
                  codebook):
    """Assemble the LLM prompt for persona generation.

    The prompt anchors every narrative instruction in the
    statistical profile so that the LLM cannot hallucinate
    characteristics unsupported by the data.
    """
    profile_text = ""
    for _, row in cluster_profiles.iterrows():
        cluster_id = row.get("cluster", "?")
        n = row.get("n", "?")
        pct = row.get("pct", "?")
        dims = []
        for col in config.NUMERIC_COLS:
            if col in row.index:
                val = row[col]
                direction = "high" if val > 0.5 else ("low" if val < -0.5 else "moderate")
                dims.append(f"{col}={val:.2f} ({direction})")

        cat_parts = []
        for col in config.CATEGORICAL_COLS:
            if col in row.index:
                cat_parts.append(f"{col}={row[col]}")

        profile_text += (
            f"\nCluster {cluster_id} (n={n}, {pct}%):\n"
            f"  Survey dimensions: {', '.join(dims)}\n"
            f"  Demographic mode: {', '.join(cat_parts)}\n"
        )

    policy_section = ""
    if policy_context:
        policy_section = (
            "\nOrganisational policy context (from RAG retrieval):\n"
            + json.dumps(policy_context, indent=2, default=str)[:2000]
        )

    prompt = (
        "Generate an evidence-based persona narrative for each cluster below. "
        "Follow these rules strictly:\n"
        "1. Ground every characterisation in the centroid values provided.\n"
        "2. Do NOT infer emotions or motivations beyond what the scores show.\n"
        "3. Give each persona a neutral, non-stigmatising name (3-5 words).\n"
        "4. Include a 2-3 sentence narrative and a statistical fingerprint.\n"
        "5. Note any policy-experience mismatches if policy context is provided.\n"
        "6. End each persona with: 'Note: narrative generated with AI assistance. "
        "The I-O psychologist retains final interpretive authority.'\n"
        "7. Do NOT use em dashes anywhere.\n\n"
        f"Cluster profiles:\n{profile_text}"
        f"{policy_section}\n\n"
        "Respond as a JSON array of objects, each with keys: "
        "cluster_id, persona_name, size, pct, statistical_fingerprint, "
        "narrative, policy_mismatches, epistemic_note."
    )

    system = (
        "You are an I-O psychologist with skills in evidence-based persona "
        "construction. You follow the epistemic risk mitigation protocol: "
        "no unfounded inferences, no fabricated quotes, statistical anchoring "
        "for every claim, and explicit uncertainty disclosure."
    )

    return prompt, system


# ── Public entry point ───────────────────────────────────────────────────


def generate_personas(cluster_profiles, construct_scores=None,
                      policy_context=None, codebook=None):
    """Generate persona narratives for each cluster.

    Parameters
    ----------
    cluster_profiles : pandas.DataFrame
        One row per cluster with centroid means on survey
        dimensions and modal demographics.
    construct_scores : dict, optional
        Additional construct-level information (e.g., from LPA
        fingerprints).
    policy_context : dict, optional
        RAG-retrieved policy passages keyed by construct.
    codebook : list[str], optional
        Codebook construct names.

    Returns
    -------
    dict
        Keys: ``personas`` (list of dicts), ``audit_entries``.
    """
    audit = []

    n_clusters = len(cluster_profiles)

    # ── Mock mode ───────────────────────────────────────────────────
    if config.MOCK_MODE:
        personas = load_mock("narrator_output.json")
        n_p = len(personas) if isinstance(personas, list) else 0
        audit.append(
            audit_entry(
                "Write", "Narrator", "Generated personas (mock)",
                {"n_personas": n_p},
            )
        )
        reasoning = (
            f"Generated {n_p} persona narratives from {n_clusters} cluster "
            f"profiles (mock mode). Each persona follows the epistemic risk "
            f"mitigation protocol (Nguyen & Welch, 2025): claims are anchored "
            f"in centroid values, uncertainty is stated explicitly, and "
            f"policy-experience mismatches are flagged. The I-O psychologist "
            f"retains final interpretive authority over all narratives."
        )
        return {"personas": personas, "reasoning": reasoning, "audit_entries": audit}

    # ── Live mode ───────────────────────────────────────────────────
    prompt, system = _build_prompt(
        cluster_profiles, construct_scores, policy_context, codebook,
    )

    raw = call_llm(prompt, system=system)

    # Parse the LLM JSON response
    try:
        personas = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON from a wrapped response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                personas = json.loads(raw[start:end])
            except json.JSONDecodeError:
                personas = [{"raw_response": raw, "parse_error": True}]
        else:
            personas = [{"raw_response": raw, "parse_error": True}]

    n_p = len(personas)
    audit.append(
        audit_entry(
            "Write", "Narrator", "Generated personas (live)",
            {"n_personas": n_p},
        )
    )

    has_policy = policy_context is not None
    reasoning = (
        f"Generated {n_p} persona narratives from {n_clusters} cluster "
        f"profiles using live LLM inference. Each narrative is anchored in "
        f"centroid values and follows the epistemic risk mitigation protocol "
        f"(Nguyen & Welch, 2025). "
        + (f"Policy context from {len(policy_context)} constructs was "
           f"incorporated via RAG retrieval. " if has_policy else
           "No policy context was provided. ")
        + f"All narratives include epistemic notes disclosing evidence basis "
        f"and limitations. The I-O psychologist retains final interpretive "
        f"authority."
    )

    return {"personas": personas, "reasoning": reasoning, "audit_entries": audit}
