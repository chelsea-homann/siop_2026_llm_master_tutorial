"""
Emergence Agent -- emergent theme and new-segment detection.

Operates in two modes:

1. **Cross-sectional** -- Analyses cluster profiles against
   codebook constructs to identify patterns that fall outside
   the established construct set.  Uses the LLM (or mock) to
   classify candidates as new, variant, or noise.

2. **Longitudinal** -- Takes weak-fit respondents from the
   Continuity Agent and runs a K+1 test to determine whether
   a genuinely new segment is forming.

References
----------
Huang, Z. (1998). Extensions to the k-means algorithm. DMKD, 2(3).
"""

import warnings

import numpy as np
import pandas as pd

from src import config
from src.utils import audit_entry, call_llm


# ── Private helpers ──────────────────────────────────────────────────────


def _analyse_profiles_vs_codebook(cluster_profiles, codebook_constructs):
    """Compare cluster profiles against the I-O codebook constructs.

    Returns a list of candidate emergent themes -- dimensions
    where cluster centroid patterns deviate substantially from
    what the 12 codebook constructs would predict.
    """
    # In the tutorial data the survey columns directly map to
    # codebook constructs, but cross-cluster *combinations* may
    # be novel (e.g., High-Trust + Low-Morale is unexpected).
    candidates = []
    for idx, row in cluster_profiles.iterrows():
        # Look for unusual combinations (high on one dimension,
        # low on another that typically co-varies positively)
        numeric_cols = [
            c for c in config.NUMERIC_COLS if c in cluster_profiles.columns
        ]
        vals = {c: row[c] for c in numeric_cols if not pd.isna(row.get(c))}

        high = [c for c, v in vals.items() if v > 0.5]
        low = [c for c, v in vals.items() if v < -0.5]

        if high and low:
            candidates.append(
                {
                    "cluster": int(row.get("cluster", idx)),
                    "pattern": f"High: {', '.join(high)} + Low: {', '.join(low)}",
                    "high_dims": high,
                    "low_dims": low,
                }
            )

    return candidates


def _llm_classify_themes(candidates, codebook_constructs):
    """Use the LLM to classify each candidate as NEW, VARIANT, or NOISE."""
    prompt_parts = [
        "You are an I-O psychologist with knowledge of construct validation, "
        "reviewing candidate emergent themes from a workforce survey clustering analysis.",
        "",
        "Codebook constructs: " + ", ".join(codebook_constructs),
        "",
        "For each candidate, classify it as:",
        "  - NEW: a genuinely novel theme not captured by existing constructs",
        "  - VARIANT: a known construct appearing in an unusual combination",
        "  - NOISE: likely a statistical artifact",
        "",
        "Candidates:",
    ]
    for c in candidates:
        prompt_parts.append(f"  Cluster {c['cluster']}: {c['pattern']}")

    prompt_parts.append(
        '\nRespond as JSON: [{"cluster": N, "classification": "...", '
        '"reasoning": "...", "confidence": 0.0}]'
    )
    prompt = "\n".join(prompt_parts)
    system = (
        "You are an I-O psychologist with expertise in construct validation. "
        "Respond only with the requested JSON array."
    )

    return call_llm(prompt, system=system)


# ── Public entry points ──────────────────────────────────────────────────


def detect_emergent_themes(cluster_profiles, codebook_constructs=None,
                           open_ended_responses=None):
    """Cross-sectional emergent theme detection.

    Analyses cluster profiles against the construct codebook to
    find patterns that may represent novel workforce experiences
    beyond the 12 established constructs.

    Parameters
    ----------
    cluster_profiles : pandas.DataFrame
        Profile summary from K-Prototypes or LPA (one row per
        cluster with mean scores on each survey dimension).
    codebook_constructs : list[str], optional
        Names of the codebook constructs.  Defaults to the five
        survey column names.
    open_ended_responses : pandas.DataFrame, optional
        Open-ended text data for additional thematic evidence.

    Returns
    -------
    dict
        Keys: ``candidates``, ``theme_report``, ``audit_entries``.
    """
    codebook_constructs = codebook_constructs or list(config.NUMERIC_COLS)
    audit = []

    # ── Identify candidates ─────────────────────────────────────────
    candidates = _analyse_profiles_vs_codebook(
        cluster_profiles, codebook_constructs,
    )

    audit.append(
        audit_entry(
            "Ground", "Emergence", "Cross-sectional scan",
            {"n_candidates": len(candidates)},
        )
    )

    # ── LLM classification ──────────────────────────────────────────
    theme_report = None
    if candidates:
        theme_report = _llm_classify_themes(candidates, codebook_constructs)

    audit.append(
        audit_entry(
            "Ground", "Emergence", "Theme classification",
            {"n_candidates": len(candidates)},
        )
    )

    # ── Build reasoning via LLM ────────────────────────────────────
    n_cand = len(candidates)
    system_r = (
        "You are the Emergence agent for an I-O psychology pipeline "
        "(Braun & Clarke, 2006; Glaser & Strauss, 2017). Summarize the "
        "cross-sectional theme scan results in 2-4 sentences. Interpret "
        "what any emergent themes mean for the organization and give a "
        "clear verdict for the I-O psychologist's Gate 3 decision."
    )
    if n_cand == 0:
        prompt_r = (
            f"Cross-sectional scan of {len(codebook_constructs)} codebook "
            f"constructs found 0 candidate emergent themes -- all cluster "
            f"profiles align with established constructs. In 2 sentences, "
            "interpret what this means for codebook coverage and Gate 3."
        )
    else:
        candidate_lines = "\n".join(
            f"  Cluster {c.get('cluster', '?')}: {c.get('pattern', '?')}"
            for c in candidates
        )
        classifications = ""
        if theme_report and isinstance(theme_report, list):
            classifications = "\nLLM classifications:\n" + "\n".join(
                f"  Cluster {t.get('cluster', '?')}: {t.get('classification', '?')} -- "
                f"{t.get('reasoning', '')[:120]}"
                for t in theme_report
            )
        prompt_r = (
            f"Cross-sectional scan identified {n_cand} candidate emergent theme(s):\n"
            f"{candidate_lines}"
            f"{classifications}\n\n"
            "In 2-4 sentences: interpret what these patterns mean for the "
            "organization, distinguish genuine novelty from statistical artifacts, "
            "and give your verdict for the I-O psychologist's Gate 3 decision."
        )
    reasoning = call_llm(prompt_r, system=system_r)

    return {
        "candidates": candidates,
        "theme_report": theme_report,
        "reasoning": reasoning,
        "audit_entries": audit,
    }


def test_new_segments(weak_fit_df, cat_cols=None, num_cols=None):
    """Longitudinal K+1 test on weak-fit respondents.

    Re-clusters the weak-fit pool (respondents who did not align
    well with any baseline centroid) to test whether genuine new
    sub-structure exists.

    Parameters
    ----------
    weak_fit_df : pandas.DataFrame
        Follow-up respondents flagged as weak-fit by the
        Continuity Agent.
    cat_cols : list[str], optional
        Categorical columns. Defaults to ``config.CATEGORICAL_COLS``.
    num_cols : list[str], optional
        Numeric columns. Defaults to ``config.NUMERIC_COLS``.

    Returns
    -------
    dict
        Keys: ``k_plus_1_results``, ``new_segment_candidates``,
        ``audit_entries``.
    """
    cat_cols = cat_cols or list(config.CATEGORICAL_COLS)
    num_cols = num_cols or list(config.NUMERIC_COLS)
    audit = []

    n_weak = len(weak_fit_df)
    if n_weak < 30:
        # Too few respondents to meaningfully cluster
        audit.append(
            audit_entry(
                "Longitudinal", "Emergence", "K+1 test skipped",
                {"reason": f"Only {n_weak} weak-fit respondents (minimum 30)"},
            )
        )
        return {
            "k_plus_1_results": {"skipped": True, "reason": "insufficient_n"},
            "new_segment_candidates": [],
            "audit_entries": audit,
        }

    # ── Simple K=1,2,3 test on the weak-fit pool ───────────────────
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score as sil_score

    X = weak_fit_df[num_cols].values.astype(float)
    # Remove NaN rows
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]

    results = {}
    for k in [1, 2, 3]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = GaussianMixture(
                n_components=k,
                random_state=config.SEED,
                n_init=5,
                max_iter=200,
            )
            gmm.fit(X)
        bic = float(gmm.bic(X))
        labels = gmm.predict(X)

        sil = None
        if k > 1 and len(np.unique(labels)) > 1:
            sil = float(sil_score(X, labels, metric="euclidean"))

        results[k] = {
            "bic": round(bic, 2),
            "silhouette": round(sil, 4) if sil is not None else None,
            "sizes": [int(s) for s in np.bincount(labels)],
        }

    # Determine if sub-structure is present
    new_candidates = []
    if results[2]["silhouette"] is not None and results[2]["silhouette"] > config.SILHOUETTE_MODERATE:
        new_candidates.append(
            {
                "k": 2,
                "silhouette": results[2]["silhouette"],
                "interpretation": "Possible new segment in weak-fit pool",
            }
        )

    audit.append(
        audit_entry(
            "Longitudinal", "Emergence", "K+1 test complete",
            {"results_by_k": results, "new_candidates": len(new_candidates)},
        )
    )

    return {
        "k_plus_1_results": results,
        "new_segment_candidates": new_candidates,
        "audit_entries": audit,
    }
