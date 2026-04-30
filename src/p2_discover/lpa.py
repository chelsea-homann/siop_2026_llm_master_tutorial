"""
LPA Agent -- Latent Profile Analysis via Gaussian Mixture Models.

Identifies latent subpopulations (psychological profiles) from
continuous survey indicators.  Unlike K-Prototypes, LPA uses only
survey response columns (not demographics) to discover person-
centered profiles grounded in psychometric theory.

Key steps:
1. Z-score standardise indicator columns.
2. Fit GMMs across a grid of K values and covariance types.
3. Select the optimal model using BIC, entropy, and parsimony.
4. Assign profile labels with posterior probabilities.
5. Flag ambiguous respondents (max posterior < 0.70).
6. Generate Psychological Fingerprints for each profile.

References
----------
Spurk, Hirschi, Wang, Valero, & Kauffeld (2020). LPA best
    practices in vocational behaviour research.
Nylund, Asparouhov, & Muthen (2007). Deciding on the number of
    classes in LCA/LPA. Structural Equation Modeling.
Nylund-Gibson & Choi (2018). Ten frequently asked questions about
    LCA/LPA. Measurement and Evaluation in Counseling.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.mixture import GaussianMixture

from src import config
from src.utils import audit_entry, call_llm


# ── Private helpers ──────────────────────────────────────────────────────


def _count_params(k, p, cov_type):
    """Count free parameters for a Gaussian Mixture Model.

    Accurate parameter counts are essential for information
    criteria (AIC, BIC, SABIC) because they penalise model
    complexity.
    """
    means_params = k * p
    mix_params = k - 1
    if cov_type == "spherical":
        cov_params = k
    elif cov_type == "diag":
        cov_params = k * p
    elif cov_type == "tied":
        cov_params = p * (p + 1) // 2
    elif cov_type == "full":
        cov_params = k * p * (p + 1) // 2
    else:
        cov_params = k * p
    return means_params + mix_params + cov_params


def _compute_entropy(posteriors, k):
    """Compute the normalised entropy (Muthen convention).

    Entropy near 1.0 means respondents are classified with high
    certainty; values below 0.60 indicate substantial overlap
    between profiles.
    """
    if k <= 1:
        return 1.0
    n = posteriors.shape[0]
    raw = -np.sum(posteriors * np.log(posteriors + 1e-15)) / n
    max_entropy = np.log(k)
    return float(1 - (raw / max_entropy))


def _build_fingerprints(X, labels, indicator_cols, k):
    """Create Psychological Fingerprints for each profile.

    A fingerprint labels each indicator dimension as High (> 0.5 SD),
    Low (< -0.5 SD), or Moderate, producing a concise qualitative
    description of each profile's survey-response pattern.
    """
    fingerprints = {}
    for profile in range(k):
        mask = labels == profile
        if mask.sum() == 0:
            fingerprints[profile] = {"label": "Empty profile", "dims": {}}
            continue

        means = pd.Series(X[mask].mean(axis=0), index=indicator_cols)
        dims = {}
        parts = []
        for col in indicator_cols:
            val = float(means[col])
            if val > 0.5:
                dims[col] = "High"
                parts.append(f"High-{col}")
            elif val < -0.5:
                dims[col] = "Low"
                parts.append(f"Low-{col}")
            else:
                dims[col] = "Moderate"

        label = " / ".join(parts) if parts else "Moderate-All"
        fingerprints[profile] = {"label": label, "dims": dims, "means": means.round(3).to_dict()}

    return fingerprints


# ── Public entry point ───────────────────────────────────────────────────


def run_lpa(df, indicator_cols=None, k_range=None):
    """Run Latent Profile Analysis on continuous survey indicators.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned survey data from the Data Steward.
    indicator_cols : list[str], optional
        Continuous indicator columns.  Defaults to
        ``config.NUMERIC_COLS``.
    k_range : range or list[int], optional
        Range of profile counts to evaluate.  Defaults to
        ``range(2, 7)``.

    Returns
    -------
    dict
        Keys: ``labels``, ``profiles``, ``fit_indices``,
        ``fingerprints``, ``ambiguous_count``, ``posteriors``,
        ``audit_entries``.
    """
    indicator_cols = indicator_cols or list(config.NUMERIC_COLS)
    k_range = k_range or range(2, 7)
    audit = []

    # ── Standardise ─────────────────────────────────────────────────
    df_work = df.copy()
    X = df_work[indicator_cols].apply(
        zscore, nan_policy="omit"
    ).values

    valid_mask = ~np.isnan(X).any(axis=1)
    X_complete = X[valid_mask]
    df_complete = df_work.loc[valid_mask].copy()
    n, p = X_complete.shape

    audit.append(
        audit_entry(
            "Discover", "LPA", "Standardised indicators",
            {"n_complete": n, "n_indicators": p, "indicators": indicator_cols},
        )
    )

    # ── Enumerate models ────────────────────────────────────────────
    # For the tutorial we test diagonal and full covariance only,
    # keeping runtime reasonable while still covering the two most
    # common parameterisations (Spurk et al., 2020).
    cov_types = ["diag", "full"]
    results = []
    models = {}

    for cov_type in cov_types:
        for k in k_range:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=config.SEED,
                    n_init=10,
                    max_iter=500,
                    tol=1e-6,
                )
                gmm.fit(X_complete)

            ll = float(gmm.score(X_complete) * n)
            n_params = _count_params(k, p, cov_type)
            bic = -2 * ll + n_params * np.log(n)
            aic = -2 * ll + 2 * n_params
            sabic = -2 * ll + n_params * np.log((n + 2) / 24)

            posteriors = gmm.predict_proba(X_complete)
            entropy = _compute_entropy(posteriors, k)

            results.append(
                {
                    "K": int(k),
                    "cov_type": cov_type,
                    "converged": bool(gmm.converged_),
                    "log_likelihood": round(ll, 2),
                    "n_params": n_params,
                    "AIC": round(aic, 2),
                    "BIC": round(bic, 2),
                    "SABIC": round(sabic, 2),
                    "entropy": round(entropy, 4),
                }
            )
            models[(k, cov_type)] = gmm

    fit_df = pd.DataFrame(results)

    # ── Model selection ─────────────────────────────────────────────
    # Filter to converged models, then select by BIC (Nylund, 2007)
    valid = fit_df[fit_df["converged"]].copy()
    if valid.empty:
        raise RuntimeError(
            "No GMM models converged. Check data quality or reduce K range."
        )

    best_idx = valid["BIC"].idxmin()
    best_row = valid.loc[best_idx]
    optimal_k = int(best_row["K"])
    optimal_cov = best_row["cov_type"]

    # Near-tie check: prefer parsimony (smaller K) if BIC is within 2%
    min_bic = valid["BIC"].min()
    near_ties = valid[valid["BIC"] <= min_bic * 1.02]
    if len(near_ties) > 1:
        parsimonious = near_ties.loc[near_ties["K"].idxmin()]
        if int(parsimonious["K"]) < optimal_k:
            optimal_k = int(parsimonious["K"])
            optimal_cov = parsimonious["cov_type"]

    rationale = (
        f"K={optimal_k} with {optimal_cov} covariance minimises BIC "
        f"among converged models."
    )

    audit.append(
        audit_entry(
            "Discover", "LPA", "Model selection",
            {"optimal_k": optimal_k, "cov_type": optimal_cov, "rationale": rationale},
        )
    )

    # ── Profile assignment ──────────────────────────────────────────
    best_gmm = models[(optimal_k, optimal_cov)]
    posteriors_final = best_gmm.predict_proba(X_complete)
    labels = best_gmm.predict(X_complete)
    max_post = posteriors_final.max(axis=1)

    df_complete["LPA_Profile"] = labels
    df_complete["posterior_prob"] = max_post
    df_complete["is_ambiguous"] = max_post < config.LPA_AMBIGUITY_POSTERIOR

    ambiguous_count = int(df_complete["is_ambiguous"].sum())

    # ── Profile summaries ───────────────────────────────────────────
    profile_rows = []
    for k in range(optimal_k):
        mask = labels == k
        profile_rows.append(
            {
                "profile": k,
                "n": int(mask.sum()),
                "pct": round(mask.mean() * 100, 1),
                "mean_posterior": round(float(posteriors_final[mask, k].mean()), 3),
            }
        )
    profiles_df = pd.DataFrame(profile_rows)

    # ── Psychological Fingerprints ──────────────────────────────────
    fingerprints = _build_fingerprints(X_complete, labels, indicator_cols, optimal_k)

    # ── Fit-index summary ───────────────────────────────────────────
    fit_indices = {
        "all_models": results,
        "selected": {
            "K": optimal_k,
            "cov_type": optimal_cov,
            "BIC": float(best_row["BIC"]),
            "entropy": float(best_row["entropy"]),
            "rationale": rationale,
        },
    }

    # ── Build reasoning via LLM ────────────────────────────────────
    entropy_val = float(best_row["entropy"])
    entropy_quality = (
        "excellent" if entropy_val > 0.80 else
        "good" if entropy_val > 0.60 else
        "acceptable" if entropy_val > 0.40 else
        "poor"
    )
    ambig_pct = round(ambiguous_count / max(len(df_complete), 1) * 100, 1)
    n_models_tested = len(results)
    n_converged = len(valid)

    fp_lines = []
    for pid, fp in fingerprints.items():
        dims_str = ", ".join(f"{d}: {v}" for d, v in fp["dims"].items())
        fp_lines.append(f"  Profile {pid} ({fp['label']}): {dims_str}")

    system = (
        "You are the LPA agent, a Latent Profile Analysis specialist for I-O "
        "psychology research (Spurk et al., 2020; Nylund et al., 2007). Provide "
        "expert commentary on the model selection and psychological fingerprints "
        "in 3-5 sentences. Evaluate entropy, interpret each profile, assess the "
        "ambiguity rate, and give a verdict for the I-O psychologist's Gate 2 decision."
    )
    prompt = (
        f"Review this LPA solution ({n} respondents, {len(indicator_cols)} indicators):\n\n"
        f"Models tested: {n_models_tested} ({n_converged} converged) across "
        f"K={list(k_range)[0]}-{list(k_range)[-1]} with diagonal and full covariance.\n"
        f"Selected: K={optimal_k}, {optimal_cov} covariance  |  BIC: {float(best_row['BIC']):.1f}\n"
        f"Selection rationale: {rationale}\n"
        f"Entropy: {entropy_val:.3f} ({entropy_quality}; 1.0 = perfect, <0.60 = substantial overlap)\n"
        f"Ambiguous respondents (posterior < {config.LPA_AMBIGUITY_POSTERIOR}): "
        f"{ambiguous_count} ({ambig_pct}%)\n"
        f"\nPsychological fingerprints:\n"
        + "\n".join(fp_lines)
        + "\n\nIn 3-5 sentences: justify model selection using BIC and entropy, "
        "interpret each fingerprint in I-O psychology terms, assess the ambiguity "
        "rate, and give your verdict for Gate 2."
    )
    reasoning = call_llm(prompt, system=system)

    return {
        "labels": labels,
        "profiles": profiles_df,
        "fit_indices": fit_indices,
        "fingerprints": fingerprints,
        "ambiguous_count": ambiguous_count,
        "posteriors": posteriors_final,
        "reasoning": reasoning,
        "audit_entries": audit,
    }
