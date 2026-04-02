"""
Continuity Agent -- longitudinal alignment specialist.

Maps follow-up survey respondents to baseline cluster centroids so
that workforce migration between segments can be tracked across
survey waves.  Uses Euclidean distance on Z-scored numeric features
combined with Hamming distance on categorical features to compute
a composite alignment score.

Respondents whose distance to every centroid exceeds the Weak-Fit
threshold are flagged -- they may belong to an emergent segment not
present at baseline.

References
----------
Lu, Z. (2025). Clustering longitudinal data: A review. ISR, 93.
Moore, Quartiroli, & Little (2025). Best-practice recommendations
    for longitudinal latent transition analysis.
Bakac, Zyberaj, & Barber (2022). Latent transition analysis in
    organizational psychology. Frontiers in Psychology.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

from src import config
from src.utils import audit_entry


# ── Private helpers ──────────────────────────────────────────────────────


def _standardise_with_baseline(followup_df, baseline_df, num_cols):
    """Standardise follow-up data using baseline mean and SD.

    Using the baseline parameters (not the follow-up's own) is
    critical; self-standardisation would mask genuine shifts in
    workforce attitudes between survey waves.
    """
    fu = followup_df.copy()
    for col in num_cols:
        bl_mean = float(baseline_df[col].mean())
        bl_sd = float(baseline_df[col].std())
        if bl_sd > 0:
            fu[col] = (fu[col] - bl_mean) / bl_sd
        # If baseline SD is zero the column has no variance and
        # should have been excluded upstream
    return fu


def _compute_distances(followup_df, centroids_numeric, centroids_cat,
                       num_cols, cat_cols):
    """Compute composite distance from each respondent to each centroid.

    Numeric distance is Euclidean on Z-scored values; categorical
    distance is the Hamming proportion (fraction of mismatched
    levels).  The composite weights each part by the number of
    features of that type.
    """
    k_base = len(centroids_numeric)
    n = len(followup_df)
    n_num = len(num_cols)
    n_cat = len(cat_cols)
    total_features = n_num + n_cat

    w_num = n_num / total_features if total_features > 0 else 0.5
    w_cat = n_cat / total_features if total_features > 0 else 0.5

    dist_matrix = np.zeros((n, k_base))

    fu_num = followup_df[num_cols].values.astype(float)
    fu_cat = followup_df[cat_cols].values

    for c_idx in range(k_base):
        centroid_num = centroids_numeric[c_idx].astype(float)
        centroid_cat = centroids_cat[c_idx]

        # Euclidean distance on numeric features (normalised by
        # sqrt of number of dimensions for comparability)
        num_diff = fu_num - centroid_num
        eucl = np.sqrt((num_diff ** 2).sum(axis=1)) / max(np.sqrt(n_num), 1)

        # Hamming distance on categorical features
        hamm = (fu_cat != centroid_cat).mean(axis=1)

        dist_matrix[:, c_idx] = w_num * eucl + w_cat * hamm

    return dist_matrix


def _build_transition_matrix(baseline_labels, followup_labels, k_base):
    """Build a transition probability matrix (baseline -> follow-up).

    Row i gives the proportion of baseline cluster-i members that
    were assigned to each follow-up cluster.  This is the aggregate
    version used when respondent IDs are not available for panel
    tracking.
    """
    baseline_props = pd.Series(baseline_labels).value_counts(normalize=True).sort_index()
    followup_props = pd.Series(followup_labels).value_counts(normalize=True).sort_index()

    # Align indices
    all_labels = sorted(set(baseline_props.index) | set(followup_props.index))
    comparison = pd.DataFrame(
        {
            "Baseline_pct": [round(baseline_props.get(l, 0) * 100, 1) for l in all_labels],
            "Followup_pct": [round(followup_props.get(l, 0) * 100, 1) for l in all_labels],
        },
        index=all_labels,
    )
    comparison["Change_ppt"] = comparison["Followup_pct"] - comparison["Baseline_pct"]
    return comparison


# ── Public entry point ───────────────────────────────────────────────────


def align_to_baseline(followup_df, baseline_centroids, cat_cols=None,
                      num_cols=None, baseline_labels=None, baseline_df=None):
    """Align follow-up respondents to baseline cluster centroids.

    Parameters
    ----------
    followup_df : pandas.DataFrame
        Cleaned follow-up survey data.
    baseline_centroids : dict
        Centroids from K-Prototypes, containing ``"numeric"`` and
        ``"categorical"`` sub-dicts plus ``"gamma"``.
    cat_cols : list[str], optional
        Categorical columns. Defaults to ``config.CATEGORICAL_COLS``.
    num_cols : list[str], optional
        Numeric columns. Defaults to ``config.NUMERIC_COLS``.
    baseline_labels : numpy.ndarray, optional
        Baseline cluster assignments (for transition matrix).
    baseline_df : pandas.DataFrame, optional
        Baseline data used for standardisation anchoring.

    Returns
    -------
    dict
        Keys: ``aligned_labels``, ``weak_fit_mask``,
        ``transition_matrix``, ``weak_fit_count``,
        ``distance_matrix``, ``audit_entries``.
    """
    cat_cols = cat_cols or list(config.CATEGORICAL_COLS)
    num_cols = num_cols or list(config.NUMERIC_COLS)
    audit = []

    # ── Reconstruct centroid arrays ─────────────────────────────────
    num_centroid_dict = baseline_centroids["numeric"]
    cat_centroid_dict = baseline_centroids["categorical"]

    # Convert dict-of-dicts to arrays
    # num_centroid_dict looks like {col: {"Cluster 0": val, ...}}
    cluster_keys = sorted(
        {k for col_d in num_centroid_dict.values() for k in col_d.keys()}
    )
    k_base = len(cluster_keys)

    centroids_numeric = np.array(
        [[num_centroid_dict[col][ck] for col in num_cols] for ck in cluster_keys]
    )
    centroids_cat = np.array(
        [[cat_centroid_dict[col][ck] for col in cat_cols] for ck in cluster_keys]
    )

    # ── Standardise follow-up with baseline parameters ──────────────
    if baseline_df is not None:
        fu = _standardise_with_baseline(followup_df, baseline_df, num_cols)
    else:
        # Fallback: Z-score to own distribution (less ideal but
        # functional when baseline data is not available)
        fu = followup_df.copy()
        for col in num_cols:
            fu[col] = zscore(fu[col].values, nan_policy="omit")

    fu = fu.dropna(subset=num_cols + cat_cols).copy()

    audit.append(
        audit_entry(
            "Longitudinal", "Continuity", "Prepared follow-up data",
            {"n_followup": len(fu), "k_baseline": k_base},
        )
    )

    # ── Distance computation ────────────────────────────────────────
    dist_matrix = _compute_distances(
        fu, centroids_numeric, centroids_cat, num_cols, cat_cols,
    )

    # ── Alignment ───────────────────────────────────────────────────
    aligned_labels = dist_matrix.argmin(axis=1)
    min_distances = dist_matrix.min(axis=1)

    # ── Weak-fit flagging ───────────────────────────────────────────
    weak_fit_mask = min_distances > config.WEAK_FIT_DISTANCE
    weak_fit_count = int(weak_fit_mask.sum())
    pct_weak = round(weak_fit_count / len(fu) * 100, 1)

    audit.append(
        audit_entry(
            "Longitudinal", "Continuity", "Alignment complete",
            {
                "weak_fit_count": weak_fit_count,
                "pct_weak_fit": pct_weak,
                "threshold": config.WEAK_FIT_DISTANCE,
                "mean_distance": round(float(min_distances.mean()), 4),
            },
        )
    )

    # ── Transition matrix (aggregate proportions) ───────────────────
    transition_matrix = None
    if baseline_labels is not None:
        transition_matrix = _build_transition_matrix(
            baseline_labels, aligned_labels, k_base,
        )

    return {
        "aligned_labels": aligned_labels,
        "weak_fit_mask": weak_fit_mask,
        "transition_matrix": transition_matrix,
        "weak_fit_count": weak_fit_count,
        "distance_matrix": dist_matrix,
        "audit_entries": audit,
    }
