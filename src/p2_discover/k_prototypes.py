"""
K-Prototypes Agent -- mixed-data clustering specialist.

Discovers natural workforce segments in datasets containing both
categorical demographics and continuous survey responses using the
K-Prototypes algorithm (Huang, 1998).  K-Prototypes combines
K-Means (Euclidean distance for numeric features) with K-Modes
(Hamming distance for categorical features) through a gamma-
weighted cost function.

Key steps:
1. Z-score standardise numeric columns so no single variable
   dominates the cost function.
2. Tune the gamma parameter (categorical vs. numeric trade-off).
3. Enumerate K values with the elbow method and silhouette scores.
4. Select optimal K via a multi-criteria framework.
5. Return labels, centroids, and elbow-plot data.

Unlike LPA (which profiles on survey responses only), K-Prototypes
explicitly uses demographics as clustering features, finding
behavioural-demographic segments.

References
----------
Huang, Z. (1998). Extensions to the k-means algorithm for
    clustering large data sets with categorical values. DMKD, 2(3).
Cao, Liang, & Bai (2009). A new initialization method for
    categorical data clustering. Expert Systems with Applications.
"""

import warnings

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from scipy.stats import zscore
from sklearn.metrics import silhouette_score

from src import config
from src.utils import audit_entry


# ── Private helpers ──────────────────────────────────────────────────────


def _prepare_data(df, cat_cols, num_cols):
    """Z-score numeric columns and build the combined data matrix.

    Returns the transformed DataFrame (copy), the numpy matrix,
    and the list of categorical column indices within that matrix.
    """
    df_work = df.copy()

    # Check if already standardised
    means = df_work[num_cols].mean()
    sds = df_work[num_cols].std()
    already_std = (means.abs() < 0.01).all() and ((sds - 1).abs() < 0.01).all()

    if not already_std:
        for col in num_cols:
            df_work[col] = zscore(df_work[col].values, nan_policy="omit")

    # Build matrix: numeric columns first, then categorical
    ordered_cols = num_cols + cat_cols
    data_matrix = df_work[ordered_cols].values
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    return df_work, data_matrix, cat_indices, ordered_cols


def _detect_elbow(costs, k_values):
    """Detect the elbow point using second-order differences.

    The elbow is where the rate of cost decrease drops most
    sharply, indicating diminishing returns from additional
    clusters.
    """
    deltas = np.diff(costs)
    if len(deltas) < 2 or np.all(deltas == 0):
        return int(k_values[0])

    # Avoid division by zero
    safe_deltas = np.where(deltas[:-1] == 0, 1e-10, deltas[:-1])
    ratios = np.abs(deltas[1:] / safe_deltas)
    elbow_idx = int(np.argmin(ratios)) + 2  # offset for K indexing
    if elbow_idx >= len(k_values):
        elbow_idx = len(k_values) - 1
    return int(k_values[elbow_idx])


def _build_profiles(df_work, labels, k_selected, num_cols, cat_cols):
    """Create a human-readable profile DataFrame for each cluster.

    For each cluster the profile includes mean Z-scores on numeric
    dimensions and the modal category for each demographic column.
    """
    rows = []
    for k in range(k_selected):
        mask = labels == k
        row = {"cluster": k, "n": int(mask.sum()), "pct": round(mask.mean() * 100, 1)}

        # Numeric means
        for col in num_cols:
            row[col] = round(float(df_work.loc[mask, col].mean()), 3)

        # Categorical modes
        for col in cat_cols:
            modes = df_work.loc[mask, col].mode()
            row[col] = modes.iloc[0] if len(modes) > 0 else "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


# ── Public entry point ───────────────────────────────────────────────────


def run_k_prototypes(df, cat_cols=None, num_cols=None, k_range=None):
    """Run K-Prototypes clustering on mixed-type survey data.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned survey data from the Data Steward.
    cat_cols : list[str], optional
        Categorical columns.  Defaults to ``config.CATEGORICAL_COLS``.
    num_cols : list[str], optional
        Numeric columns.  Defaults to ``config.NUMERIC_COLS``.
    k_range : range or list[int], optional
        Range of K values to evaluate.  Defaults to ``range(2, 8)``.

    Returns
    -------
    dict
        Keys: ``labels``, ``centroids``, ``k_selected``,
        ``elbow_data``, ``profiles``, ``audit_entries``.
    """
    cat_cols = cat_cols or list(config.CATEGORICAL_COLS)
    num_cols = num_cols or list(config.NUMERIC_COLS)
    k_range = k_range or range(2, 8)
    audit = []

    # ── Prepare data ────────────────────────────────────────────────
    df_complete = df.dropna(subset=cat_cols + num_cols).copy()
    df_work, data_matrix, cat_indices, ordered_cols = _prepare_data(
        df_complete, cat_cols, num_cols
    )

    # Gamma: Huang default = mean SD of numeric cols (post-standardisation ~ 1.0)
    gamma = float(df_work[num_cols].std().mean())

    audit.append(
        audit_entry(
            "Discover", "K-Prototypes", "Prepared data",
            {
                "n_complete": len(df_complete),
                "gamma": round(gamma, 4),
                "cat_cols": cat_cols,
                "num_cols": num_cols,
            },
        )
    )

    # ── Enumerate K values ──────────────────────────────────────────
    costs = []
    sil_scores = {}
    all_labels = {}

    for k in k_range:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kp = KPrototypes(
                n_clusters=k,
                init="Cao",
                random_state=config.SEED,
                n_init=5,
                max_iter=100,
                gamma=gamma,
            )
            lbls = kp.fit_predict(data_matrix, categorical=cat_indices)

        costs.append(float(kp.cost_))
        all_labels[k] = lbls

        # Silhouette on numeric features only (Gower distance is expensive
        # and the tutorial keeps dependencies light)
        numeric_data = df_work[num_cols].values
        if len(np.unique(lbls)) > 1:
            sil = float(
                silhouette_score(numeric_data, lbls, metric="euclidean",
                                 sample_size=min(5000, len(numeric_data)),
                                 random_state=config.SEED)
            )
        else:
            sil = -1.0
        sil_scores[k] = sil

    k_vals = list(k_range)

    # ── Select optimal K ────────────────────────────────────────────
    elbow_k = _detect_elbow(np.array(costs), np.array(k_vals))
    best_sil_k = max(sil_scores, key=sil_scores.get)

    # Multi-criteria selection
    if elbow_k == best_sil_k:
        k_selected = elbow_k
        rationale = (
            f"Elbow and silhouette converge at K={k_selected}."
        )
    else:
        # Prefer silhouette (external validation) when they disagree
        k_selected = best_sil_k
        rationale = (
            f"Elbow suggests K={elbow_k}, silhouette suggests K={best_sil_k}. "
            f"Selected K={k_selected} based on silhouette."
        )

    labels = all_labels[k_selected]

    audit.append(
        audit_entry(
            "Discover", "K-Prototypes", "K selection",
            {
                "elbow_k": elbow_k,
                "best_sil_k": best_sil_k,
                "k_selected": k_selected,
                "rationale": rationale,
            },
        )
    )

    # ── Final model centroids ───────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = KPrototypes(
            n_clusters=k_selected,
            init="Cao",
            random_state=config.SEED,
            n_init=10,
            max_iter=100,
            gamma=gamma,
        )
        labels = final_model.fit_predict(data_matrix, categorical=cat_indices)

    raw_centroids = final_model.cluster_centroids_

    # Split centroids into numeric and categorical parts
    numeric_centroids = pd.DataFrame(
        raw_centroids[:, : len(num_cols)].astype(float),
        columns=num_cols,
        index=[f"Cluster {i}" for i in range(k_selected)],
    )
    categorical_centroids = pd.DataFrame(
        raw_centroids[:, len(num_cols) :],
        columns=cat_cols,
        index=[f"Cluster {i}" for i in range(k_selected)],
    )

    centroids = {
        "numeric": numeric_centroids.to_dict(),
        "categorical": categorical_centroids.to_dict(),
        "gamma": gamma,
    }

    # ── Build cluster profiles ──────────────────────────────────────
    profiles = _build_profiles(df_work, labels, k_selected, num_cols, cat_cols)

    # ── Elbow data (for plotting in the notebook) ───────────────────
    elbow_data = {
        "k_values": k_vals,
        "costs": costs,
        "silhouette_scores": {int(k): round(v, 4) for k, v in sil_scores.items()},
    }

    # ── Build reasoning narrative ──────────────────────────────────
    n_respondents = len(df_work)
    cluster_sizes = [int((labels == c).sum()) for c in range(k_selected)]
    cluster_pcts = [round(s / n_respondents * 100, 1) for s in cluster_sizes]
    best_sil = sil_scores.get(k_selected, 0)
    sil_quality = (
        "strong" if best_sil > 0.70 else
        "reasonable" if best_sil > 0.50 else
        "fair" if best_sil > 0.25 else
        "weak"
    )
    reasoning_parts = [
        f"K-Prototypes clustering (Huang, 1998) on {n_respondents:,} "
        f"respondents with {len(num_cols)} numeric and {len(cat_cols)} "
        f"categorical features.",
        f"Gamma (categorical weight) set to {gamma:.3f} based on mean SD "
        f"of standardized numeric features.",
        rationale,
        f"Selected solution has {k_selected} clusters with sizes: "
        + ", ".join(f"n={s} ({p}%)" for s, p in zip(cluster_sizes, cluster_pcts))
        + ".",
        f"Silhouette score at K={k_selected}: {best_sil:.3f} ({sil_quality} "
        f"separation per Rousseeuw, 1987).",
    ]
    if any(p < 5 for p in cluster_pcts):
        small = [f"Cluster {i}" for i, p in enumerate(cluster_pcts) if p < 5]
        reasoning_parts.append(
            f"Note: {', '.join(small)} contain(s) < 5% of respondents. "
            f"Small clusters may be unstable or represent outlier groups."
        )
    reasoning = " ".join(reasoning_parts)

    return {
        "labels": labels,
        "centroids": centroids,
        "k_selected": int(k_selected),
        "elbow_data": elbow_data,
        "profiles": profiles,
        "reasoning": reasoning,
        "audit_entries": audit,
    }
