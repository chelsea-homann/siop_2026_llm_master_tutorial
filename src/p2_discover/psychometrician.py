"""
Psychometrician Agent -- statistical auditor for cluster validation.

Validates cluster integrity by computing silhouette coefficients,
flagging outliers by centroid distance, and (when two solutions
are available) computing the Adjusted Rand Index (ARI) as a
measure of cross-method agreement.

The ARI measures cross-method agreement between two independent
clustering solutions, corrected for chance under a hypergeometric
null distribution (Hubert & Arabie, 1985).

References
----------
Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to cluster
    validation. J. Computational and Applied Mathematics, 20.
Hubert, L. & Arabie, P. (1985). Comparing partitions. Journal
    of Classification, 2(1).
Steinley, D. (2004). Properties of the Hubert-Arabie adjusted
    Rand index. Psychological Methods, 9(3).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

from src import config
from src.utils import audit_entry


# ── Private helpers ──────────────────────────────────────────────────────


def _compute_centroid_distances(X, labels, k):
    """Euclidean distance from each observation to its cluster centroid.

    Centroid distance is the primary signal for outlier detection.
    Respondents far from their assigned centroid may represent
    qualitatively different experiences not captured by the
    current cluster solution.
    """
    centroids = np.array([X[labels == c].mean(axis=0) for c in range(k)])
    distances = np.zeros(len(X))
    for i in range(len(X)):
        distances[i] = float(np.linalg.norm(X[i] - centroids[labels[i]]))
    return distances, centroids


def _interpret_silhouette(score):
    """Map a global silhouette score to an interpretation band.

    Bands follow Rousseeuw (1987) and Kaufman & Rousseeuw (2009).
    """
    if score > 0.70:
        return "EXCELLENT", "Strong structure -- clusters are well-defined and separated"
    elif score > config.SILHOUETTE_STRONG:
        return "GOOD", "Reasonable structure -- meaningful groupings detected"
    elif score > config.SILHOUETTE_MODERATE:
        return "FAIR", "Weak structure -- clusters overlap substantially"
    else:
        return "POOR", "No substantial structure -- clusters may be artificial"


def _interpret_ari(ari):
    """Map an ARI value to descriptive agreement bands.

    Bands based on Steinley (2004) and Hubert & Arabie (1985).
    """
    if ari > config.ARI_STRONG:
        return "STRONG", "Both models capture similar latent structure"
    elif ari > config.ARI_MODERATE:
        return "MODERATE", "Models capture related but distinct aspects of the data"
    else:
        return "WEAK", "Models may capture fundamentally different constructs"


# ── Public entry point ───────────────────────────────────────────────────


def run_validation(df, labels_kproto, labels_lpa=None, feature_cols=None):
    """Validate cluster solutions with silhouette, ARI, and outlier flagging.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset used for clustering.
    labels_kproto : numpy.ndarray
        Primary cluster labels (typically from K-Prototypes).
    labels_lpa : numpy.ndarray, optional
        Secondary labels (typically from LPA) for cross-model
        comparison via ARI.
    feature_cols : list[str], optional
        Numeric feature columns used for distance computation.
        Defaults to ``config.NUMERIC_COLS``.

    Returns
    -------
    dict
        Keys: ``silhouette_overall``, ``silhouette_per_cluster``,
        ``outlier_flags``, ``ari``, ``ari_interpretation``,
        ``validation_summary``, ``audit_entries``.
    """
    feature_cols = feature_cols or list(config.NUMERIC_COLS)
    audit = []

    # Align lengths -- use only rows present in both df and labels
    n = min(len(df), len(labels_kproto))
    X = df[feature_cols].iloc[:n].values.astype(float)
    primary_labels = np.asarray(labels_kproto[:n])
    k = len(np.unique(primary_labels))

    # ── Centroid distances and outlier flagging ──────────────────────
    distances, centroids = _compute_centroid_distances(X, primary_labels, k)
    threshold = float(np.percentile(distances, config.OUTLIER_PERCENTILE))
    outlier_flags = distances > threshold
    n_outliers = int(outlier_flags.sum())

    audit.append(
        audit_entry(
            "Discover", "Psychometrician", "Outlier flagging",
            {
                "threshold_percentile": config.OUTLIER_PERCENTILE,
                "threshold_value": round(threshold, 4),
                "n_outliers": n_outliers,
                "pct_outliers": round(n_outliers / n * 100, 1),
            },
        )
    )

    # ── Silhouette analysis ─────────────────────────────────────────
    if k > 1:
        sil_overall = float(
            silhouette_score(
                X, primary_labels, metric="euclidean",
                sample_size=min(5000, n),
                random_state=config.SEED,
            )
        )
        sil_samples = silhouette_samples(X, primary_labels, metric="euclidean")
    else:
        sil_overall = 0.0
        sil_samples = np.zeros(n)

    quality, interpretation = _interpret_silhouette(sil_overall)

    sil_per_cluster = {}
    for c in range(k):
        mask = primary_labels == c
        cluster_sils = sil_samples[mask]
        sil_per_cluster[int(c)] = {
            "mean": round(float(cluster_sils.mean()), 4),
            "pct_negative": round(float((cluster_sils < 0).mean() * 100), 1),
        }

    audit.append(
        audit_entry(
            "Discover", "Psychometrician", "Silhouette analysis",
            {
                "global_silhouette": round(sil_overall, 4),
                "quality": quality,
            },
        )
    )

    # ── Cross-model validation (ARI) ────────────────────────────────
    ari = None
    ari_quality = None
    ari_interp = None
    if labels_lpa is not None:
        secondary = np.asarray(labels_lpa[:n])
        ari = float(adjusted_rand_score(primary_labels, secondary))
        ari_quality, ari_interp = _interpret_ari(ari)

        audit.append(
            audit_entry(
                "Discover", "Psychometrician", "Cross-model ARI",
                {"ari": round(ari, 4), "quality": ari_quality},
            )
        )

    # ── Validation summary text ─────────────────────────────────────
    lines = [
        f"Global Silhouette: {sil_overall:.4f} ({quality})",
        f"  Interpretation: {interpretation}",
        f"Outliers flagged (top {config.OUTLIER_PERCENTILE}th pctile): "
        f"{n_outliers} ({n_outliers / n * 100:.1f}%)",
    ]
    if ari is not None:
        lines.append(f"ARI (K-Proto vs LPA): {ari:.4f} ({ari_quality})")
        lines.append(f"  Interpretation: {ari_interp}")
    else:
        lines.append("ARI: N/A (single clustering solution)")

    validation_summary = "\n".join(lines)

    # ── Build reasoning narrative ──────────────────────────────────
    reasoning_parts = [
        f"Validated cluster solution with {k} groups on {n:,} respondents.",
        f"Global silhouette coefficient: {sil_overall:.3f} ({quality}). "
        f"{interpretation}",
    ]
    # Per-cluster detail
    weak_clusters = [
        c for c, d in sil_per_cluster.items()
        if d["mean"] < 0.25
    ]
    if weak_clusters:
        reasoning_parts.append(
            f"Cluster(s) {', '.join(str(c) for c in weak_clusters)} show "
            f"weak internal cohesion (silhouette < 0.25). Members of these "
            f"clusters may not be well-differentiated from adjacent groups."
        )
    reasoning_parts.append(
        f"Outliers: {n_outliers} respondents ({n_outliers/n*100:.1f}%) "
        f"flagged as distant from their cluster centroid (top "
        f"{config.OUTLIER_PERCENTILE}th percentile). These individuals "
        f"may represent qualitatively different experiences not captured "
        f"by the current solution."
    )
    if ari is not None:
        reasoning_parts.append(
            f"Cross-method agreement (ARI): {ari:.3f} ({ari_quality}). "
            f"K-Prototypes and LPA are compared as independent partition solutions "
            f"over the same respondents. {ari_interp}"
        )
    reasoning = " ".join(reasoning_parts)

    return {
        "silhouette_overall": round(sil_overall, 4),
        "silhouette_per_cluster": sil_per_cluster,
        "outlier_flags": outlier_flags,
        "centroid_distances": distances,
        "ari": round(ari, 4) if ari is not None else None,
        "ari_interpretation": ari_interp,
        "validation_summary": validation_summary,
        "reasoning": reasoning,
        "quality": quality,
        "audit_entries": audit,
    }
