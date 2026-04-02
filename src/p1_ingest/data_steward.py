"""
Data Steward Agent -- survey data quality gatekeeper.

Screens raw survey data using the Survey Data Quality Evaluation
Model (SDQEM; Papp et al., 2026) multi-hurdle approach and best
practices from Osborne (2013).  Quality gates include:

1. Schema validation (detect categorical vs. numeric columns)
2. Careless responding detection (longstring, IRV, multi-hurdle)
3. Sparsity gate (columns with >20% missing)
4. Variance gate (items with SD < 0.5 excluded from clustering)
5. Distribution screening (skewness, kurtosis, IQR outliers)

The Data Steward does NOT standardize numeric columns.
Standardization is delegated to downstream agents (K-Prototypes,
LPA) to prevent double-standardization errors.

References
----------
Papp, Baker, Dutcher, & McClelland (2026). The Survey Data Quality
    Evaluation Model. Teaching of Psychology.
Osborne, J. W. (2013). Best Practices in Data Cleaning. SAGE.
Curran, P. G. (2016). Methods for the detection of carelessly
    invalid responses. JESP, 66, 4-19.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from src import config
from src.utils import audit_entry


# ── Private helpers ──────────────────────────────────────────────────────


def _max_longstring(row):
    """Length of the longest run of identical consecutive responses.

    Straight-lining (giving the same answer repeatedly) is a common
    indicator of careless responding in Likert-scale surveys. A
    respondent who selects "3" for all items produces a longstring
    equal to the number of items.
    """
    values = row.dropna().values
    if len(values) == 0:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def _validate_schema(df, cat_cols, num_cols):
    """Verify that expected columns exist and have correct types.

    Returns a dict summarising detected column types and any
    issues found.
    """
    issues = []
    detected_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    detected_num = df.select_dtypes(include=["number"]).columns.tolist()

    for c in cat_cols:
        if c not in df.columns:
            issues.append(f"Expected categorical column '{c}' not found")
    for c in num_cols:
        if c not in df.columns:
            issues.append(f"Expected numeric column '{c}' not found")

    return {
        "detected_categorical": detected_cat,
        "detected_numeric": detected_num,
        "expected_categorical": cat_cols,
        "expected_numeric": num_cols,
        "issues": issues,
    }


def _screen_careless(df, survey_cols):
    """Apply the multi-hurdle careless responding screen.

    Two independent indicators are used when attention-check items
    and response-time data are unavailable (the typical situation
    for this tutorial's synthetic data):

    * **Longstring** -- flags respondents whose longest run of
      identical answers exceeds half the number of items.
    * **Individual Response Variability (IRV)** -- flags
      respondents whose within-person SD across all items is
      near zero (< 0.2), indicating non-differentiated responding.

    A respondent must trigger on >= CARELESS_HURDLES independent
    indicators to be flagged (Curran, 2016).
    """
    result_df = df.copy()
    n_items = len(survey_cols)

    # ---- Longstring ----
    result_df["_longstring"] = result_df[survey_cols].apply(_max_longstring, axis=1)
    longstring_threshold = max(n_items * 0.5, 3)
    result_df["_flag_longstring"] = (
        result_df["_longstring"] > longstring_threshold
    ).astype(int)

    # ---- IRV ----
    result_df["_irv"] = result_df[survey_cols].std(axis=1)
    result_df["_flag_irv"] = (result_df["_irv"] < 0.2).astype(int)

    # ---- Multi-hurdle composite ----
    result_df["_quality_flags"] = (
        result_df["_flag_longstring"] + result_df["_flag_irv"]
    )
    result_df["_is_careless"] = (
        result_df["_quality_flags"] >= config.CARELESS_HURDLES
    )

    summary = {
        "longstring_threshold": longstring_threshold,
        "longstring_flagged": int(result_df["_flag_longstring"].sum()),
        "irv_threshold": 0.2,
        "irv_flagged": int(result_df["_flag_irv"].sum()),
        "multi_hurdle_flagged": int(result_df["_is_careless"].sum()),
        "pct_flagged": round(
            result_df["_is_careless"].mean() * 100, 2
        ),
    }

    return result_df, summary


def _sparsity_gate(df, all_cols):
    """Flag columns that exceed the sparsity threshold.

    Columns with > 20% missing values are unsuitable for
    clustering without substantial imputation.  Remaining
    missing values are imputed: median for numeric, mode or
    'Unknown' for categorical (Osborne, 2013).
    """
    missing_pct = df[all_cols].isnull().mean()
    flagged = missing_pct[missing_pct > config.SPARSITY_THRESHOLD].to_dict()
    flagged = {k: round(v * 100, 2) for k, v in flagged.items()}
    return flagged


def _variance_gate(df, num_cols):
    """Identify low-variance survey items (SD < threshold).

    Consensus items where virtually every respondent gave the same
    answer carry no discriminating power for clustering and are
    excluded from downstream analysis.
    """
    stats = {
        col: {
            "sd": round(float(df[col].std()), 4),
            "range": round(float(df[col].max() - df[col].min()), 4),
            "unique": int(df[col].nunique()),
        }
        for col in num_cols
    }
    excluded = [c for c, s in stats.items() if s["sd"] < config.VARIANCE_THRESHOLD_SD]
    return stats, excluded


def _distribution_screen(df, num_cols):
    """Screen numeric columns for problematic skewness, kurtosis, and IQR outliers.

    Flags are informational; the Data Steward does not auto-
    transform or remove outliers.  Downstream agents must decide
    how to handle non-normality (Osborne, 2013, Ch. 7-8).
    """
    report = []
    for col in num_cols:
        data = df[col].dropna()
        sk = float(skew(data))
        kt = float(kurtosis(data))  # excess kurtosis

        q1 = float(data.quantile(0.25))
        q3 = float(data.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((data < lower) | (data > upper)).sum())

        issues = []
        if abs(sk) > 2.0:
            issues.append(f"skew={sk:.2f}")
        if abs(kt) > 7.0:
            issues.append(f"kurtosis={kt:.2f}")
        if n_outliers > len(data) * 0.05:
            issues.append(
                f"{n_outliers} IQR outliers ({n_outliers / len(data) * 100:.1f}%)"
            )

        report.append(
            {
                "column": col,
                "skewness": round(sk, 4),
                "kurtosis": round(kt, 4),
                "iqr_outliers": n_outliers,
                "issues": issues,
            }
        )

    return report


# ── Public entry point ───────────────────────────────────────────────────


def run_data_steward(df, survey_cols=None, cat_cols=None):
    """Run the full Data Steward quality-screening pipeline.

    This is the single public entry point for the agent.  It
    performs schema validation, careless-responding detection,
    sparsity and variance gates, and distribution screening,
    then returns a clean DataFrame together with a structured
    quality report and audit trail.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw survey data (baseline or follow-up).
    survey_cols : list[str], optional
        Numeric survey item columns.  Defaults to
        ``config.NUMERIC_COLS``.
    cat_cols : list[str], optional
        Categorical demographic columns.  Defaults to
        ``config.CATEGORICAL_COLS``.

    Returns
    -------
    dict
        Keys: ``clean_df``, ``removed_count``, ``quality_report``,
        ``audit_entries``.
    """
    survey_cols = survey_cols or list(config.NUMERIC_COLS)
    cat_cols = cat_cols or list(config.CATEGORICAL_COLS)
    audit = []
    original_n = len(df)

    # ── 1. Schema validation ────────────────────────────────────────
    schema_report = _validate_schema(df, cat_cols, survey_cols)
    audit.append(
        audit_entry(
            "Ingest", "Data Steward", "Schema validation",
            {"issues": schema_report["issues"]},
        )
    )

    # ── 2. Careless responding screen ───────────────────────────────
    df_screened, careless_summary = _screen_careless(df, survey_cols)

    # Remove careless respondents
    clean_mask = ~df_screened["_is_careless"]
    df_clean = df_screened.loc[clean_mask].copy()
    removed_count = int((~clean_mask).sum())

    audit.append(
        audit_entry(
            "Ingest", "Data Steward", "Careless responding screen",
            careless_summary,
        )
    )

    # Drop internal screening columns
    internal_cols = [c for c in df_clean.columns if c.startswith("_")]
    df_clean.drop(columns=internal_cols, inplace=True)

    # ── 3. Sparsity gate ────────────────────────────────────────────
    all_analysis_cols = [
        c for c in cat_cols + survey_cols if c in df_clean.columns
    ]
    sparse_cols = _sparsity_gate(df_clean, all_analysis_cols)
    audit.append(
        audit_entry(
            "Ingest", "Data Steward", "Sparsity gate",
            {"columns_exceeding_threshold": sparse_cols},
        )
    )

    # Impute remaining missing values (columns that passed the gate)
    for col in survey_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in cat_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna("Unknown")

    # ── 4. Variance gate ────────────────────────────────────────────
    retained_survey = [c for c in survey_cols if c in df_clean.columns]
    var_stats, excluded_cols = _variance_gate(df_clean, retained_survey)
    retained_survey = [c for c in retained_survey if c not in excluded_cols]

    audit.append(
        audit_entry(
            "Ingest", "Data Steward", "Variance gate",
            {
                "excluded_low_variance": excluded_cols,
                "retained_survey_cols": retained_survey,
            },
        )
    )

    # ── 5. Distribution screening ───────────────────────────────────
    dist_report = _distribution_screen(df_clean, retained_survey)
    audit.append(
        audit_entry(
            "Ingest", "Data Steward", "Distribution screening",
            {
                "columns_with_issues": [
                    d["column"] for d in dist_report if d["issues"]
                ]
            },
        )
    )

    # ── 6. Confidence score ─────────────────────────────────────────
    cols_passing = len(retained_survey) / max(len(survey_cols), 1)
    rows_passing = len(df_clean) / max(original_n, 1)
    confidence = round(cols_passing * rows_passing, 4)

    # ── Assemble quality report ─────────────────────────────────────
    quality_report = {
        "schema": schema_report,
        "careless_responding": careless_summary,
        "sparsity": {"columns_flagged": sparse_cols},
        "variance": {"stats": var_stats, "excluded": excluded_cols},
        "distributions": dist_report,
        "retained_survey_cols": retained_survey,
        "retained_categorical_cols": cat_cols,
        "original_rows": original_n,
        "clean_rows": len(df_clean),
        "removed_count": removed_count,
        "confidence": confidence,
    }

    # ── Build reasoning narrative ──────────────────────────────────
    removal_pct = round(removed_count / max(original_n, 1) * 100, 1)
    reasoning_parts = [
        f"Screened {original_n:,} respondents through a multi-hurdle "
        f"quality protocol (Papp et al., 2026; Curran, 2016).",
        f"Removed {removed_count} respondents ({removal_pct}%) for "
        f"careless responding. Each tripped {config.CARELESS_HURDLES}+ "
        f"independent hurdles: longstring analysis flagged "
        f"{careless_summary['longstring_flagged']}, IRV analysis flagged "
        f"{careless_summary['irv_flagged']}.",
    ]
    if removal_pct < 2:
        reasoning_parts.append(
            "The removal rate is below 2%, which is unusually low. "
            "Consider whether screening thresholds are too conservative."
        )
    elif removal_pct > 8:
        reasoning_parts.append(
            "The removal rate exceeds 8%, which is above the typical "
            "range for organizational surveys. Investigate whether data "
            "collection conditions contributed to elevated careless responding."
        )
    else:
        reasoning_parts.append(
            f"The {removal_pct}% removal rate falls within the typical "
            f"2-8% range for organizational surveys (Curran, 2016)."
        )
    if sparse_cols:
        reasoning_parts.append(
            f"Sparsity gate: {len(sparse_cols)} column(s) exceeded the "
            f"{config.SPARSITY_THRESHOLD * 100:.0f}% missing threshold: "
            f"{', '.join(sparse_cols)}."
        )
    else:
        reasoning_parts.append(
            "All columns passed the sparsity gate (no column exceeded "
            f"{config.SPARSITY_THRESHOLD * 100:.0f}% missing)."
        )
    if excluded_cols:
        reasoning_parts.append(
            f"Variance gate: {len(excluded_cols)} item(s) excluded for "
            f"SD below {config.VARIANCE_THRESHOLD_SD}: {', '.join(excluded_cols)}. "
            f"Low-variance items cannot discriminate between groups."
        )
    else:
        reasoning_parts.append(
            "All survey items passed the variance gate (SD >= "
            f"{config.VARIANCE_THRESHOLD_SD})."
        )
    reasoning_parts.append(
        f"Final clean dataset: {len(df_clean):,} respondents, "
        f"{len(retained_survey)} survey items retained. "
        f"Data quality confidence: {confidence:.2%}."
    )
    reasoning = " ".join(reasoning_parts)

    return {
        "clean_df": df_clean,
        "removed_count": removed_count,
        "quality_report": quality_report,
        "reasoning": reasoning,
        "audit_entries": audit,
    }
