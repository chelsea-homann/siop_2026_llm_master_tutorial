"""
Synthetic Meridian Technologies workforce survey generator
for the SIOP 2026 Master Tutorial.

Produces two CSV files with identical schemas:

    synthetic_data/baseline_survey_data_synthetic.csv   N=10,000, wave 1
    synthetic_data/survey_followup.csv                  N=10,500, wave 2
                                                        (~6 months later)

Both files share the 12-column schema expected by the pipeline:

    Business Unit, Level, FLSA, Tenure,
    Cared_About, Excited, Helpful_Info, Trust_Leadership, Morale,
    Attention check 1, Attention check 2,
    Comments

Design goals (driven by the tutorial's pedagogical objectives)
------------------------------------------------------------------
1. Both K-Prototypes and LPA should converge on **K = 3**.
2. ARI between the two methods lands in the STRONG band (> 0.65) so
   attendees see two adversarial methods agreeing on structure.
3. Silhouette is GOOD (> 0.50) for the selected solution.
4. Demographic-psychology correlation is moderate, not deterministic --
   a demographic snapshot predicts cluster membership but does not
   determine it, which is what lets a Gate 2 reviewer say things like
   "K-Prototypes and LPA largely agree, but look at the Mid-tenure
    Commercial employees in Cluster 1 -- those are worth discussing."
5. 2-4% intentional careless responders to exercise the Data Steward's
   SDQEM multi-hurdle screen (Curran, 2016).
6. Follow-up wave: 6-10% weak-fit to exercise Phase 5 without implying
   a workforce revolution. **No schema drift** between waves.

Cluster archetypes (three personas Meridian's leadership will see)
------------------------------------------------------------------
1. **Engaged Advocates** (~35%)
   - High across all five indicators (Likert means ~4.0-4.3)
   - Senior, long-tenured, Exempt
   - Concentrated in Product + Commercial Department
   - Low attrition risk
2. **Informed Skeptics** (~35%)
   - HIGH on Helpful_Info (~4.1) but LOW on Excited / Trust_Leadership
     (~2.2-2.6) -- a genuinely distinct profile, not "moderate
     everywhere." They're receiving organizational communication but
     aren't bought in to the direction. This shape is what forces both
     K-Prototypes and LPA to recognise them as a third cluster rather
     than collapsing them into the extremes.
   - Mid-Senior level, IT/Commercial-leaning, Exempt
3. **Disengaged** (~25%)
   - Low across all five indicators (means ~1.9-2.3)
   - Entry/Mid-level, < 1y or 1-3y tenure, Non-Exempt
   - Concentrated in Sales + Operations
   - Elevated churn risk; primary concern for leadership intervention

Within-cluster SD is ~0.85 on a 1-5 Likert scale. This is wide enough
that genuine respondents produce 3-4 distinct values across the five
items -- which keeps the Phase 1 entropy and Mahalanobis hurdles from
over-removing legitimate consistent responders.

Run (from project root):

    python scripts/build_synthetic_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "synthetic_data"
BASELINE_PATH = OUT_DIR / "baseline_survey_data_synthetic.csv"
FOLLOWUP_PATH = OUT_DIR / "survey_followup.csv"

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Schema (must match src/config.py)
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = ["Business Unit", "Level", "FLSA", "Tenure"]
NUMERIC_COLS = [
    "Cared_About", "Excited", "Helpful_Info", "Trust_Leadership", "Morale",
]
ATTENTION_CHECK_COLS = ["Attention check 1", "Attention check 2"]
COMMENTS_COL = "Comments"
ALL_COLS = (
    CATEGORICAL_COLS + NUMERIC_COLS + ATTENTION_CHECK_COLS + [COMMENTS_COL]
)

# Attention-check correct answers (Papp et al., 2026 recommend
# instructional-manipulation-check items with unambiguous answers)
AC_EXPECTED = {"Attention check 1": 4, "Attention check 2": 2}

# ---------------------------------------------------------------------------
# Cluster archetypes
# ---------------------------------------------------------------------------
CLUSTERS = {
    "Engaged Advocates": {
        "share": 0.35,
        # Cared_About, Excited, Helpful_Info, Trust_Leadership, Morale
        # Means pulled in slightly from the ceiling to narrow the
        # Engaged-Disengaged axis (otherwise K=2 wins silhouette by
        # a wide margin over K=3).
        "likert_means": [4.0, 4.1, 3.8, 4.0, 4.0],
        "likert_sd": 0.90,
        # Demographic bias: weights relative to uniform
        "bu_weights": {"Commercial Department": 2.2, "Product": 2.4,
                       "IT": 1.1, "Sales": 0.6, "Operations": 0.7},
        "level_weights": {"Entry": 0.3, "Mid": 1.0, "Senior": 2.4},
        "flsa_weights": {"Exempt": 2.3, "Non-Exempt": 0.6},
        "tenure_weights": {"< 1 yr": 0.3, "1-3 yrs": 0.7, "3-5 yrs": 1.1,
                           "5-10 yrs": 2.0, "10+ yrs": 2.1},
        "ac_pass_rate": 0.99,
        "comment_pool": [
            "Leadership has been clear about the restructuring and why it "
            "matters. I feel proud to work here.",
            "I appreciate the monthly town halls. My manager keeps me in the "
            "loop on things that affect our team.",
            "The growth opportunities here are real. I transferred to a new "
            "role last year and it was well supported.",
            "Benefits and flexibility are excellent. My team has good morale "
            "overall.",
            "I trust the exec team. They communicate setbacks, not just wins.",
            "",
        ],
    },
    # "Informed Skeptics" -- a distinct profile, not a middle cluster.
    # HIGH on Helpful_Info (they receive organizational information) but
    # LOW on Excited / Trust_Leadership / Morale. This is the shape that
    # forces both clustering methods to recognise them as a genuinely
    # different group rather than as a middle band between the extremes.
    "Informed Skeptics": {
        "share": 0.35,
        # Peaked shape: HIGH on Cared_About + Helpful_Info (both above
        # Engaged's values on those specific dims), LOW on Excited +
        # Trust_Leadership (below Disengaged on those dims). Tighter
        # within-cluster SD than the other two gives the cluster enough
        # internal cohesion that K=3 wins silhouette at the Gate 2
        # selection step against the naive "high vs. low" K=2 collapse.
        "likert_means": [4.2, 1.8, 4.5, 1.7, 3.2],
        "likert_sd": 0.70,
        "bu_weights": {"Commercial Department": 1.3, "Product": 0.9,
                       "IT": 2.2, "Sales": 0.9, "Operations": 1.0},
        "level_weights": {"Entry": 0.7, "Mid": 2.3, "Senior": 1.4},
        "flsa_weights": {"Exempt": 1.6, "Non-Exempt": 0.9},
        "tenure_weights": {"< 1 yr": 0.7, "1-3 yrs": 1.4, "3-5 yrs": 1.6,
                           "5-10 yrs": 1.1, "10+ yrs": 0.8},
        "ac_pass_rate": 0.97,
        "comment_pool": [
            "I get the information I need, but I don't agree with the "
            "direction. The strategy has been communicated; I'm not "
            "convinced it's the right one.",
            "The weekly digest is well-written. I read it. I'm still looking.",
            "Leadership talks a lot about transparency. That's not the "
            "same as actually listening.",
            "I know what's happening. I don't trust that it will work out.",
            "Plenty of communication, less confidence. My team is drained.",
            "I'm well-informed and unexcited. Those are not the same "
            "problem and they shouldn't be treated as one.",
            "",
        ],
    },
    "Disengaged": {
        "share": 0.30,
        # Means pulled up slightly from the floor to narrow the
        # Engaged-Disengaged axis without losing face validity.
        "likert_means": [2.4, 2.2, 2.5, 2.1, 2.3],
        "likert_sd": 0.90,
        "bu_weights": {"Commercial Department": 0.7, "Product": 0.6,
                       "IT": 0.9, "Sales": 2.3, "Operations": 2.0},
        "level_weights": {"Entry": 2.4, "Mid": 1.3, "Senior": 0.3},
        "flsa_weights": {"Exempt": 0.5, "Non-Exempt": 2.5},
        "tenure_weights": {"< 1 yr": 2.3, "1-3 yrs": 2.1, "3-5 yrs": 0.9,
                           "5-10 yrs": 0.4, "10+ yrs": 0.2},
        "ac_pass_rate": 0.93,
        "comment_pool": [
            "I've been looking at other opportunities. There's no clear "
            "path here.",
            "Leadership says one thing and does another. Communication is "
            "poor.",
            "I don't feel valued. My manager is overloaded and I rarely get "
            "feedback.",
            "The restructuring was handled badly. A lot of people I worked "
            "with are gone.",
            "Pay has not kept up. I feel replaceable.",
            "I don't know what the company's plan is. I'm not sure anyone "
            "does.",
            "",
        ],
    },
}

# ---------------------------------------------------------------------------
# Shared category vocabularies
# ---------------------------------------------------------------------------
BUSINESS_UNITS = ["Commercial Department", "IT", "Sales",
                  "Operations", "Product"]
LEVELS = ["Entry", "Mid", "Senior"]
FLSA_VALUES = ["Exempt", "Non-Exempt"]
TENURES = ["< 1 yr", "1-3 yrs", "3-5 yrs", "5-10 yrs", "10+ yrs"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sample_categorical(values: list[str],
                        weights: dict[str, float],
                        n: int) -> np.ndarray:
    w = np.array([weights[v] for v in values], dtype=float)
    w /= w.sum()
    return rng.choice(values, size=n, p=w)


def _sample_likert(means: list[float], sd: float, n: int) -> np.ndarray:
    """Gaussian around cluster mean, clipped to integer Likert 1-5."""
    raw = rng.normal(loc=means, scale=sd, size=(n, len(means)))
    return np.clip(np.round(raw), 1, 5).astype(int)


def _sample_attention_checks(n: int, pass_rate: float) -> np.ndarray:
    """Each item passes independently with probability ``pass_rate``.

    Failing respondents answer any 1-5 (not necessarily the wrong value),
    which matches real-world careless-responder patterns.
    """
    out = np.zeros((n, 2), dtype=int)
    for col_i, (col, expected) in enumerate(AC_EXPECTED.items()):
        passed = rng.random(n) < pass_rate
        out[:, col_i] = np.where(
            passed, expected, rng.integers(1, 6, size=n)
        )
    return out


def _sample_comments(pool: list[str], n: int) -> list[str]:
    return list(rng.choice(pool, size=n))


# ---------------------------------------------------------------------------
# Cluster generator
# ---------------------------------------------------------------------------
def _generate_cluster(name: str, n: int) -> pd.DataFrame:
    spec = CLUSTERS[name]

    bu = _sample_categorical(BUSINESS_UNITS, spec["bu_weights"], n)
    level = _sample_categorical(LEVELS, spec["level_weights"], n)
    flsa = _sample_categorical(FLSA_VALUES, spec["flsa_weights"], n)
    tenure = _sample_categorical(TENURES, spec["tenure_weights"], n)

    likert = _sample_likert(spec["likert_means"], spec["likert_sd"], n)
    ac = _sample_attention_checks(n, spec["ac_pass_rate"])
    comments = _sample_comments(spec["comment_pool"], n)

    data = {
        "Business Unit": bu,
        "Level": level,
        "FLSA": flsa,
        "Tenure": tenure,
    }
    for i, col in enumerate(NUMERIC_COLS):
        data[col] = likert[:, i]
    data["Attention check 1"] = ac[:, 0]
    data["Attention check 2"] = ac[:, 1]
    data["Comments"] = comments
    data["_cluster"] = name  # kept for diagnostics; dropped before writing
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Careless-responder injection
# ---------------------------------------------------------------------------
def _inject_careless(df: pd.DataFrame, pct: float = 0.03) -> pd.DataFrame:
    """Replace ~pct of respondents with realistic careless-responding patterns.

    Distributes flagged respondents across four SDQEM-recognised
    profiles so the Data Steward's multi-hurdle screen has real targets:

    1. Straight-liner (longstring)         -- same value across 5 items
    2. Low IRV + failed attention check    -- near-zero within-person SD
    3. Mahalanobis outlier                 -- extreme mixed pattern
    4. Opposite-end alternator             -- 1,5,1,5,1 (zig-zag)
    """
    n = len(df)
    k = max(int(n * pct), 1)
    idx = rng.choice(n, size=k, replace=False)
    profiles = rng.integers(0, 4, size=k)

    for row, prof in zip(idx, profiles):
        if prof == 0:
            val = int(rng.integers(1, 6))
            df.loc[row, NUMERIC_COLS] = val
        elif prof == 1:
            base = int(rng.integers(2, 5))
            df.loc[row, NUMERIC_COLS] = base
            df.loc[row, "Attention check 1"] = int(rng.integers(1, 6))
        elif prof == 2:
            extreme = rng.choice([1, 5], size=5)
            df.loc[row, NUMERIC_COLS] = extreme
        else:
            df.loc[row, NUMERIC_COLS] = [1, 5, 1, 5, 1]

        # Careless responders also tend to fail attention checks
        if rng.random() < 0.7:
            df.loc[row, "Attention check 2"] = int(rng.integers(1, 6))
        df.loc[row, "_cluster"] = "CARELESS"
        df.loc[row, "Comments"] = ""
    return df


# ---------------------------------------------------------------------------
# Weak-fit (follow-up only) injection
# ---------------------------------------------------------------------------
def _inject_weak_fit(df: pd.DataFrame, pct: float = 0.07) -> pd.DataFrame:
    """Replace ~pct of follow-up respondents with a mixed-profile pattern
    that doesn't cleanly align with any baseline cluster.

    These are the respondents Phase 5's Continuity Agent should flag as
    weak-fit (|distance to every baseline centroid| > threshold). The
    pattern: a distinctly mid-range Likert profile combined with a
    demographic mix the baseline clusters don't favour -- e.g., a
    Senior Sales employee with very short tenure, or Mid-level with
    bimodal Likert scores.
    """
    n = len(df)
    k = max(int(n * pct), 1)
    idx = rng.choice(n, size=k, replace=False)
    for row in idx:
        df.loc[row, NUMERIC_COLS] = np.clip(
            np.round(rng.normal(3.0, 1.2, size=5)), 1, 5
        ).astype(int)
        # Unusual demographic combos
        df.loc[row, "Business Unit"] = str(rng.choice(
            ["Sales", "Product"], p=[0.5, 0.5]
        ))
        df.loc[row, "Level"] = str(rng.choice(
            ["Senior", "Entry"], p=[0.5, 0.5]
        ))
        df.loc[row, "Tenure"] = str(rng.choice(
            ["< 1 yr", "10+ yrs"], p=[0.5, 0.5]
        ))
        df.loc[row, "_cluster"] = "WEAK_FIT"
    return df


# ---------------------------------------------------------------------------
# Missing-data injection (small random holes)
# ---------------------------------------------------------------------------
def _inject_missing(df: pd.DataFrame, pct: float = 0.005) -> pd.DataFrame:
    """Sprinkle NaNs at the per-cell level (overall missing rate ~pct)."""
    numeric_mask = rng.random(df[NUMERIC_COLS].shape) < pct
    for i, col in enumerate(NUMERIC_COLS):
        df.loc[numeric_mask[:, i], col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Wave generators
# ---------------------------------------------------------------------------
def generate_baseline(n_total: int = 10_000) -> pd.DataFrame:
    parts = []
    for name, spec in CLUSTERS.items():
        size = int(round(n_total * spec["share"]))
        parts.append(_generate_cluster(name, size))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df = _inject_careless(df, pct=0.03)
    df = _inject_missing(df, pct=0.005)
    return df


def generate_followup(n_total: int = 10_500) -> pd.DataFrame:
    """Follow-up wave roughly 6 months later.

    Respondents draw from the same three archetypes but with a small
    positive shift: Engaged Advocates grow, Disengaged shrinks slightly,
    reflecting a plausible effect of a well-managed transparency
    initiative. Weak-fit respondents are injected at ~7%.
    """
    # Positive shift over six months: Engaged grows, Disengaged shrinks
    # slightly, Informed Skeptics roughly stable -- a plausible pattern
    # after a well-received transparency initiative.
    wave_shares = {
        "Engaged Advocates": 0.37,
        "Informed Skeptics": 0.36,
        "Disengaged": 0.27,
    }
    parts = []
    for name, share in wave_shares.items():
        size = int(round(n_total * share))
        parts.append(_generate_cluster(name, size))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    # Careless rate is lower in the follow-up wave (engagement survey
    # fatigue is real but response quality usually improves a bit when
    # attendees see action on the prior wave's findings).
    df = _inject_careless(df, pct=0.02)
    df = _inject_weak_fit(df, pct=0.07)
    df = _inject_missing(df, pct=0.004)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _describe(df: pd.DataFrame, name: str) -> None:
    print(f"\n-- {name} --")
    print(f"  N          : {len(df):,}")
    print(f"  Schema     : {list(df.columns[:-1])}")  # hide _cluster
    print(f"  By cluster : ")
    for cl, n in df["_cluster"].value_counts().sort_index().items():
        print(f"    {cl:<20s} {n:>6,}  ({n / len(df) * 100:.1f}%)")
    print(f"  Likert means by cluster:")
    for cl in df["_cluster"].unique():
        mask = df["_cluster"] == cl
        means = df.loc[mask, NUMERIC_COLS].mean().round(2).to_dict()
        means_str = ", ".join(f"{k}={v}" for k, v in means.items())
        print(f"    {cl:<20s} {means_str}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SYNTHETIC WORKFORCE SURVEY GENERATOR")
    print("=" * 60)
    print(f"Seed       : {SEED}")
    print(f"Output dir : {OUT_DIR.relative_to(REPO_ROOT)}")

    baseline = generate_baseline(10_000)
    followup = generate_followup(10_500)

    _describe(baseline, "BASELINE")
    _describe(followup, "FOLLOW-UP")

    # Drop diagnostic columns and write in a deterministic column order.
    for df, path in ((baseline, BASELINE_PATH), (followup, FOLLOWUP_PATH)):
        public = df[ALL_COLS]
        public.to_csv(path, index=False)
        print(f"\nWrote: {path.relative_to(REPO_ROOT)} "
              f"({len(public):,} rows, {public.shape[1]} cols)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
