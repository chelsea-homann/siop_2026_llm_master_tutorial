"""
Pipeline configuration for the SIOP 2026 tutorial.

Controls model selection, file paths, and analysis thresholds.
An ANTHROPIC_API_KEY environment variable must be set to run
the pipeline.

Thresholds are drawn from the Project Manager Agent's Global
Configuration Registry and grounded in published I-O psychology
best practices.
"""

import os

PIPELINE_VERSION = "0.2.0"

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Random seed for reproducibility across all agents
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------------------
DATA_DIR = "synthetic_data"
ORG_DOCS_DIR = "synthetic_data/org_documents"
OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# Quality-gate thresholds (Project Manager Global Configuration Registry)
# ---------------------------------------------------------------------------

# Data Steward: columns with more than 20% missing values are flagged
SPARSITY_THRESHOLD = 0.20

# Data Steward: survey items with SD below this are near-consensus and
# excluded from clustering (they cannot discriminate between groups)
VARIANCE_THRESHOLD_SD = 0.5

# Continuity Agent: respondents whose composite distance to the nearest
# baseline centroid exceeds this value are flagged as "Weak-Fit"
WEAK_FIT_DISTANCE = 0.35

# Psychometrician: silhouette interpretation bands (Rousseeuw, 1987)
SILHOUETTE_STRONG = 0.50
SILHOUETTE_MODERATE = 0.25

# LPA Agent: respondents whose maximum posterior probability falls below
# this threshold are flagged as "Psychologically Ambiguous"
LPA_AMBIGUITY_POSTERIOR = 0.70

# Psychometrician: observations above this distance percentile from their
# cluster centroid are flagged as outliers
OUTLIER_PERCENTILE = 90

# Data Steward: respondents must be flagged on at least this many
# independent careless-responding indicators to be removed (Curran, 2016)
CARELESS_HURDLES = 2

# Psychometrician: ARI interpretation bands (Steinley, 2004; Hubert & Arabie, 1985)
ARI_STRONG = 0.65
ARI_MODERATE = 0.30

# ---------------------------------------------------------------------------
# Survey schema -- matches the synthetic Meridian Technologies dataset
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = ["Business Unit", "Level", "FLSA", "Tenure"]
NUMERIC_COLS = [
    "Cared_About",
    "Excited",
    "Helpful_Info",
    "Trust_Leadership",
    "Morale",
]
ATTENTION_CHECK_COLS = ["Attention check 1", "Attention check 2"]
COMMENTS_COL = "Comments"
