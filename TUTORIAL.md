# Human-Centered Personas at Scale — Tutorial

**SIOP 2026 Master Tutorial | New Orleans, LA | April 30, 2026**
Chelsea Wymer | Diana Wolfe | Alice Choe

> A five-phase pipeline that pairs statistical rigor with human judgment. The agents do the computation. You make the judgments.

---

## How to Read This Document

Each phase follows the same three-step structure that mirrors the React tutorial artifact you've seen:

- **The Science** — the I-O psychology behind the method and why it matters during disruption.
- **Step 1 → Step 2 → Step 3** — upload / compute / review. You run one script per phase.
- **Gate N: Your Decision** — three mutually exclusive options with an evidence summary. The gate is not a rubber stamp; it is the point of the pipeline.

Every phase ends with a routing callout telling you where the next phase picks up.

## Orientation

```
survey-clustering/
├── TUTORIAL.md                         ← you are here
├── scripts/
│   ├── run_phase1_data_steward.py      Phase 1: ingest + clean (Data Steward)
│   ├── run_phase2_discover.py          Phase 2: K-Prototypes + LPA + Psychometrician
│   ├── run_phase3_ground.py            Phase 3: RAG + Emergence
│   ├── run_phase4_narrate.py           Phase 4: Narrator + Ethics + Project Manager
│   └── run_phase5_longitudinal.py      Phase 5: Continuity + Emergence (longitudinal)
├── src/                                agent implementations by phase
├── synthetic_data/
│   ├── baseline_survey_data_synthetic.csv
│   ├── survey_followup.csv
│   └── org_documents/
└── outputs/                            every phase writes here
    └── phaseN_*/
        ├── report.md                   human-readable Gate review
        ├── summary.json                machine-readable results
        ├── audit_reports/              per-phase screening detail + bias audits
        └── reflection_logs/            agent self-reflection + success report
```

Each runner script is self-contained. It reads the output of the previous phase, runs its agents, writes artifacts into `outputs/phaseN_*/`, and prints a Gate N review checklist at the end.

## Two modes: mock and live

The pipeline auto-detects which mode to run in based on your environment.

| Condition | Mode | Phase 3 RAG | Phase 4 Narrator |
|---|---|---|---|
| `ANTHROPIC_API_KEY` **not** set | **MOCK** | loads pre-generated JSON | loads pre-generated JSON |
| `ANTHROPIC_API_KEY` **is** set  | **LIVE** | calls Claude Sonnet 4 for relevance | calls Claude Sonnet 4 for narratives |

Phases 1, 2, and 5 are pure statistics and behave identically in both modes. Only Phases 3 and 4 differ.

To run live, set `ANTHROPIC_API_KEY` as a persistent Windows user environment variable (see `SETUP.md`) and fully restart VS Code / Claude Code so child processes inherit the new value.

---

# 📊 Phase 1: Ingest and Clean

**Agent:** Data Steward
**Subtitle:** Survey Data Quality Gatekeeper — SDQEM Multi-Hurdle Framework (Papp et al., 2026)

## The Science

Before any analysis, survey data must be screened for quality using the **Survey Data Quality Evaluation Model (SDQEM)**. Raw survey data contains noise: respondents who rush through with identical answers, respondents who give the same number to every item, incomplete responses, and items where every respondent answered the same way and therefore cannot discriminate between groups. Five sequential quality gates ensure the data is clean, representative, and trustworthy:

1. **Schema validation** — expected numeric and categorical columns are present and typed correctly.
2. **Careless responding** — multi-hurdle: ≥ 2 independent flags required before removal (Curran, 2016). Hurdles: attention-check failures, longstring (> 50% of items), low Individual Response Variability (SD < 0.2), Mahalanobis outliers on the Likert items, and low response entropy.
3. **Sparsity gate** — columns with > 20% missing are flagged.
4. **Variance gate** — items with SD < 0.5 are excluded from clustering (Osborne, 2013). Low-variance items cannot discriminate between groups.
5. **Distribution screening** — skewness > |2| and kurtosis > |7| are flagged for downstream agents.

The Data Steward deliberately does **not** standardize numeric columns. Z-scoring is delegated to the K-Prototypes and LPA agents to prevent double-scaling errors.

> **Routing:** New baseline → K-Prototypes Agent. Follow-up survey → Continuity Agent.

## Step 1 of 3 — Upload Baseline Survey

```bash
python scripts/run_phase1_data_steward.py
```

The script reads `synthetic_data/baseline_survey_data_synthetic.csv`. No other input is needed.

## Step 2 of 3 — Quality Report

The Data Steward writes every artifact into `outputs/phase1_data_quality_report/`:

```
outputs/phase1_data_quality_report/
├── survey_baseline_clean.csv              ← the analytic dataset
├── report.md                              ← Gate 1 review
├── summary.json                           ← machine-readable
├── audit_reports/
│   ├── variance_heatmap.svg               SD per Likert item; red = excluded
│   ├── data_quality_screening.md          per-column hurdle detail
│   └── data_steward_bias_audit.md         demographic shifts before vs. after cleaning
└── reflection_logs/
    ├── data_steward_reflection.json       structured self-reflection
    └── data_steward_success_report.txt    status, metrics, artifacts produced
```

Open `report.md` first. It opens with a **Data Quality Confidence Score**. If the score is ≥ 0.90, Gate 1 has passed on the numbers; if lower, the report flags exactly what failed.

## Step 3 of 3 — Gate 1: Your Decision

> **Do you accept this cleaned dataset for Phase 2 clustering?**

**Evidence to consider**

- Removal rate within the typical **2–8% band** for organizational surveys (Curran, 2016)
- No demographic group disproportionately removed (see `data_steward_bias_audit.md`)
- Variance gate passed: SD > 0.5 on every retained item
- Missing data < 20% on every retained column
- Distribution flags noted for Phase 2 (skew, kurtosis, IQR outliers are informational — Osborne, 2013)

**Your three options**

| Option | When to choose it | What happens |
|---|---|---|
| ✓ **Yes — Accept Screening Results** | Score ≥ 0.90 and demographic parity holds | Clean dataset routes to K-Prototypes Agent |
| 🔍 **Investigate** | Concerns you want noted but not blocking | Flag concerns, continue; adjustments recorded in audit trail |
| ✗ **No — Reject** | Data quality confidence below 0.90, or bias audit flags demographic distortion | Return to data collection. Pipeline halted by Project Manager Agent |

---

# 🎯 Phase 2: Discover Workforce Segments

**Agents:** K-Prototypes, LPA, Psychometrician
**Subtitle:** Two independent clustering methods. One validated truth.

## The Science

Now that we have clean data, we need to discover the natural workforce segments. We run **two independent clustering methods** that tell different stories about the same people:

- **K-Prototypes** (Huang, 1998) — clusters on **demographics + survey items together** using a gamma-weighted cost function that combines Euclidean distance for numeric features with Hamming distance for categorical features. Answers *who are these people?*
- **Latent Profile Analysis** via Gaussian Mixture Models (Spurk et al., 2020; Nylund et al., 2007) — clusters on **survey items only**. Ignores demographics deliberately. Answers *what psychological profiles exist?*

This is deliberately adversarial: if two very different algorithms find similar groups, you can be more confident those groups are real. If they disagree, that disagreement is itself an informative finding.

The **Psychometrician Agent** then validates both solutions:

- **Silhouette coefficient** (Rousseeuw, 1987) — measures cluster separation. Interpretation bands: > 0.70 = strong, > 0.50 = GOOD, > 0.25 = FAIR, below = POOR.
- **Adjusted Rand Index** (Hubert & Arabie, 1985) — cross-method agreement corrected for chance. Bands: > 0.65 = STRONG, > 0.30 = MODERATE, below = WEAK.
- **Centroid-distance outlier flagging** — respondents in the top 10th percentile of distance from their cluster centroid may represent qualitatively different experiences.
- **LPA ambiguity flag** — respondents whose maximum posterior probability is below 0.70 are flagged as *Psychologically Ambiguous*.

Every cluster gets a **Psychological Fingerprint**: High / Moderate / Low labels on each indicator based on Z-scored means.

> **Routing:** Accepted cluster solution → RAG Agent (for policy grounding) and Emergence Agent (for theme discovery).

## Step 1 of 3 — Run Both Methods

```bash
python scripts/run_phase2_discover.py
```

Expect 1–3 minutes. The K-Prototypes elbow search covers K=2 to K=7; the LPA grid covers K=2 to K=6 across diagonal and full covariance.

## Step 2 of 3 — Cluster Validation Report

```
outputs/phase2_cluster_validation/
├── report.md                              ← Gate 2 review
├── summary.json
├── cluster_labels.csv                     per-respondent: kproto, lpa, posterior, outlier
├── kproto_profiles.csv                    one row per K-Prototypes cluster
├── kproto_centroids.json                  needed by Phase 5
├── lpa_profiles.csv
├── lpa_fingerprints.json                  Psychological Fingerprints (High/Mod/Low)
├── audit_reports/
│   └── figures/kproto_elbow.svg
└── reflection_logs/
    └── phase2_success_report.txt
```

## Step 3 of 3 — Gate 2: Your Decision

> **Do you accept this cluster solution (K-Prototypes + LPA consensus)?**

**Evidence Summary**

- Both methods converge on a consistent K (independent evidence) — or they disagree, and the disagreement is substantive
- Silhouette quality band is adequate for downstream personas (**GOOD** or **FAIR** with interpretation)
- ARI captures demographics-vs.-psychology alignment:
  - **HIGH (> 0.65):** both methods see the same structure. Strong confidence.
  - **MODERATE (0.30–0.65):** methods see related but distinct structure. This is the most common and most informative result — demographics and psychology overlap but are not redundant.
  - **WEAK (< 0.30):** demographics and psychology tell different stories. This is itself a finding.
- Outlier rate (top 10th percentile from centroid) does not mask a legitimate sub-group
- LPA ambiguous rate is not elevated (> 15%) without explanation

**Your three options**

| Option | When to choose it | What happens |
|---|---|---|
| ✓ **Yes — Accept Cluster Solution** | Both methods produce defensible solutions; validation bands are acceptable | Clusters route to RAG + Emergence (Phase 3) |
| 🔍 **Investigate** | You want to note a discrepancy (e.g., ARI < 0.30) but proceed with K-Prototypes as primary | Flag in audit trail; continue |
| ✗ **No — Reject** | Silhouette POOR or the solution is unstable in ways you cannot defend to leadership | Return to parameter tuning. Pipeline halted |

---

# 🔍 Phase 3: Ground in Organizational Reality

**Agents:** RAG, Emergence
**Subtitle:** Clusters are statistical abstractions. To become useful to leadership, they must be grounded in organizational context.

## The Science

This phase bridges the gap between numbers and reality by running two agents in parallel:

**🔍 RAG (Retrieval-Augmented Generation)** (Lewis et al., 2020) — *What do organizational documents say?*

- Builds a searchable knowledge base from `synthetic_data/org_documents/` (policy memos, benefits updates, restructuring announcements, team charters, FAQs)
- Chunks each document, embeds via TF-IDF + cosine similarity (the tutorial uses TF-IDF to keep dependencies light; production systems typically use dense embeddings)
- Retrieves relevant passages for each of the **12 I-O psychology codebook constructs**
- In live mode, Claude Sonnet 4 rates each retrieved passage as HIGH / MODERATE / LOW relevance
- Every narrative claim in Phase 4 will trace back to a cited passage

**💡 Emergence Analysis** (Glaser & Strauss, 2017; Braun & Clarke, 2006) — *What patterns fall outside the codebook?*

- Scans cluster profiles for unusual construct combinations (e.g., High-Trust + Low-Morale)
- Classifies each candidate as one of three:
  - **NEW** — a genuinely novel theme not captured by the existing 12 constructs. Consider adding to the codebook.
  - **VARIANT** — a known construct appearing in an unusual combination. Consider adding a sub-label.
  - **NOISE** — a statistical artifact. Exclude.

### The 12 I-O Psychology Constructs

The codebook anchors every classification to validated constructs from the I-O psychology literature:

| Construct | Domain | Core Citation |
|---|---|---|
| Psychological Safety | Team Climate | Edmondson (1999) |
| Organizational Commitment | Attachment | Meyer & Allen (1991) |
| Perceived Organizational Support | Social Exchange | Eisenberger et al. (1986) |
| Change Readiness | Change Management | Armenakis et al. (1993) |
| Role Ambiguity | Role Stress | Rizzo et al. (1970) |
| Leader-Member Exchange (LMX) | Leadership | Graen & Uhl-Bien (1995) |
| Procedural Justice | Justice | Colquitt (2001) |
| Trust in Leadership | Governance | Mayer et al. (1995) |
| Work Engagement | Motivation | Schaufeli et al. (2002) |
| Communication Effectiveness | Sensemaking | Bordia et al. (2004) |
| Career Development | Growth | Kraimer et al. (2011) |
| Work-Life Balance | Well-Being | Greenhaus & Beutell (1985) |

> **Note on the five-item baseline.** The synthetic survey only measures five of these constructs directly (`Cared_About`, `Excited`, `Helpful_Info`, `Trust_Leadership`, `Morale`). The full 12-construct codebook is the theoretical framework; the five items are the empirical proxy. Phase 3 maps cluster profiles back onto the full codebook via organizational-document retrieval.

> **Routing:** Accepted themes and codebook expansion → Narrator Agent (for persona writing). Any NEW theme that survives Gate 3 gets added to the codebook for the next survey wave.

## Step 1 of 3 — Build Knowledge Base and Scan

```bash
python scripts/run_phase3_ground.py
```

Fast — under 30 seconds in mock mode, 1–2 minutes in live mode.

## Step 2 of 3 — Grounding Report

```
outputs/phase3_emergent_themes/
├── report.md                              ← Gate 3 review
├── summary.json
├── construct_grounding.json               per-construct passages + LLM assessment
├── emergent_themes.json                   candidates + NEW/VARIANT/NOISE classification
├── knowledge_base_index.json              document / chunk manifest
├── audit_reports/
│   └── rag_retrieval_detail.md
└── reflection_logs/
    └── phase3_success_report.txt
```

## Step 3 of 3 — Gate 3: Your Decision

> **Do you accept the emergent themes and codebook expansion?**

**Evidence Summary**

- Every construct in the codebook has at least one policy passage with defensible relevance, OR the missing constructs are explicitly flagged
- Policy–experience mismatches are surfaced (these feed the Phase 4 persona narratives)
- Emergent theme classifications are supportable: NEW themes have evidence of > 15% cluster frequency, VARIANT themes map to an existing construct, NOISE themes are genuinely idiosyncratic
- Any **NEW** theme is evidence-based, not stereotype

**Your three options**

| Option | When to choose it | What happens |
|---|---|---|
| ✓ **Accept & Expand Codebook** | NEW themes are defensible and worth tracking in future waves | NEW themes added; personas route to Narrator |
| 📋 **Request Review** | You want another I-O psychologist's read before adding a construct | Flag for review, proceed with existing codebook |
| 🔁 **Revise** | A classification is wrong (e.g., NOISE should be VARIANT) | Re-run Phase 3 after fixing inputs |

---

# ✍️ Phase 4: Write and Validate Personas

**Agents:** Narrator + Ethics Checkpoint + Project Manager Governance
**Subtitle:** Evidence-grounded narratives. Ethics checkpoint required before approval.

## The Science

After clustering, grounding, and emergence analysis, we synthesize everything into **evidence-grounded persona narratives**. Each persona pairs:

- **Psychometric Fingerprint** — key cluster statistics (z-scores, percentiles) directly from Phase 2
- **Verbatim Quotes** — real employee voices from open-ended responses, representing different dimensions (not cherry-picked for eloquence)
- **Policy Citations** — links to organizational documents retrieved in Phase 3
- **Risk Flags** — epistemic uncertainty and anthropomorphic language warnings

### Narrative Principles (Braun & Clarke, 2006)

- ✓ Every claim traces to a statistical centroid or a retrieved policy passage
- ✓ Quotes are verbatim from respondent data (no paraphrasing)
- ✓ Quotes span different dimensions (not just the most extreme or eloquent)
- ✓ Confidence levels are stated (z-scores indicate evidence strength)
- ✓ Anthropomorphic language flagged for human review
- ✓ The I-O psychologist retains final interpretive authority

### Epistemic Risk Mitigation (Nguyen & Welch, 2025)

The Narrator follows a strict protocol to prevent three common failure modes:

1. **Anthropomorphic interpretation** — treating a cluster as a person with desires and intentions rather than a statistical abstraction. No "this cluster *wants*…" — only "scores on X indicate Y observable tendency."
2. **Fabricated quotations** — inventing representative quotes that sound right. The Narrator quotes only from respondent data.
3. **The Oracle Effect** — masking uncertainty with confident language and treating the AI as ground truth. The narrative must state its uncertainty explicitly.

### Ethics Checkpoint (6 Bias Audits — Required Before Gate 4)

Before approving personas for leadership, the pipeline requires a bias audit across six dimensions. Gate 4 is locked until all six are acknowledged:

| # | Bias Type | What You Check |
|---|---|---|
| 📊 | **Input Bias** | Response rate by demographic (all groups ≥ 70%?); no group oversampled; removal rates unbiased |
| 🎯 | **Clustering Bias** | Cluster membership independent of demographics (chi-square test); no cluster = demographic group; silhouette fair across all groups |
| 📝 | **Narrative Bias** | No stereotypical language; claims grounded in data; would employees recognize themselves fairly? |
| 📚 | **Retrieval Bias** | RAG corpus includes exec AND frontline voices; missing perspectives noted (e.g., union, critics); inclusive vs. exclusive phrasing checked |
| ⚠️ | **Epistemic Risk** | Low-confidence claims (z < 1.0) flagged as tentative; Oracle Effect guarded against; recommendations marked with confidence levels |
| 🧠 | **Anthropomorphism** | No "want / feel / desire" without evidence; observable behaviors described, not inferred intentions; correlation vs. causation marked |

> **Routing:** Approved personas → leadership presentation. When a follow-up wave arrives → Phase 5 (Continuity Agent).

## Step 1 of 3 — Generate Narratives

```bash
python scripts/run_phase4_narrate.py
```

## Step 2 of 3 — Persona Narratives

```
outputs/phase4_persona_narratives/
├── personas.md                            ← the deliverable (Gate 4 review)
├── personas.json                          structured
├── personas.csv                           one row per persona with fingerprint columns
├── audit_reports/
│   └── ethics_checkpoint.md               six-dimension bias audit worksheet
└── reflection_logs/
    └── phase4_success_report.txt
```

## Step 3 of 3 — Gate 4: Your Decision

> **Do you approve personas for presentation?**
> Do narratives match statistical fingerprints? Are they evidence-based or over-interpreted? Would an employee be comfortable if they recognized their group?

**Prerequisite:** The ethics checkpoint must be completed (all six bias audits acknowledged) before Gate 4 unlocks.

**Evidence Summary**

- Every claim in every narrative traces to a centroid value or retrieved passage
- Persona names are neutral and non-stigmatising
- Epistemic notes are present on every persona
- Policy-experience mismatches reflect real organizational context, not LLM speculation
- No fabricated quotes, no imputed emotions beyond what scores support
- You would be comfortable standing behind each narrative in front of leadership

**Your three options**

| Option | When to choose it | What happens |
|---|---|---|
| ✅ **Approve** | Ethics checkpoint clean; narratives defensible | Personas ready for leadership presentation |
| 🔁 **Revise** | One or two narratives overreach or misname | Revise the specific narratives; re-approve |
| ✗ **Reject** | A bias audit failed; the narrative frame is unsound | Return to Phase 3 to reconsider grounding |

---

# 📈 Phase 5: Longitudinal Alignment (Bonus)

**Agents:** Continuity + Emergence (Longitudinal Mode)
**Subtitle:** Following the workforce across time.

## The Science

When a follow-up pulse arrives, two questions matter: *who stayed where?* and *are new segments forming?*

**🔗 Continuity Agent** — *Who stayed where?*

- Standardizes follow-up numeric columns using the **baseline** mean and SD (not the follow-up's own). Self-standardization would mask genuine shifts in attitudes between waves.
- Computes composite distance to each baseline centroid: Euclidean on Z-scored numerics + Hamming on categoricals, weighted by feature count.
- Assigns each follow-up respondent to the nearest baseline cluster.
- Flags **Weak-Fit** respondents whose minimum distance exceeds `WEAK_FIT_DISTANCE = 0.35`. A weak-fit respondent may belong to an emergent segment not present at baseline.
- Builds a **transition matrix** showing cluster-level migration patterns.

**🆕 Emergence Agent (Longitudinal Mode)** — *Are new segments forming?*

- Runs a K+1 test on the weak-fit pool: fits GMMs at K = 1, 2, 3 and compares BIC + silhouette.
- If a coherent sub-structure emerges (Δ silhouette > 0.05 and BIC preferred), the weak-fit pool is flagged as a candidate new segment for the next wave.
- Distinguishes **genuine emergence** from **centroid drift** (the whole distribution shifts) and from **individual drift** (idiosyncratic movement).

### Measurement Invariance

Before trusting longitudinal comparisons, we check whether the survey itself is measuring the same construct across waves:

- **Kolmogorov–Smirnov test** (p > 0.05 = stable) on each numeric item's distribution
- **Chi-square test** (p > 0.05 = stable) on each categorical level's proportions

If invariance holds, observed migration reflects *genuine workforce change*, not a measurement artifact.

> **Note:** Phase 5 is optional. Run only when a follow-up survey is available. The baseline personas from Phase 4 remain valid and actionable regardless.

## Step 1 of 3 — Run Alignment + K+1 Test

```bash
python scripts/run_phase5_longitudinal.py
```

## Step 2 of 3 — Longitudinal Report

```
outputs/phase5_longitudinal/
├── report.md                              ← Gate 5 review
├── summary.json
├── followup_cleaned.csv
├── aligned_labels.csv                     per-respondent: cluster, weak-fit flag, min distance
├── transition_matrix.csv                  aggregate baseline -> follow-up proportions
├── k_plus_one_test.json                   BIC / silhouette / sizes for K=1,2,3 on weak-fit pool
├── audit_reports/
│   └── schema_drift_audit.md              categorical-level vocabulary drift between waves
└── reflection_logs/
    └── phase5_success_report.txt
```

## Step 3 of 3 — Gate 5: Your Decision (Continue or Conclude)

Unlike Gates 1–4, Phase 5 has no hard accept/reject; it produces a **findings + recommendations + next-steps** summary.

**Findings to review**

- Does the baseline cluster structure hold (≥ 90% strongly aligned, weak-fit rate ≤ 10%)?
- Does measurement invariance hold across the wave?
- Are migration patterns consistent with known workforce events (layoffs, reorgs, acquisitions)?
- Does the K+1 test suggest a new segment? If so, is its size and silhouette large enough to warrant follow-up?

**Two recommendations**

| Outcome | What it means |
|---|---|
| 🔄 **Continue longitudinal analysis** | When the next follow-up arrives, re-run Continuity + Emergence. Track the weak-fit pool. |
| 📌 **Conclude analysis** | Baseline personas remain valid. Use them for the current decision cycle. |

**Common schema gotcha:** the follow-up CSV may use different categorical level vocabulary from the baseline (e.g., baseline `Senior` vs. follow-up `Sr`; baseline `5-10 yrs` vs. follow-up `5+y`). The runner's `schema_drift_audit.md` surfaces this. If you see a 100% weak-fit rate, fix the level vocabulary before drawing conclusions — the Hamming distance on demographics will max out for every respondent.

---

## Running the Full Pipeline End-to-End

For a demo (not production), chain all five phases:

```bash
python scripts/run_phase1_data_steward.py \
  && python scripts/run_phase2_discover.py \
  && python scripts/run_phase3_ground.py \
  && python scripts/run_phase4_narrate.py \
  && python scripts/run_phase5_longitudinal.py
```

In production, don't. The gates are there for a reason.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Wrong Python interpreter or missing venv | Create a venv with `python -m venv venv` and `pip install -r requirements.txt` |
| Phase 2 crashes reading `survey_baseline_clean.csv` | Phase 1 did not finish | Re-run Phase 1, check for schema or bias-audit errors |
| Phase 3 runs in MOCK despite `ANTHROPIC_API_KEY` set | VS Code was open before the variable was set | Fully restart VS Code / Claude Code so child processes inherit the new value |
| Phase 5 reports 100% weak-fit | Categorical level vocabulary drift between waves | Harmonize follow-up levels to match baseline before running; see `schema_drift_audit.md` |
| Silhouette POOR (< 0.25) | Data does not cluster cleanly | This may be the finding itself — report it honestly; consider additional indicators |

## Where to Look Next

- [`README.md`](README.md) — tutorial context, learning objectives, citation
- [`SETUP.md`](SETUP.md) — environment setup for macOS / Windows
- [`resources/io_codebook.md`](resources/io_codebook.md) — the 12 validated I-O constructs with operational definitions
- [`resources/ethics_checklist.md`](resources/ethics_checklist.md) — responsible persona practice
- [`src/config.py`](src/config.py) — every threshold the pipeline applies, with literature citations
- [`agents/`](agents/) — per-agent SKILL.md specifications

## The Point

The agents do the computation. You make the judgments. Every gate is a moment where I-O expertise is least substitutable — a NEW theme from the LLM is a hypothesis, not a conclusion; an ARI of 0.34 is a finding, not a failure; a 100% weak-fit rate is a schema warning, not a workforce revolution.

Epistemic humility is the pipeline's job. Interpretive authority is yours.
