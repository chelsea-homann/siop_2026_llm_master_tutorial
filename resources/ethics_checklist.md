# Responsible Persona Practice Checklist

**SIOP 2026 Master Tutorial | Wymer, Wolfe, & Choe**

---

## Purpose

This checklist provides actionable, pass/fail checkpoints for ensuring that the LLM-based persona-building pipeline operates ethically at every stage. It is designed for the I-O psychologist, data analyst, or research team member who is running the pipeline and needs to verify that ethical obligations have been met before, during, and after each gate.

### Scope

The checklist covers data privacy, algorithmic fairness, narrative integrity, and responsible use of AI-generated outputs. It is mapped to the pipeline's four-gate architecture so that relevant checkpoints are reviewed at each decision point.

### Who Should Use This

- The I-O psychologist responsible for final interpretive authority
- Any team member running the pipeline or reviewing its outputs
- Organizational stakeholders evaluating whether to act on persona findings
- IRB reviewers assessing the ethical posture of the research methodology

---

## Pre-Pipeline Considerations

These checkpoints must be satisfied before any data enters the pipeline.

- [ ] **Informed consent:** Participants were informed that their survey responses would be analyzed using computational methods, including AI-based classification. Consent language does not need to describe specific algorithms, but it must not misrepresent the analysis as purely human-conducted.
  - *Rationale:* APA Ethics Code Standard 8.02 requires informed consent for research participation. AI-assisted analysis is a material aspect of the methodology.

- [ ] **Data use authorization:** The research team has authorization to use the survey data for persona development. If the data was originally collected for a different purpose, secondary use authorization has been obtained.
  - *Rationale:* Repurposing employee survey data beyond its original scope may violate organizational data governance policies and erode employee trust.

- [ ] **De-identification:** All direct identifiers (names, email addresses, employee IDs) have been removed or replaced with pseudonyms before data enters the pipeline. Indirect identifiers (small business units, unique role titles) have been assessed for re-identification risk.
  - *Rationale:* Even when data is "anonymized," combinations of demographic variables can uniquely identify individuals, especially in small organizational units.

- [ ] **Minimum cell size policy:** A minimum cell size threshold (recommended: n >= 5 per demographic subgroup within each cluster) has been established to prevent inadvertent identification of individuals through their cluster membership.
  - *Rationale:* A cluster containing only 2 people from a specific department with a specific job level is effectively identified data.

- [ ] **Data retention plan:** A plan exists for how long pipeline outputs, intermediate artifacts, and reflection logs will be retained, and how they will be securely disposed of.
  - *Rationale:* Cluster assignments and persona narratives contain sensitive characterizations that should not persist indefinitely without purpose.

---

## Gate 1 Checkpoints: Ingest and Clean

These checkpoints apply after the Data Steward Agent has processed the raw data and before cleaned data moves to clustering.

- [ ] **Screening criteria documented:** All careless responding detection criteria (longstring analysis, response time thresholds, IRV thresholds, attention check scoring) are documented in the audit trail with their thresholds and justifications.
  - *Rationale:* Undocumented screening decisions cannot be reviewed, replicated, or challenged.

- [ ] **Demographic impact of screening:** The demographic composition of respondents removed by quality screening has been compared against the full sample. No protected demographic group is disproportionately removed (removal rate for any subgroup should not exceed 2x the overall rate).
  - *Rationale:* Careless responding detection methods may systematically disadvantage certain populations. For example, respondents completing surveys on mobile devices may appear as "speeders" at higher rates. Non-native language speakers may show unusual response patterns flagged by longstring detection.

- [ ] **Imputation transparency:** All imputed values are flagged in the cleaned dataset. The proportion of imputed values per column is documented. No column with >20% imputed values proceeds without explicit justification.
  - *Rationale:* Heavy imputation can mask systematic non-response patterns that carry meaningful information about the workforce.

- [ ] **Excluded columns justified:** Columns removed for low variance or high missingness are documented with rationale. Exclusion decisions have been reviewed to ensure that no construct-relevant items were dropped without awareness.
  - *Rationale:* A "low variance" column may reflect genuine consensus rather than a measurement problem. Excluding it is a substantive decision, not merely a technical one.

- [ ] **Open-ended response handling:** If open-ended survey responses are included, they have been screened for personally identifying information before being passed to AI agents. Any responses containing names, locations, or other identifiers have been redacted.
  - *Rationale:* Open-ended text is the highest-risk data type for inadvertent identification.

---

## Gate 2 Checkpoints: Discover Segments

These checkpoints apply after K-Prototypes and LPA have produced cluster assignments and before cluster outputs move to grounding and narrative generation.

- [ ] **Protected characteristic alignment:** Cluster membership has been cross-tabulated with available demographic variables (business unit, level, FLSA status, tenure). No cluster is defined primarily by a single protected characteristic unless this has been explicitly identified and justified.
  - *Rationale:* If a cluster is 95% one demographic group, the "persona" is effectively a demographic stereotype rather than an attitudinal segment. This does not mean demographics cannot correlate with attitudes, but the correlation should be examined and documented.

- [ ] **Cluster size adequacy:** All clusters contain sufficient respondents for stable estimates (recommended: n >= 30 per cluster). Clusters below this threshold are flagged and their inclusion justified.
  - *Rationale:* Very small clusters produce unstable profiles and may represent outliers rather than meaningful segments. Personas based on small clusters risk overgeneralizing from limited data.

- [ ] **Representativeness check:** The overall sample represented by the clustering solution has been compared against the known workforce population. Significant underrepresentation of any workforce segment has been documented.
  - *Rationale:* If certain employee groups are underrepresented in the survey (e.g., hourly workers, field employees, non-English speakers), the personas will not represent their experiences. The absence of a group from the analysis is itself a finding that should be reported.

- [ ] **Outlier treatment ethics:** Respondents flagged as outliers by the Psychometrician Agent have been reviewed before exclusion. Outliers are retained in the data with flags rather than silently removed, unless they meet careless responding criteria.
  - *Rationale:* Outliers may represent genuinely extreme experiences that are important for the organization to understand. Automatic removal of people who "do not fit the model" can silence the most affected voices.

- [ ] **Multiple solutions compared:** At least two clustering approaches (K-Prototypes and LPA) have been compared using ARI or equivalent metrics. Areas of disagreement between methods have been documented and interpreted.
  - *Rationale:* Over-reliance on a single clustering method can produce artifacts specific to that algorithm. Cross-method validation provides evidence that segments are substantively real, not methodological byproducts.

- [ ] **K selection rationale:** The number of clusters (K) has been selected using statistical criteria (Silhouette, BIC, gap statistic) combined with substantive interpretability, not simply the solution that maximizes a single metric.
  - *Rationale:* The "optimal" number of clusters is a judgment, not a fact. Statistical criteria inform the decision but cannot make it. The I-O psychologist must assess whether the segments are meaningful and actionable.

---

## Gate 3 Checkpoints: Ground in Reality

These checkpoints apply after the RAG Agent and Emergence Agent have provided organizational context and emergent themes, and before the Narrator Agent generates persona narratives.

- [ ] **Stereotyping risk assessment:** Cluster profiles have been reviewed for characterizations that could reinforce harmful stereotypes (e.g., associating particular demographic groups with negative attitudes). If demographic-attitude correlations exist, the narrative must contextualize them structurally (organizational conditions) rather than dispositionally (group traits).
  - *Rationale:* Personas that attribute attitudes to demographic identity rather than organizational experience can entrench bias in leadership decision-making.

- [ ] **Codebook completeness review:** The 12-construct codebook has been evaluated against the actual survey data and organizational context. If employee responses systematically reference themes not captured by any codebook construct, this gap has been documented.
  - *Rationale:* The Emergence Agent detects novel themes, but it requires the I-O psychologist to confirm whether gaps represent genuine new constructs or measurement artifacts.

- [ ] **Emergent theme validation:** All candidate emergent themes flagged by the Emergence Agent have been reviewed by the I-O psychologist. Each has been classified as genuinely new, a variant of an existing construct, or noise, with documented rationale.
  - *Rationale:* AI-detected "emergent themes" may reflect model biases, prompt engineering artifacts, or linguistic patterns rather than genuine organizational phenomena. Human validation is required before any candidate enters the codebook.

- [ ] **Document bias assessment:** The organizational documents used by the RAG Agent have been evaluated for representativeness and bias. Documents reflect the organization's official position, which may differ from employee experience. This limitation has been documented.
  - *Rationale:* Corporate communications are authored by specific stakeholders with specific goals. Grounding persona narratives in these documents adds useful context but does not add "ground truth." The RAG Agent retrieves what the organization says, not what employees experience.

- [ ] **Policy-experience gap documentation:** Any contradictions between organizational policy language (RAG output) and employee survey responses have been flagged and will be addressed in persona narratives.
  - *Rationale:* When policy language and employee experience diverge, this disconnect is itself a finding that should be surfaced to leadership, not papered over by selecting only supporting documents.

- [ ] **Cross-cluster fairness:** Persona profiles across clusters have been compared for balanced characterization. No cluster is described using predominantly negative language while others receive predominantly positive framing.
  - *Rationale:* Narrative framing shapes how leaders perceive and respond to workforce segments. If the "disengaged" cluster receives a deficit-focused description while the "engaged" cluster receives strength-based language, the resulting interventions will reflect that framing bias.

---

## Gate 4 Checkpoints: Write Personas

These checkpoints apply to the final persona narratives produced by the Narrator Agent, before delivery to organizational stakeholders.

- [ ] **Employee comfort test:** Each persona narrative has been evaluated using the question: "Would an employee who recognizes themselves in this persona feel fairly and respectfully represented?" If the answer is uncertain, the narrative requires revision.
  - *Rationale:* Personas are descriptions of real people. Characterizations that reduce employees to data points or caricatures undermine the human-centered purpose of the methodology.

- [ ] **Evidence traceability:** Every factual claim in each persona narrative traces to a specific statistical metric (cluster mean, percentage, centroid value) or verbatim quote. No narrative claim relies solely on AI inference without supporting data.
  - *Rationale:* This is the primary defense against hallucination. The Narrator Agent generates plausible text, but plausibility is not accuracy. Every claim must be verifiable.

- [ ] **Dual-use risk assessment:** The research team has considered how persona outputs could be misused: targeting specific workforce segments for adverse actions, justifying predetermined restructuring decisions, or replacing genuine employee engagement with persona-based assumptions.
  - *Rationale:* Persona outputs are tools. Like any tool, they can be used constructively (designing targeted support interventions) or destructively (identifying which segment to lay off first). The research team has a professional obligation to anticipate misuse.

- [ ] **Stakeholder transparency:** The methodology section accompanying persona deliverables clearly states: (a) that AI agents were used in the analysis, (b) what role the AI played versus what role human judgment played, (c) the limitations of the approach, and (d) that personas represent statistical tendencies, not predictions about specific individuals.
  - *Rationale:* Organizational leaders who receive persona outputs must understand what they are and what they are not. Presenting AI-generated personas without disclosing the methodology violates professional transparency norms.

- [ ] **Confidence reporting:** Each persona includes a confidence or quality indicator (e.g., cluster Silhouette score, sample size, ARI agreement between methods) so that stakeholders can distinguish between well-supported and tentative characterizations.
  - *Rationale:* Not all clusters are equally robust. A persona based on a tight, well-separated cluster with n=500 should be weighted differently than one based on a loose cluster with n=40. Reporting confidence prevents false precision.

- [ ] **Actionability review:** Each persona includes specific, constructive recommendations for organizational action. Descriptions that characterize a segment without suggesting how to support it do not fulfill the purpose of the methodology.
  - *Rationale:* Personas without actionable recommendations become organizational labels. "The Disengaged" is a category, not a strategy. The value of the methodology lies in connecting what the data reveals to what the organization can do.

- [ ] **Longitudinal caveats:** If baseline and follow-up data are compared, the narrative explicitly states that observed shifts may reflect sample composition changes (different people responding at Time 2) rather than individual-level attitude change, unless panel data with respondent linking is available.
  - *Rationale:* Attributing group-level shifts to individual change without panel data is an ecological fallacy. The narrative must distinguish between compositional and genuine attitudinal change.

---

## Post-Pipeline Responsibilities

These obligations extend beyond the technical pipeline into the professional responsibilities of the research team.

- [ ] **Decision influence audit:** If persona outputs will inform organizational decisions (restructuring, resource allocation, talent strategy), the research team has assessed whether the data quality and methodological rigor are sufficient to support those decisions.
  - *Rationale:* The threshold for "interesting research finding" is lower than the threshold for "basis for organizational action." The research team bears responsibility for communicating this distinction.

- [ ] **Hallucination verification:** A sample of AI-generated narrative claims has been manually verified against the source data. Any claims that could not be verified have been flagged or removed.
  - *Rationale:* Even with evidence traceability requirements, LLMs can generate plausible claims that are subtly inaccurate. Random auditing provides an additional layer of quality assurance.

- [ ] **Ongoing monitoring plan:** If the persona pipeline will be run on subsequent survey waves, a plan exists for monitoring construct stability, codebook relevance, and model drift over time.
  - *Rationale:* Constructs that are relevant during one disruption may not be relevant during the next. The codebook and pipeline should be treated as living instruments, not fixed artifacts.

- [ ] **Revision protocol:** A clear process exists for incorporating new constructs, retiring outdated ones, updating organizational documents, and adjusting pipeline parameters based on validation results.
  - *Rationale:* The pipeline is a methodology, not a product. It improves through iterative refinement informed by validation data and stakeholder feedback.

- [ ] **Professional standards alignment:** The research team has reviewed the analysis against relevant professional standards, including the SIOP Principles for the Validation and Use of Personnel Selection Procedures, APA Ethical Principles, and any applicable organizational data governance policies.
  - *Rationale:* AI-assisted workforce analysis operates at the intersection of research ethics, data privacy, and organizational consulting. Multiple professional standards may apply.

---

## Summary

This checklist contains 30 checkpoints organized across six stages of the pipeline. Each checkpoint is designed to be a concrete, actionable item that can be marked pass or fail by the reviewing I-O psychologist. The checklist is not exhaustive; novel ethical concerns may arise as the pipeline is applied to new organizational contexts. When in doubt, the guiding principle is: *Would the employees whose data is being analyzed feel that this process treats them fairly, respects their dignity, and serves their interests as well as the organization's?*
