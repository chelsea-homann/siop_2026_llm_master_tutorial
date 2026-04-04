# Key Concepts and Definitions

> **Where this is used:** Phase 3 (Ground in Organizational Reality) uses embeddings and retrieval-augmented generation to connect survey constructs to organizational documents. Understanding these concepts will help you interpret how the RAG agent finds relevant policy passages.

This document defines the core terms and theories that anchor the tutorial. If you are new to any of these, read this before the session. No prior experience with AI, coding, or computational methods is assumed.

## What Is a Large Language Model (LLM)?

A Large Language Model is an AI system trained on massive amounts of text that can read, interpret, and generate human language. Examples include ChatGPT (by OpenAI) and Claude (by Anthropic). In this tutorial, we use LLMs to do things that would otherwise require hundreds of hours of human effort: reading thousands of open-ended survey comments, classifying them against I-O psychology constructs, and drafting persona narratives grounded in evidence. The LLM does not replace the I-O psychologist. It acts as a research assistant that works under the psychologist's supervision and rules.

## What Is an "Agent"?

In everyday language, an agent is someone who acts on your behalf according to rules you set. In this pipeline, an "agent" is a specialized AI assistant with a single job, a defined set of rules, and clear boundaries on what it can and cannot do. Think of each agent as a team member with a job description:

- The **Data Steward** cleans and validates incoming data (like a research assistant screening surveys for quality).
- The **Narrator** writes persona descriptions from statistical evidence (like a technical writer who must cite every claim).
- The **Psychometrician** checks the statistical validity of results (like a methodologist reviewing your analysis).

Each agent's complete job description is documented in a `SKILL.md` file. These files specify exactly what the agent does, what rules it follows, what it produces, and what happens when something goes wrong. No agent acts autonomously. Every critical decision routes back to a human I-O psychologist for review.

## Personas

Research-based profiles representing groups of individuals, designed to foster empathy, align interventions, and support cross-functional communication (Pruitt & Adlin, 2010). In organizational contexts, personas translate fragmented employee data into human-centered narratives that leaders can act on. They bridge the gap between aggregate metrics ("engagement score: 3.4") and the lived experience of actual workforce segments.

## Grounded Theory

A systematic methodology where categories and themes emerge directly from the data rather than being imposed from preexisting frameworks (Glaser & Strauss, 2017). In this pipeline, grounded theory operates alongside deductive coding: the I-O Psychology Codebook provides validated constructs for deductive classification, while the pipeline also surfaces emergent themes that do not map to any existing construct. When a new theme survives human review, it is added to the codebook for the next survey wave.

## K-Prototypes Clustering

A grouping algorithm designed for **mixed-type data**, meaning datasets containing both categorical variables (e.g., department, tenure band) and continuous variables (e.g., survey scores). Most clustering methods require all variables to be the same type. K-Prototypes handles both by combining two distance measures: one for numeric data and one for categorical data (Huang, 1998). The result is **behavioral-demographic segments**, groups defined by both who people are and how they responded.

*Analogy:* Imagine sorting employees into groups where membership depends on both their job characteristics (department, level, tenure) and their survey responses (engagement, trust, morale) at the same time, rather than treating these as separate analyses.

## Latent Profile Analysis (LPA)

A model-based, person-centered approach that identifies unobserved subpopulations (latent profiles) from continuous survey response patterns using statistical mixture models. Unlike K-Prototypes, LPA ignores demographics entirely. It discovers **psychological profiles** (or "mindsets") based solely on how people responded to the survey (Spurk et al., 2020; Nylund et al., 2007).

Running K-Prototypes and LPA in parallel provides **methodological triangulation**, the statistical equivalent of having two independent raters code the same data. If both methods find similar groups, confidence increases. If they disagree, the disagreement itself is informative and worth investigating.

## Retrieval-Augmented Generation (RAG)

When AI generates text, it sometimes produces claims that sound plausible but are not actually true (a phenomenon called "hallucination"). RAG is a technique that prevents this by forcing the AI to look up real documents before writing anything (Lewis et al., 2020).

In this pipeline, organizational documents such as policies, leadership memos, and FAQ pages are indexed into a searchable knowledge base. When the pipeline generates persona narratives, it retrieves and cites actual company policies rather than making up plausible-sounding but unfounded claims. This grounds every persona narrative in organizational reality.

*Analogy:* Instead of asking someone to write a report from memory, you give them a file cabinet of source documents and require them to cite specific pages for every claim they make.

## Adjusted Rand Index (ARI)

A measure of similarity between two independent clustering solutions, adjusted for chance. Ranges from -1 (worse than chance) through 0 (chance-level agreement) to 1 (perfect agreement). In this pipeline, ARI measures how much K-Prototypes and LPA agree on which respondents belong together, corrected for chance using a hypergeometric null distribution (Hubert & Arabie, 1985; Steinley, 2004).

*In I-O terms:* If two different analytical methods independently sorted employees into groups using the same survey data, ARI measures how much those groupings overlap.

## Silhouette Coefficient

A cluster validation metric that measures how similar each observation is to its own cluster compared to the nearest neighboring cluster. Ranges from -1 (misclassified) to +1 (well-clustered). Values above 0.50 indicate reasonable structure; below 0.25 suggests weak or artificial groupings (Rousseeuw, 1987).

*In plain language:* Are the groups we found genuinely distinct from each other, or did the algorithm just force arbitrary divisions into the data?

## Evidence-Based Change Management

The organizational translation framework used throughout this pipeline draws on Stouten, Rousseau, and De Cremer (2018), who synthesized ten evidence-based steps for successful organizational change. Key principles include: diagnose before acting, assess organizational readiness, engage stakeholders in interpreting data, and ground every recommendation in specific evidence. The Project Manager Agent enforces these principles by requiring that all findings are translated into actionable organizational language before reaching stakeholders.

## Epistemic Risk Mitigation

When AI generates narrative text about data patterns, it introduces risks identified by Nguyen and Welch (2025): generating plausible but unfounded claims, producing unreliable outputs, anthropomorphizing data patterns (treating statistical groups as if they have feelings or intentions), and the "Oracle Effect" (stakeholders treating AI output as authoritative simply because a computer generated it). This pipeline mitigates these risks by requiring every narrative claim to trace to a specific statistical metric or verbatim respondent quote, and by preserving the I-O psychologist's final interpretive authority.
