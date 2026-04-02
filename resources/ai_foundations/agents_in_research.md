# AI Agents in Research

> **Where this is used:** The entire pipeline is built as a multi-agent system with 9 specialized agents. This document explains the agent architecture pattern and why it matters for research transparency and reproducibility.

**What "agents" mean, how they differ from chatbots, and why this pipeline uses them.**

---

## Chatbot vs. Agent: What Is the Difference?

Most people's first experience with AI language models is through a chatbot -- you type a question, it types an answer, and the conversation continues. Chatbots are reactive: they wait for you to ask something, then respond.

An **agent** is different. An agent is an AI system that:

1. **Has a defined role** -- not a general-purpose assistant, but a specialist with a specific job
2. **Follows documented procedures** -- step-by-step instructions for how to do its job, including what to do when things go wrong
3. **Produces specific outputs** -- mandatory artifacts (files, reports, validated datasets) that downstream processes depend on
4. **Operates within boundaries** -- clear rules about what it can and cannot do, and when it must defer to a human
5. **Connects to other agents** -- its outputs feed into the next agent's inputs, forming a pipeline

### An Analogy

Think of a research team conducting a large-scale survey study:

| Research Team Member | Pipeline Agent Equivalent |
|---------------------|--------------------------|
| Research assistant who screens surveys for straight-lining and missing data | Data Steward |
| Quantitative analyst who runs cluster analysis | K-Prototypes Agent |
| Methodologist who validates the analysis | Psychometrician Agent |
| Technical writer who drafts the results section | Narrator Agent |
| Project manager who keeps everyone coordinated | Project Manager Agent |

Each person has a defined role, follows established procedures, produces specific deliverables, and routes their work to the right person next. The agents in this pipeline work the same way -- except they are AI systems operating under written specifications rather than human team members.

---

## Why Use Agents Instead of a Single AI?

You could, in theory, give a single AI model one enormous prompt that says "clean this data, cluster it, validate the clusters, and write persona narratives." There are several reasons this pipeline does not do that:

### 1. Specialization Improves Quality

Each agent is given only the context, instructions, and tools it needs for its specific task. A Data Steward that only thinks about data quality will do a better job than a general-purpose AI trying to handle everything at once.

### 2. Validation Gates Catch Errors

When one agent passes its output to the next, there is an explicit handoff point where results can be checked. If the Data Steward flags too many respondents as careless, that is visible before the clustering agent ever sees the data.

### 3. Reproducibility and Auditability

Each agent's specification is documented in a SKILL.md file. Anyone can read exactly what the agent was instructed to do, what data it received, and what it produced. This creates an audit trail that is critical for research credibility.

### 4. Human Oversight at Every Stage

The multi-agent architecture creates natural decision points where the I-O psychologist can review, approve, modify, or reject outputs before the pipeline continues.

### 5. Modularity

If you want to replace one clustering method with another, you only need to change one agent specification. The rest of the pipeline is unaffected.

---

## How Agent Specifications Work in This Pipeline

Each agent in this pipeline is defined by a `SKILL.md` file that contains:

| Section | What It Contains |
|---------|-----------------|
| **Role** | One-sentence description of what this agent does |
| **Upstream dependencies** | What data or outputs this agent expects to receive |
| **Step-by-step procedures** | Exactly what the agent does, in order, with decision rules |
| **Python code** | Implementation code for statistical and data processing steps |
| **Validation gates** | Quality checks that must pass before the agent's output is accepted |
| **Mandatory artifacts** | Files and reports the agent must produce |
| **Failure protocols** | What happens when something goes wrong (data quality issues, model failures, ambiguous results) |
| **Human gates** | Decision points where a human I-O psychologist must review and approve |
| **References** | Academic literature supporting the methods used |

These specifications serve a dual purpose:

1. **As system prompts for AI agents** -- you can give a SKILL.md file to an AI model and it will follow the instructions to perform the specified task
2. **As methodology documentation** -- a human analyst can read the same file and follow the procedures manually, step by step

---

## Agents in Social Science: Where This Is Heading

The use of AI agents in social science research is new, but the underlying idea is not. Computational social science has long used automated pipelines for data processing, text analysis, and statistical modeling. What is new is the ability of language models to handle tasks that previously required human judgment -- like reading open-ended survey responses and classifying them against theoretical constructs.

### Current Applications in Research

- **Qualitative coding at scale** -- Using AI to apply codebooks to thousands of open-ended responses, with human validation of a sample
- **Literature review assistance** -- Agents that search, filter, and summarize research papers based on specific criteria
- **Data quality screening** -- Automated detection of careless responding, straight-lining, and other response quality issues
- **Mixed-methods integration** -- Combining statistical clustering with AI-generated narrative synthesis

### Risks to Be Aware Of

- **The Oracle Effect** -- Stakeholders may treat AI-generated outputs as more authoritative than they deserve, simply because a computer produced them
- **Hallucination** -- AI can generate plausible but unsupported claims, which is particularly dangerous in organizational research where decisions affect people's careers
- **Bias amplification** -- AI models can reflect and amplify biases present in their training data
- **Loss of methodological judgment** -- Over-reliance on automated pipelines can erode the researcher's interpretive role

This pipeline addresses these risks through mandatory evidence tracing, human decision gates, statistical validation, and explicit epistemic risk warnings in every narrative output.
