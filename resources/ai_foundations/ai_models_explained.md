# AI Language Models Explained

> **Where this is used:** Phases 3 and 4 use LLM inference. Phase 3's Emergence Agent uses the LLM to classify candidate themes. Phase 4's Narrator Agent uses the LLM to generate persona narratives. Understanding how these models generate text helps you evaluate whether the output is evidence-grounded or hallucinated.

**For I-O psychologists and social scientists who are new to this space.**

---

## What Is a Language Model?

A language model is a computer program that has been trained to understand and generate human language. It does this by learning statistical patterns from enormous amounts of text -- books, articles, websites, code -- and using those patterns to predict what words should come next in a sequence.

When you type a question into ChatGPT or Claude, the model is not "looking up" an answer in a database. It is generating a response word by word, based on patterns it learned during training. This is why language models can write fluently about almost any topic, but can also produce confident-sounding statements that are factually wrong.

### Key Terms

| Term | What It Means |
|------|--------------|
| **LLM** | Large Language Model -- a language model with billions of parameters, capable of complex reasoning and generation |
| **Token** | The basic unit a model reads and generates. Roughly 3/4 of a word in English. "Organizational" is about 3 tokens. |
| **Context window** | How much text the model can "see" at once. A 200K-token context window can process roughly a 150,000-word document in a single conversation. |
| **Prompt** | The text you send to the model -- your question, instructions, or data. Prompt quality dramatically affects output quality. |
| **System prompt** | Special instructions given to the model that define its role, rules, and constraints. The SKILL.md files in this repo function as system prompts. |
| **Temperature** | A setting that controls randomness. Low temperature (0.0) = deterministic, consistent outputs. High temperature (1.0) = more creative, varied outputs. For research, lower is usually better. |
| **Hallucination** | When a model generates text that sounds plausible but is factually incorrect or unsupported. A major concern for research applications. |
| **Fine-tuning** | Training an existing model on your own data to specialize it for a specific task. Not required for this tutorial. |
| **API** | Application Programming Interface -- a way to send prompts to a model and receive responses programmatically (via code) rather than through a chat interface. |

---

## The Major Models

### Commercial Models (Cloud-Based, Accessed via API)

| Model Family | Provider | Strengths | Considerations |
|-------------|----------|-----------|----------------|
| **Claude** (Opus, Sonnet, Haiku) | Anthropic | Strong reasoning, long context windows (up to 1M tokens), careful about making unsupported claims, good at following complex instructions | Requires API key; pay per token |
| **GPT-4 / GPT-4o** | OpenAI | Widely adopted, strong general performance, good code generation, large ecosystem of tools | Requires API key; pay per token |
| **Gemini** (Pro, Ultra) | Google | Integrated with Google ecosystem, multimodal (can process images, video, audio), competitive performance | Requires API key; pay per token |

### Open-Weight Models (Can Run Locally)

| Model Family | Provider | Strengths | Considerations |
|-------------|----------|-----------|----------------|
| **Llama 3** | Meta | Free to use, strong performance for its size, active research community | Requires local GPU or cloud compute to run; not as capable as largest commercial models |
| **Mistral / Mixtral** | Mistral AI | Efficient, good performance-to-size ratio, European company (relevant for GDPR) | Same compute requirements as Llama |
| **Phi-3** | Microsoft | Very small models that punch above their weight, can run on laptops | Limited capability compared to full-size models |

### What "Open-Weight" Means

When a model is "open-weight," it means the model's learned parameters are publicly available and you can download and run it on your own hardware. This matters for research because:

- **Data never leaves your machine** -- important for sensitive employee data
- **No per-query cost** -- useful for large-scale analysis
- **Reproducibility** -- the exact same model version is available indefinitely

The trade-off: open-weight models are generally less capable than the largest commercial models, and running them requires technical setup and hardware (a decent GPU).

---

## How to Choose a Model for Research

### For This Tutorial

Any model works. The agent specifications (SKILL.md files) are model-agnostic -- they describe the task, not the tool. During the tutorial, we will demonstrate the workflow and you can use whichever model you have access to.

### For Your Own Research

| Priority | Recommended Approach |
|----------|---------------------|
| **Data sensitivity is the top concern** | Use an open-weight model running locally, or a commercial model with a data processing agreement (DPA) |
| **Quality of classification is the top concern** | Use the most capable commercial model available (currently Claude Opus or GPT-4) |
| **Cost is the top concern** | Use a smaller commercial model (Claude Haiku, GPT-4o-mini) or an open-weight model |
| **Reproducibility is the top concern** | Use an open-weight model at a specific version, or a commercial model with a fixed version endpoint |

### A Note on Validation

Regardless of which model you choose, the pipeline includes validation at every stage. The Psychometrician Agent checks statistical quality. The Narrator Agent requires evidence for every claim. The I-O psychologist reviews all outputs. The model is a tool in the pipeline, not the final authority.

---

## How Models Are Used in This Pipeline

In this tutorial, AI models are used for three specific tasks:

1. **Classification** -- Reading open-ended survey comments and determining which I-O psychology constructs they reflect (e.g., "This comment reflects low Psychological Safety")
2. **Retrieval** -- Finding relevant organizational documents to ground persona narratives in real policies and communications
3. **Generation** -- Writing persona narrative drafts that synthesize statistical findings with verbatim quotes and cited policies

The models are NOT used for:
- Making final decisions about workforce segments
- Interpreting what the data "means" for the organization
- Recommending interventions without human review
- Any task where the I-O psychologist's judgment should prevail
