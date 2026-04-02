# Getting Access to AI Models

> **Where this is used:** Setup. If you need to configure your environment or obtain an API key, start here.

**Step-by-step instructions for setting up access to the models you can use with this pipeline.**

---

## Quick Reference: Which Model Should I Start With?

| Your Situation | Recommendation |
|---------------|---------------|
| I just want to follow along in the tutorial | No model setup needed -- the clustering and validation steps run without an AI model |
| I want to try the full pipeline with minimal setup | Sign up for an Anthropic or OpenAI API account (see below) |
| I need to keep employee data on my own machine | Install a local open-weight model (see "Local Models" section) |
| My organization already uses one of these providers | Use what you have -- the pipeline is model-agnostic |

---

## Option 1: Anthropic (Claude)

Claude is made by Anthropic and is available through their API.

### Getting an API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Create an account (email + password)
3. Navigate to **API Keys** in the left sidebar
4. Click **Create Key** and give it a name (e.g., "SIOP Tutorial")
5. Copy the key -- you will not be able to see it again

### Setting the API Key

**macOS / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

To make this permanent, add the export line to your shell profile (`~/.zshrc`, `~/.bashrc`, or equivalent).

### Cost

Anthropic charges per token (roughly per word). For reference, classifying 1,000 open-ended comments using Claude Sonnet costs approximately $1-5 depending on comment length and prompt complexity. Check [anthropic.com/pricing](https://www.anthropic.com/pricing) for current rates.

### Installing the SDK

```bash
pip install anthropic
```

---

## Option 2: OpenAI (GPT-4)

GPT-4 is made by OpenAI and is available through their API.

### Getting an API Key

1. Go to [platform.openai.com](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to **API Keys** in the left sidebar
4. Click **Create new secret key** and give it a name
5. Copy the key immediately

### Setting the API Key

**macOS / Linux:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

### Cost

OpenAI charges per token. Costs are comparable to Anthropic. Check [openai.com/pricing](https://openai.com/pricing) for current rates.

### Installing the SDK

```bash
pip install openai
```

---

## Option 3: Local Models (Advanced)

If you need to keep data on your own machine -- common when working with sensitive employee data -- you can run an open-weight model locally.

### What You Need

- **Hardware:** A computer with a modern GPU (NVIDIA with at least 8 GB VRAM for smaller models, 24+ GB for larger models). Some smaller models can run on a Mac with Apple Silicon (M1/M2/M3/M4).
- **Software:** [Ollama](https://ollama.com/) is the easiest way to run local models.

### Setting Up Ollama

1. Download Ollama from [ollama.com](https://ollama.com/)
2. Install and run it
3. Open a terminal and pull a model:

```bash
# Smaller, faster model (good for testing)
ollama pull llama3.2

# Larger, more capable model (requires more GPU memory)
ollama pull llama3.1:70b
```

4. Test it:

```bash
ollama run llama3.2 "Classify this comment as relevant or not relevant to Psychological Safety: 'I'm afraid to speak up in meetings because my manager dismisses new ideas.'"
```

### Trade-offs

| Advantage | Disadvantage |
|-----------|-------------|
| Data never leaves your machine | Requires GPU hardware |
| No per-query cost after setup | Setup is more technical |
| Full reproducibility (exact model version) | Open-weight models are generally less capable than the largest commercial models |
| No internet required after download | Slower than cloud-based APIs for most hardware |

---

## Verifying Your Setup

After setting your API key, run the verification script from the repository root:

```bash
python verify_setup.py
```

The script will check whether an API key is detected and display the result. The API key is optional -- the clustering and statistical validation steps in this pipeline do not require an AI model.

---

## A Note on Data Privacy

When you send data to a commercial API (Anthropic, OpenAI, Google), that data leaves your machine and is processed on their servers. For this tutorial, we use **synthetic data** -- no real employee information is involved.

When working with real employee data:

- Review your provider's data usage policy (most major providers do NOT train on API data by default)
- Check whether your organization has a data processing agreement (DPA) with the provider
- Consider using a local model if your data governance requirements prohibit cloud processing
- Never send personally identifiable information (PII) to any AI model unless you have explicit authorization

The pipeline is designed to work with de-identified, aggregate-level survey data. Individual-level comments should be stripped of names, email addresses, and other identifying information before processing.
