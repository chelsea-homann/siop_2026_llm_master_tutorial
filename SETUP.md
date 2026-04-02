# Setup Guide

**SIOP 2026 Master Tutorial: Human-Centered Personas at Scale**

Complete these steps **before the tutorial session** if possible.

> **New to AI, Python, or coding in general?** Start with the [`docs/ai_foundations/`](docs/ai_foundations/) folder. It covers what AI models are, how agents work in research, and how to get API keys, all written for I-O psychologists with no prior technical experience.

---

## Prerequisites

1. **Python 3.9 or later** from [python.org/downloads](https://www.python.org/downloads/)
2. **Git** from [git-scm.com](https://git-scm.com/)
3. **A code editor** such as [Visual Studio Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/download/)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/wymerc/siop_2026_llm_master_tutorial.git
cd siop_2026_llm_master_tutorial
```

No Git installed? Download the ZIP from the GitHub page (green "Code" button > "Download ZIP") and extract it.

---

## Step 2: Create a Virtual Environment

A virtual environment keeps this project's packages separate from your system Python.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt.

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install kmodes gower
pip install sentence-transformers
```

If you plan to work with organizational documents (PDF, DOCX) for the RAG pipeline:

```bash
pip install pdfplumber python-docx langchain
```

---

## Step 4: Verify Your Installation

```bash
python verify_setup.py
```

This checks all required packages, loads the synthetic datasets, and confirms everything is working. You should see green checkmarks for each component.

---

## Step 5: Set Up an LLM (Optional)

The agent specifications in this repository are model-agnostic. The clustering and statistical validation steps run without any AI model. If you want to run the full pipeline with LLM-powered agents, you will need an API key.

For detailed instructions on choosing a provider, getting API keys, cost estimates, local model options, and data privacy considerations, see [`docs/ai_foundations/install_and_access.md`](docs/ai_foundations/install_and_access.md).

Quick version:

```bash
# Set your API key as an environment variable
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `command not found: python3` | Try `python` or `py` instead. |
| `pip install` fails with permissions errors | Make sure your virtual environment is active (look for `(venv)` in your prompt). Do not use `sudo pip install`. |
| `kmodes` installation fails | Run `pip install --no-build-isolation kmodes`, or use `conda install -c conda-forge kmodes` if you use Anaconda. |
| `sentence-transformers` is slow to install | It downloads PyTorch (~2 GB). This is expected. Skip it if you are only running the clustering pipeline. |
| `ModuleNotFoundError` when running code | Your virtual environment is not active. Re-run the activate command from Step 2. |

If you are still stuck:

1. Check the [Issues](https://github.com/wymerc/siop_2026_llm_master_tutorial/issues) page on GitHub
2. Open a new issue with your operating system and Python version
3. During the tutorial session, raise your hand and we will help
