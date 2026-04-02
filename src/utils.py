"""
Shared utilities for the SIOP 2026 tutorial pipeline.

Provides the central LLM calling helper (with automatic mock
fallback), mock-output loading, display helpers for Jupyter
notebooks, and output-persistence functions. Every LLM-dependent
agent routes its API calls through ``call_llm`` so that a single
toggle (``config.MOCK_MODE``) switches the entire pipeline between
live and offline operation.
"""

import json
import os
from datetime import datetime, timezone

from src import config


# ── LLM helpers ───────────────────────────────────────────────────────────


def call_llm(prompt, system=None, mock_key=None):
    """Call the Anthropic LLM or return a mock response.

    In mock mode the function loads the pre-generated JSON file
    identified by *mock_key* (e.g. ``"narrator_output.json"``).
    In live mode it instantiates the Anthropic client and sends
    the request.  This indirection keeps every agent module free
    of direct SDK imports and makes the pipeline runnable without
    an API key.

    Parameters
    ----------
    prompt : str
        The user-turn message content.
    system : str, optional
        An optional system prompt providing agent identity and
        constraints.
    mock_key : str, optional
        Filename inside ``src/mock_outputs/`` to load when mock
        mode is active.

    Returns
    -------
    str
        The LLM response text (or the loaded mock content).
    """
    if config.MOCK_MODE and mock_key:
        return load_mock(mock_key)

    # Late import so the SDK is only required in live mode.
    try:
        import anthropic  # noqa: F811
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for live LLM calls. "
            "Install it with: pip install anthropic"
        )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.MODEL,
        max_tokens=config.MAX_TOKENS,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Mock-output loading ──────────────────────────────────────────────────


def load_mock(filename):
    """Load a pre-generated mock output from the mock_outputs directory.

    Parameters
    ----------
    filename : str
        Name of the JSON file (e.g. ``"narrator_output.json"``).

    Returns
    -------
    dict or list or str
        The parsed JSON content, or the raw string if the file is
        not valid JSON.
    """
    filepath = os.path.join(config.MOCK_OUTPUT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            fh.seek(0)
            return fh.read()


# ── Display helpers (for Jupyter notebooks) ──────────────────────────────


def display_table(df, title=None, max_rows=20):
    """Pretty-print a pandas DataFrame for notebook display.

    Parameters
    ----------
    df : pandas.DataFrame
        The table to display.
    title : str, optional
        An optional header printed above the table.
    max_rows : int
        Maximum rows to show before truncation.

    Returns
    -------
    dict
        ``{"title": ..., "shape": ..., "preview": ...}`` so the
        notebook can decide how to render the output.
    """
    preview = df.head(max_rows).to_string(index=True)
    result = {
        "title": title or "Table",
        "shape": f"{df.shape[0]} rows x {df.shape[1]} cols",
        "preview": preview,
    }
    return result


def display_report(text, title=None):
    """Return a formatted text report dict for notebook display.

    Parameters
    ----------
    text : str
        The report body.
    title : str, optional
        An optional heading.

    Returns
    -------
    dict
        ``{"title": ..., "content": ...}``
    """
    return {"title": title or "Report", "content": text}


# ── Persistence ──────────────────────────────────────────────────────────


def save_output(data, path):
    """Save a Python object as JSON with a metadata timestamp.

    Creates parent directories as needed.  Non-serializable values
    are converted to strings.

    Parameters
    ----------
    data : dict or list
        The object to persist.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    envelope = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "0.1.0",
        "data": data,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2, default=str)


# ── Audit-trail helpers ──────────────────────────────────────────────────


def audit_entry(phase, agent, action, details=None):
    """Create a single audit-trail entry dict.

    Parameters
    ----------
    phase : str
        Pipeline phase (e.g. "Ingest", "Discover", "Ground", "Write").
    agent : str
        Agent name (e.g. "Data Steward").
    action : str
        Short description of the action taken.
    details : dict, optional
        Supplementary key-value pairs.

    Returns
    -------
    dict
        Timestamped audit record.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "agent": agent,
        "action": action,
        "details": details or {},
    }


# ── Phase output helpers ────────────────────────────────────────────────


def build_report(title, sections):
    """Build a human-readable markdown report from structured sections.

    Parameters
    ----------
    title : str
        Report title (e.g. "Phase 1: Data Quality Report").
    sections : list[dict]
        Each dict has ``"heading"`` (str) and ``"body"`` (str).
        Body can be plain text or markdown.

    Returns
    -------
    str
        Formatted markdown string ready to write to a .md file.
    """
    lines = [
        f"# {title}",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Pipeline version: {config.PIPELINE_VERSION}*",
        "",
    ]
    for section in sections:
        lines.append(f"## {section['heading']}")
        lines.append("")
        lines.append(section["body"])
        lines.append("")
    return "\n".join(lines)


def save_phase_outputs(phase_dir, report_md=None, dataframes=None,
                       raw_json=None):
    """Save all output formats for a phase to its output directory.

    Parameters
    ----------
    phase_dir : str
        Path to the phase output directory
        (e.g. ``"outputs/phase1_data_quality_report"``).
    report_md : str, optional
        Markdown report content. Saved as ``report.md``.
    dataframes : dict[str, pandas.DataFrame], optional
        Named DataFrames to save as CSVs. Keys become filenames
        (e.g. ``{"screening_results": df}`` → ``screening_results.csv``).
    raw_json : dict or list, optional
        Raw results for reproducibility. Saved as ``results.json``.

    Returns
    -------
    list[str]
        Paths of all files written.
    """
    import pandas as pd

    os.makedirs(phase_dir, exist_ok=True)
    written = []

    if report_md is not None:
        path = os.path.join(phase_dir, "report.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report_md)
        written.append(path)

    if dataframes is not None:
        for name, df in dataframes.items():
            if isinstance(df, pd.DataFrame):
                path = os.path.join(phase_dir, f"{name}.csv")
                df.to_csv(path, index=False)
                written.append(path)

    if raw_json is not None:
        path = os.path.join(phase_dir, "results.json")
        save_output(raw_json, path)
        written.append(path)

    return written


def save_audit_csv(audit_trail, path):
    """Save the audit trail as a CSV for easy review in Excel.

    Parameters
    ----------
    audit_trail : list[dict]
        List of audit entry dicts (from ``audit_entry()``).
    path : str
        Destination CSV path.
    """
    import pandas as pd

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for entry in audit_trail:
        row = {
            "timestamp": entry.get("timestamp", ""),
            "phase": entry.get("phase", ""),
            "agent": entry.get("agent", ""),
            "action": entry.get("action", ""),
        }
        details = entry.get("details", {})
        if isinstance(details, dict):
            for k, v in details.items():
                row[f"detail_{k}"] = str(v)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
