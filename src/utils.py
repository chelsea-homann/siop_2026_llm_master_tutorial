"""
Shared utilities for the SIOP 2026 tutorial pipeline.

Provides the central LLM calling helper, display helpers for Jupyter
notebooks, and output-persistence functions. Every LLM-dependent
agent routes its API calls through ``call_llm``.
"""

import json
import os
from datetime import datetime, timezone

from src import config


# ── LLM helpers ───────────────────────────────────────────────────────────


def call_llm(prompt, system=None):
    """Call the Anthropic LLM and return the response text.

    Parameters
    ----------
    prompt : str
        The user-turn message content.
    system : str, optional
        An optional system prompt providing agent identity and
        constraints.

    Returns
    -------
    str
        The LLM response text.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required. "
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


def write_success_report(path, phase, agents, status, metrics, artifacts, notes=""):
    """Write a plain-text agent success report to disk.

    Parameters
    ----------
    path : str
        Destination file path (created with parent dirs).
    phase : str
        Pipeline phase label (e.g. "Phase 1 -- Ingest and Clean").
    agents : str or list[str]
        Agent name(s) responsible for this phase.
    status : str
        One-line status (e.g. "COMPLETE -- Gate 1: yes").
    metrics : dict
        Key-value pairs summarising quantitative outcomes.
    artifacts : list[str]
        Relative paths of every file written by this phase.
    notes : str, optional
        Methodology or caveats note appended at the bottom.

    Returns
    -------
    str
        The path that was written.
    """
    agent_str = agents if isinstance(agents, str) else ", ".join(agents)
    lines = [
        "=" * 65,
        "AGENT SUCCESS REPORT",
        "=" * 65,
        f"Phase:     {phase}",
        f"Agent(s):  {agent_str}",
        f"Status:    {status}",
        f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Model:     {config.MODEL}",
        "",
        "METRICS",
        "-" * 40,
    ]
    for k, v in metrics.items():
        lines.append(f"  {k:<35} {v}")
    lines += ["", "ARTIFACTS PRODUCED", "-" * 40]
    for i, a in enumerate(artifacts, 1):
        lines.append(f"  {i:2d}. {a}")
    if notes:
        lines += ["", "NOTES", "-" * 40, f"  {notes}"]
    lines += ["", "=" * 65, ""]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def save_json(data, path):
    """Write *data* as pretty-printed JSON (no metadata envelope).

    Parameters
    ----------
    data : any JSON-serialisable object
        Data to write.
    path : str
        Destination file path (parent dirs created automatically).

    Returns
    -------
    str
        The path that was written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path


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
