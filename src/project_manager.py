"""
Project Manager Agent -- audit trail management and governance.

The Project Manager is the "air traffic controller" for the
pipeline.  It maintains a timestamped audit trail of every
significant decision, validates governance metadata produced by
other agents, and generates a final governance report.

References
----------
Stouten, Rousseau, & De Cremer (2018). Successful organizational
    change: ten evidence-based steps. J. Applied Behavioral Science.
Sahay & Goldthwaite (2024). Participatory practices during
    organizational change. J. Organizational Behavior.
"""

from datetime import datetime, timezone

from src import config


# ── Audit trail management ───────────────────────────────────────────────


def create_audit_trail():
    """Initialise an empty audit trail.

    The audit trail is a simple list of dicts, each containing a
    timestamp, phase, agent name, action description, and
    supplementary details.  This flat structure makes it easy to
    convert to a DataFrame or JSON at the end of the pipeline.

    Returns
    -------
    list
        An empty list ready to receive entries via ``add_entry``.
    """
    return []


def add_entry(trail, phase, agent, action, details=None):
    """Append a timestamped entry to the audit trail.

    Parameters
    ----------
    trail : list
        The audit trail list (modified in place).
    phase : str
        Pipeline phase (Ingest, Discover, Ground, Write).
    agent : str
        Name of the agent recording the entry.
    action : str
        Short description of what happened.
    details : dict, optional
        Supplementary key-value pairs.

    Returns
    -------
    dict
        The newly created entry (also appended to *trail*).
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "agent": agent,
        "action": action,
        "details": details or {},
    }
    trail.append(entry)
    return entry


# ── Governance checks ────────────────────────────────────────────────────


def check_governance(results_dict):
    """Validate that a results dict contains required governance metadata.

    Each agent is expected to return an ``audit_entries`` list.
    This function checks for its presence and verifies that key
    thresholds were recorded.

    Parameters
    ----------
    results_dict : dict
        Output from any agent's ``run_*`` function.

    Returns
    -------
    dict
        ``{"passed": bool, "issues": list[str]}``
    """
    issues = []

    if "audit_entries" not in results_dict:
        issues.append("Missing 'audit_entries' in results")

    if "quality_report" in results_dict:
        qr = results_dict["quality_report"]
        if qr.get("confidence", 1.0) < 0.90:
            issues.append(
                f"Data quality confidence ({qr['confidence']}) below 0.90 threshold"
            )

    return {"passed": len(issues) == 0, "issues": issues}


# ── Report generation ────────────────────────────────────────────────────


def generate_audit_report(trail):
    """Format the complete audit trail as a human-readable report.

    Parameters
    ----------
    trail : list[dict]
        The accumulated audit trail.

    Returns
    -------
    dict
        Keys: ``report_text`` (str), ``n_entries`` (int),
        ``phases`` (list of unique phase names),
        ``agents`` (list of unique agent names).
    """
    if not trail:
        return {
            "report_text": "No audit entries recorded.",
            "n_entries": 0,
            "phases": [],
            "agents": [],
        }

    phases = sorted(set(e["phase"] for e in trail))
    agents = sorted(set(e["agent"] for e in trail))

    lines = [
        "=" * 56,
        "  PIPELINE GOVERNANCE AUDIT REPORT",
        "=" * 56,
        f"  Generated: {datetime.now(timezone.utc).isoformat()}",
        f"  Total entries: {len(trail)}",
        f"  Phases covered: {', '.join(phases)}",
        f"  Agents reporting: {', '.join(agents)}",
        "-" * 56,
    ]

    for phase in phases:
        lines.append(f"\n  Phase: {phase}")
        lines.append("  " + "-" * 40)
        phase_entries = [e for e in trail if e["phase"] == phase]
        for entry in phase_entries:
            ts = entry["timestamp"]
            agent = entry["agent"]
            action = entry["action"]
            lines.append(f"    [{ts}] {agent}: {action}")
            if entry.get("details"):
                for k, v in entry["details"].items():
                    lines.append(f"      {k}: {v}")

    lines.append("\n" + "=" * 56)

    return {
        "report_text": "\n".join(lines),
        "n_entries": len(trail),
        "phases": phases,
        "agents": agents,
    }
