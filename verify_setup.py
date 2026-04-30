"""
SIOP 2026 Master Tutorial -- Setup Verification
=================================================
Run this script to confirm your environment is ready:

    python verify_setup.py

It checks required packages, source modules, data files,
organizational documents, the tutorial notebook, and
optionally verifies API connectivity.
"""

import sys
import os

PASS = "[OK]"
FAIL = "[!!]"
SKIP = "[--]"


def check_python_version():
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        print(f"  {PASS} Python {v.major}.{v.minor}.{v.micro}")
        return True
    else:
        print(f"  {FAIL} Python {v.major}.{v.minor}.{v.micro} -- need 3.10+")
        return False


def check_package(name, import_name=None):
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "installed")
        print(f"  {PASS} {name} ({version})")
        return True
    except ImportError:
        print(f"  {FAIL} {name} -- not installed (pip install {name})")
        return False


def check_data():
    try:
        import pandas as pd
    except ImportError:
        print(f"  {FAIL} Cannot check data -- pandas not installed")
        return False

    ok = True

    for filename in ["baseline_survey_data_synthetic.csv", "survey_followup.csv"]:
        path = os.path.join("synthetic_data", filename)
        if not os.path.exists(path):
            print(f"  {FAIL} {path} -- not found")
            ok = False
            continue

        df = pd.read_csv(path)
        print(f"  {PASS} {path} -- {df.shape[0]:,} rows x {df.shape[1]} columns")

        expected_cols = [
            "Business Unit", "Level", "FLSA", "Tenure",
            "Cared_About", "Excited", "Helpful_Info", "Trust_Leadership", "Morale",
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"       {FAIL} Missing expected columns: {missing}")
            ok = False

    return ok


def check_src_modules():
    """Verify that src/ package and all phase modules can be imported."""
    modules = [
        ("src", "src"),
        ("src.config", "src.config"),
        ("src.utils", "src.utils"),
        ("src.project_manager", "src.project_manager"),
        ("src.p1_ingest", "src.p1_ingest"),
        ("src.p2_discover", "src.p2_discover"),
        ("src.p3_ground", "src.p3_ground"),
        ("src.p4_narrate", "src.p4_narrate"),
        ("src.p5_longitudinal", "src.p5_longitudinal"),
    ]
    ok = True
    for display_name, import_name in modules:
        try:
            __import__(import_name)
            print(f"  {PASS} {display_name}")
        except ImportError as e:
            print(f"  {FAIL} {display_name} -- {e}")
            ok = False
    return ok


def check_org_documents():
    """Verify that synthetic_data/org_documents/ exists and contains files."""
    doc_dir = os.path.join("synthetic_data", "org_documents")
    if not os.path.isdir(doc_dir):
        print(f"  {FAIL} {doc_dir}/ directory not found")
        return False

    files = [
        f for f in os.listdir(doc_dir)
        if os.path.isfile(os.path.join(doc_dir, f)) and not f.startswith(".")
    ]
    if len(files) == 0:
        print(f"  {FAIL} {doc_dir}/ is empty (need organizational documents for RAG)")
        return False

    print(f"  {PASS} {doc_dir}/ -- {len(files)} documents found")
    for f in sorted(files):
        print(f"       {f}")
    return True


def check_notebook():
    """Verify that the tutorial notebook exists."""
    nb_path = os.path.join("notebooks", "tutorial.ipynb")
    if os.path.exists(nb_path):
        print(f"  {PASS} {nb_path}")
        return True
    else:
        print(f"  {FAIL} {nb_path} -- not found")
        return False


def check_output_dirs():
    """Verify that the output directories exist."""
    dirs = [
        "outputs/phase1_data_quality_report",
        "outputs/phase2_cluster_validation",
        "outputs/phase3_emergent_themes",
        "outputs/phase4_persona_narratives",
        "outputs/audit_trail",
    ]
    ok = True
    for d in dirs:
        if os.path.isdir(d):
            print(f"  {PASS} {d}/")
        else:
            print(f"  {FAIL} {d}/ -- not found")
            ok = False
    return ok


def check_resources():
    """Verify resources directory and key files exist."""
    files = [
        "resources/io_codebook.md",
        "resources/ethics_checklist.md",
        "resources/ai_foundations/key_concepts.md",
    ]
    ok = True
    for f in files:
        if os.path.exists(f):
            print(f"  {PASS} {f}")
        else:
            print(f"  {FAIL} {f} -- not found")
            ok = False
    return ok


def check_api():
    """Optional: check if an Anthropic API key is set."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if anthropic_key:
        masked = anthropic_key[:8] + "..." + anthropic_key[-4:]
        print(f"  {PASS} ANTHROPIC_API_KEY is set ({masked})")
        return True
    else:
        print(f"  {SKIP} No ANTHROPIC_API_KEY found (optional -- mock mode will be used)")
        return True


def main():
    print()
    print("SIOP 2026 Master Tutorial -- Setup Verification")
    print("=" * 48)

    all_ok = True

    print("\n1. Python Version")
    all_ok &= check_python_version()

    print("\n2. Core Packages")
    all_ok &= check_package("numpy")
    all_ok &= check_package("pandas")
    all_ok &= check_package("scipy")
    all_ok &= check_package("scikit-learn", "sklearn")
    all_ok &= check_package("matplotlib")
    all_ok &= check_package("seaborn")

    print("\n3. Clustering Packages")
    all_ok &= check_package("kmodes")

    print("\n4. LLM Packages (optional)")
    check_package("anthropic")

    print("\n5. Source Modules")
    all_ok &= check_src_modules()

    print("\n6. Synthetic Datasets")
    all_ok &= check_data()

    print("\n7. Organizational Documents")
    all_ok &= check_org_documents()

    print("\n8. Tutorial Notebook")
    all_ok &= check_notebook()

    print("\n9. Output Directories")
    all_ok &= check_output_dirs()

    print("\n10. Resources")
    all_ok &= check_resources()

    print("\n11. API Key (optional)")
    check_api()

    print()
    print("=" * 48)
    if all_ok:
        print("Ready! Open notebooks/tutorial.ipynb to begin.")
    else:
        print("Some checks failed. See above for details.")
        print("Refer to SETUP.md for troubleshooting.")
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
