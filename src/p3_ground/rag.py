"""
RAG Agent -- document retrieval using TF-IDF for construct grounding.

Builds a lightweight knowledge base from organisational documents
(Markdown and plain-text files) by chunking them and computing
TF-IDF vectors with scikit-learn.  Queries are answered via cosine
similarity against the chunk vectors.

This implementation intentionally uses TF-IDF rather than dense
embeddings (sentence-transformers) to keep the tutorial dependency
footprint small and avoid the large PyTorch download.  Production
systems would typically use dense embeddings for better semantic
matching.

References
----------
Lewis et al. (2020). Retrieval-augmented generation for
    knowledge-intensive NLP tasks. NeurIPS.
Gao et al. (2024). Retrieval-augmented generation for large
    language models: A survey. arXiv:2312.10997.
"""

import glob
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.utils import audit_entry, call_llm, load_mock


# ── Private helpers ──────────────────────────────────────────────────────


def _load_documents(doc_dir):
    """Load all .md and .txt files from *doc_dir* recursively.

    Returns a list of dicts with ``name``, ``path``, ``content``,
    and ``size_chars`` keys.
    """
    documents = []
    for ext in ("*.md", "*.txt"):
        for filepath in glob.glob(
            os.path.join(doc_dir, "**", ext), recursive=True,
        ):
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                documents.append(
                    {
                        "name": os.path.basename(filepath),
                        "path": filepath,
                        "content": content,
                        "size_chars": len(content),
                    }
                )
            except OSError:
                pass  # skip unreadable files
    return documents


def _chunk_text(text, chunk_size=500, overlap=100):
    """Split *text* into overlapping chunks of approximately *chunk_size* characters.

    Attempts to split on paragraph and sentence boundaries first,
    falling back to whitespace.  This replaces the LangChain
    ``RecursiveCharacterTextSplitter`` to avoid an external
    dependency.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to find a clean break point
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                brk = text.rfind(sep, start, end)
                if brk > start:
                    end = brk + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            break
    return chunks


# ── Public entry points ──────────────────────────────────────────────────


def build_knowledge_base(doc_dir=None):
    """Load organisational documents, chunk them, and build a TF-IDF index.

    Parameters
    ----------
    doc_dir : str, optional
        Path to the directory containing organisational documents.
        Defaults to ``config.ORG_DOCS_DIR``.

    Returns
    -------
    dict
        Keys: ``chunks`` (list of dicts with ``text``, ``source``,
        ``chunk_index``), ``vectors`` (sparse TF-IDF matrix),
        ``vectorizer`` (fitted ``TfidfVectorizer``),
        ``n_documents``, ``n_chunks``.
    """
    doc_dir = doc_dir or config.ORG_DOCS_DIR
    documents = _load_documents(doc_dir)

    chunks = []
    for doc in documents:
        doc_chunks = _chunk_text(doc["content"])
        for i, text in enumerate(doc_chunks):
            chunks.append(
                {
                    "text": text,
                    "source": doc["name"],
                    "chunk_index": i,
                }
            )

    if not chunks:
        # Return an empty but usable knowledge base
        vectorizer = TfidfVectorizer()
        return {
            "chunks": [],
            "vectors": None,
            "vectorizer": vectorizer,
            "n_documents": 0,
            "n_chunks": 0,
        }

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    vectors = vectorizer.fit_transform(texts)

    return {
        "chunks": chunks,
        "vectors": vectors,
        "vectorizer": vectorizer,
        "n_documents": len(documents),
        "n_chunks": len(chunks),
    }


def query_knowledge_base(kb, query, top_k=3):
    """Retrieve the *top_k* most relevant chunks for *query*.

    Uses cosine similarity between the query's TF-IDF vector and
    the pre-computed chunk vectors.

    Parameters
    ----------
    kb : dict
        Knowledge base returned by ``build_knowledge_base``.
    query : str
        Natural-language query string.
    top_k : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Each entry has ``text``, ``source``, and ``score`` keys.
    """
    if kb["vectors"] is None or len(kb["chunks"]) == 0:
        return []

    query_vec = kb["vectorizer"].transform([query])
    sims = cosine_similarity(query_vec, kb["vectors"]).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        score = float(sims[idx])
        if score > 0:
            results.append(
                {
                    "text": kb["chunks"][idx]["text"],
                    "source": kb["chunks"][idx]["source"],
                    "score": round(score, 4),
                }
            )

    return results


def ground_constructs(kb, codebook=None):
    """Map each codebook construct to its most relevant policy passages.

    In live mode the LLM synthesises relevance judgements; in mock
    mode, pre-generated mappings are loaded from
    ``rag_output.json``.

    Parameters
    ----------
    kb : dict
        Knowledge base from ``build_knowledge_base``.
    codebook : list[str], optional
        Construct names.  Defaults to ``config.NUMERIC_COLS``.

    Returns
    -------
    dict
        Keys: ``mappings`` (construct -> list of passage dicts),
        ``audit_entries``.
    """
    codebook = codebook or list(config.NUMERIC_COLS)
    audit = []

    if config.MOCK_MODE:
        mappings = load_mock("rag_output.json")
        audit.append(
            audit_entry(
                "Ground", "RAG", "Construct grounding (mock)",
                {"constructs": codebook},
            )
        )
        reasoning = (
            f"Built knowledge base from {kb['n_documents']} organizational "
            f"documents ({kb['n_chunks']} text chunks). "
            f"Grounded {len(codebook)} codebook constructs against the corpus "
            f"(mock mode: pre-generated relevance assessments)."
        )
        return {"mappings": mappings, "reasoning": reasoning, "audit_entries": audit}

    mappings = {}
    for construct in codebook:
        passages = query_knowledge_base(kb, construct, top_k=3)
        if passages:
            # Use LLM to assess relevance
            passage_texts = "\n---\n".join(
                [f"[{p['source']}] {p['text']}" for p in passages]
            )
            prompt = (
                f"Assess the relevance of these policy passages to the I-O "
                f"psychology construct '{construct}'. For each passage, rate "
                f"relevance as HIGH, MODERATE, or LOW and explain briefly.\n\n"
                f"{passage_texts}"
            )
            system = (
                "You are an I-O psychologist with knowledge of grounding survey constructs in "
                "organisational policy documents. Be concise."
            )
            llm_response = call_llm(prompt, system=system)

            mappings[construct] = {
                "passages": passages,
                "llm_assessment": llm_response,
            }
        else:
            mappings[construct] = {"passages": [], "llm_assessment": "No relevant passages found."}

    audit.append(
        audit_entry(
            "Ground", "RAG", "Construct grounding (live)",
            {"constructs": codebook, "n_with_passages": sum(1 for m in mappings.values() if m["passages"])},
        )
    )

    # ── Build reasoning narrative ──────────────────────────────────
    n_with = sum(1 for m in mappings.values() if isinstance(m, dict) and m.get("passages"))
    reasoning_parts = [
        f"Built knowledge base from {kb['n_documents']} organizational "
        f"documents ({kb['n_chunks']} text chunks).",
        f"Grounded {len(codebook)} codebook constructs against the "
        f"organizational document corpus using live LLM relevance assessment.",
        f"{n_with} of {len(codebook)} constructs found relevant policy "
        f"passages.",
    ]
    ungrounded = [c for c in codebook
                  if isinstance(mappings.get(c), dict)
                  and not mappings[c].get("passages")]
    if ungrounded:
        reasoning_parts.append(
            f"Constructs without grounding: {', '.join(ungrounded)}. "
            f"These may lack representation in the current document corpus."
        )
    reasoning = " ".join(reasoning_parts)

    return {"mappings": mappings, "reasoning": reasoning, "audit_entries": audit}
