import os, glob, json, re, numpy as np, pickle
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RUN_ID = "fresh_run_2026_04_26"
DOC_DIR = "synthetic_data/org_documents"
OUT_DIR = "outputs/phase3_emergent_themes"
os.makedirs(f"{OUT_DIR}/rag_vector_store", exist_ok=True)
os.makedirs(f"{OUT_DIR}/audit_reports", exist_ok=True)
os.makedirs(f"{OUT_DIR}/reflection_logs", exist_ok=True)


def split_text(text, chunk_size=500, overlap=100):
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(t, seps):
        if not seps or len(t) <= chunk_size:
            return [t] if t.strip() else []
        sep = seps[0]
        parts = t.split(sep) if sep else list(t)
        chunks, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        result = []
        for c in chunks:
            if len(c) > chunk_size:
                result.extend(_split(c, seps[1:]))
            else:
                result.append(c)
        return result

    raw = _split(text, separators)
    final = []
    for i, c in enumerate(raw):
        if i > 0 and final:
            prev_tail = final[-1][-overlap:] if len(final[-1]) >= overlap else final[-1]
            merged = (prev_tail + " " + c)[:chunk_size]
            final.append(merged)
        else:
            final.append(c)
    return [f for f in final if len(f.strip()) >= 30]


def get_section(chunk, full_text):
    anchor = chunk[:40] if len(chunk) >= 40 else chunk[:20]
    pos = full_text.find(anchor)
    lines = full_text[:pos].split("\n") if pos > 0 else []
    for line in reversed(lines):
        if re.match(r"^#+\s", line):
            return line.lstrip("#").strip()
    return "General"


# Load documents
docs = []
all_chunks = []
for fp in sorted(glob.glob(os.path.join(DOC_DIR, "*.md"))):
    with open(fp, "r", encoding="utf-8") as f:
        text = f.read()
    doc = {"name": os.path.basename(fp), "content": text, "size": len(text)}
    docs.append(doc)
    raw_chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, ct in enumerate(raw_chunks):
        all_chunks.append({
            "text": ct,
            "metadata": {
                "document_name": doc["name"],
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "section": get_section(ct, text),
                "chunk_id": f"{doc['name']}_chunk_{i}",
                "run_id": RUN_ID,
            },
        })
    print(f"  {doc['name']}: {doc['size']} chars -> {len(raw_chunks)} chunks")

print(f"\nTotal: {len(docs)} docs, {len(all_chunks)} chunks")
avg = sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
print(f"Avg chunk size: {avg:.0f} chars")

# Embed
print("\nEmbedding with all-MiniLM-L6-v2 ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
n_nan = int(np.isnan(embeddings).any(axis=1).sum())
print(f"Shape: {embeddings.shape}, NaN: {n_nan}")
assert n_nan == 0, "Embedding gate failed"
print("Embedding gate: PASSED")


def retrieve(query, top_k=3, threshold=0.22):
    qe = model.encode([query])
    sims = cosine_similarity(qe, embeddings)[0]
    idx_sorted = np.argsort(sims)[-top_k:][::-1]
    return [
        {"text": all_chunks[i]["text"], "score": float(sims[i]),
         "metadata": all_chunks[i]["metadata"]}
        for i in idx_sorted if sims[i] >= threshold
    ]


# Policy integrity
print("\n=== POLICY INTEGRITY ===")
topics = [
    "compensation benefits", "remote work flexibility",
    "performance review", "training development",
    "restructuring change", "team culture values",
    "leadership communication trust", "employee wellbeing",
]
gaps = []
for topic in topics:
    res = retrieve(topic, top_k=1, threshold=0.28)
    if not res:
        gaps.append(topic)
        print(f"  [GAP] {topic}")
    else:
        print(f"  [OK]  {topic}: {res[0]['score']:.3f} ({res[0]['metadata']['document_name']})")

# Per-cluster queries
with open("outputs/phase2_cluster_validation/lpa_fingerprints.json") as f:
    fps = json.load(f)

cluster_queries = {
    "0": ["disengagement low morale feeling unsupported",
          "employee recognition feeling valued caring",
          "information transparency communication"],
    "1": ["trust leadership skepticism distrust",
          "organizational change restructuring",
          "informed employees mid-career concerns"],
    "2": ["employee engagement high performance thriving",
          "career development senior employees growth",
          "positive culture values teamwork"],
}
cluster_names = {"0": "Disengaged", "1": "Informed-Skeptical", "2": "Thriving"}

retrieval_results = {}
print("\n=== PER-CLUSTER RETRIEVAL ===")
for cid, queries in cluster_queries.items():
    print(f"\nCluster {cid} ({cluster_names[cid]})")
    hits, seen = [], set()
    for q in queries:
        for r in retrieve(q, top_k=3, threshold=0.22):
            ckey = r["metadata"]["chunk_id"]
            if ckey not in seen:
                seen.add(ckey)
                hits.append(r)
                print(f"  {r['score']:.3f} | {r['metadata']['document_name']} [{r['metadata']['section']}]")
                print(f"         {r['text'][:110]}...")
    retrieval_results[cid] = hits[:6]

# Save artifacts
np.save(f"{OUT_DIR}/rag_vector_store/embeddings.npy", embeddings)
with open(f"{OUT_DIR}/rag_vector_store/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

manifest = [
    {"chunk_id": c["metadata"]["chunk_id"], "document": c["metadata"]["document_name"],
     "section": c["metadata"]["section"], "preview": c["text"][:100] + "..."}
    for c in all_chunks
]
with open(f"{OUT_DIR}/rag_chunk_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

with open(f"{OUT_DIR}/audit_reports/rag_retrieval_detail.json", "w") as f:
    json.dump(
        {cid: [{"score": r["score"], "doc": r["metadata"]["document_name"],
                "section": r["metadata"]["section"], "text": r["text"]}
               for r in hits]
         for cid, hits in retrieval_results.items()},
        f, indent=2,
    )

integrity = {
    "documents": len(docs), "chunks": len(all_chunks),
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": int(embeddings.shape[1]),
    "nan_embeddings": n_nan, "coverage_gaps": gaps,
    "outdated_documents": [], "contradiction_candidates": [],
}
with open(f"{OUT_DIR}/audit_reports/policy_integrity_audit.json", "w") as f:
    json.dump(integrity, f, indent=2)

grounding = {
    cid: {
        "cluster_name": cluster_names[cid],
        "fingerprint": fps.get(cid, ""),
        "policy_snippets": [
            {"score": r["score"], "document": r["metadata"]["document_name"],
             "section": r["metadata"]["section"], "text": r["text"]}
            for r in hits
        ],
    }
    for cid, hits in retrieval_results.items()
}
with open(f"{OUT_DIR}/construct_grounding.json", "w") as f:
    json.dump(grounding, f, indent=2)

reflection = {
    "agent": "RAG Agent", "run_id": RUN_ID,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "operating_mode": "pipeline",
    "corpus": {
        "documents": len(docs), "chunks": len(all_chunks),
        "avg_chunk_size": int(avg),
        "chunk_params": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
    },
    "embedding": {
        "model": "all-MiniLM-L6-v2",
        "dim": int(embeddings.shape[1]),
        "nan": n_nan,
    },
    "policy_integrity": integrity,
    "per_cluster_hits": {cid: len(hits) for cid, hits in retrieval_results.items()},
}
with open(f"{OUT_DIR}/reflection_logs/rag_reflection.json", "w") as f:
    json.dump(reflection, f, indent=2)

print("\nAll RAG artifacts saved.")
print(f"Coverage gaps: {gaps if gaps else 'None'}")
print("RAG Agent: COMPLETE")
