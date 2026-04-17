# Swiss Legal Agentic RAG

An end-to-end retrieval system for Swiss federal law, built as an entry to
Kaggle's [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)
competition. Given a German legal exam scenario, the system retrieves the
specific statute articles and court decisions needed to solve it.

**What makes this interesting**: the corpus is 2.9M chunks (176k law
articles + 2.5M court decision snippets), queries are full exam-length
scenarios (~1,400 chars of German legal prose), and the citation format
has subtle normalization mismatches between queries and corpus that cap
naive F1 at 71% — but a paragraph-aware matcher lifts the ceiling to 98%.

## Architecture

Four Kaggle notebooks, each producing a utility artifact that downstream
notebooks attach. This design makes the expensive stages reproducible once
and reusable everywhere.

┌─────────────────────────────────┐
│ 1. utility-swiss-legal-         │  Downloads wheels + bge-m3 model.
│    wheels-and-models            │  CPU, ~15 min. No internet in
│                                 │  downstream notebooks.
└──────────────┬──────────────────┘
│ (wheels, model, weaviate binary)
▼
┌─────────────────────────────────┐
│ 2. utility-swiss-legal-corpus   │  Chunks + encodes 2.9M texts
│                                 │  with bge-m3 fp16 on dual T4.
│                                 │  GPU, ~4.5 hrs. Outputs
│                                 │  corpus.parquet (~6.4 GB).
└──────────────┬──────────────────┘
│ (corpus.parquet)
▼
┌─────────────────────────────────┐
│ 3. utility-swiss-legal-lancedb  │  Builds LanceDB hybrid index:
│                                 │  IVF_HNSW_SQ vector + BM25 FTS.
│                                 │  CPU, ~1-3 hrs. Outputs
│                                 │  lancedb_data/ (~10 GB).
└──────────────┬──────────────────┘
│ (lancedb_data/)
▼
┌─────────────────────────────────┐
│ 4. swiss-legal-retrieval        │  Submission notebook. Hybrid
│    (submission)                 │  search, alpha tuning, para-
│                                 │  graph-aware matching. Writes
│                                 │  submission.csv.
└─────────────────────────────────┘


## Technical stack

- **Embeddings**: `BAAI/bge-m3` (1024-dim, multilingual, 8k context).
  Encoded in fp16 on dual T4 GPUs.
- **Vector store**: LanceDB with native fp16 storage. IVF_HNSW_SQ index
  (IVF partitioning + HNSW per partition + scalar quantization).
- **Full-text search**: LanceDB's built-in BM25 via Tantivy.
- **Hybrid search**: LanceDB fuses vector + BM25 results natively.
- **Retrieval strategy**: hybrid dense/sparse → dynamic score threshold
  (not fixed top-k) → paragraph-aware citation normalization.

## The road here

This isn't the architecture I planned. The notes below document real
dead-ends I hit so readers can skip them.

### Weaviate, twice

First attempt: embedded Weaviate, fp32 vectors, HNSW built during insert.
OOMed at ~15GB RAM with 12GB of fp32 vectors still in Python memory plus
Weaviate's in-process Go heap.

Second attempt, with streaming ingest and `GOMEMLIMIT`: finished the
ingest but hit the 19.5 GB Kaggle output limit. Weaviate requires fp32
input (5.96 GB → 11.9 GB just for vectors), plus HNSW graph overhead and
commit-log churn pushed total disk to 20.94 GB. At 90% disk, Weaviate
auto-set shards to READ-ONLY and the last 500k inserts silently failed.
The BM25 index was also corrupt because LSM flushes halted mid-build.

Root cause in one sentence: Weaviate's Python client serializes vectors
via protobuf `float32`. There's no fp16 path at all, even though the
storage engine supports quantization. For a RAM- and disk-constrained
environment like Kaggle, this is a hard no.

### Milvus, Qdrant, FAISS

Considered briefly as alternatives. Milvus Lite supports fp16 natively
but its BM25/full-text search only works in Milvus Standalone (needs
Docker, not Kaggle-friendly). Qdrant supports fp16 and has sparse vectors
for BM25 via external tooling but no built-in tokenizer. FAISS has no
hybrid search at all.

### LanceDB

Landed here. Native fp16 (with explicit PyArrow schema — inferred schemas
silently upcast to fp32). Built-in BM25 via Tantivy. Hybrid search in one
call. No server process, just files. Index build is actually slower on
fp16 than fp32 (a known LanceDB bug), but total time on Kaggle CPU is
still tractable.

### Data findings that shaped retrieval

From the corpus investigation notebook:

- **98.8% of train queries cite laws**, not court decisions. Initially
  this suggested filtering the corpus to laws only. But 27% of *val*
  queries cite BGE (court decisions) — inconsistency between splits.
  Keep everything in the corpus.
- **Train citations format-mismatch at 71.24%**: gold cites like
  `Art. 975 ZGB` are absent from the corpus, but `Art. 975 Abs. 1 ZGB`
  and `Art. 975 Abs. 2 ZGB` are present. Confirmed by the competition
  host on the forum: "LeXam doesn't have consistent normalization. But
  we always put the paragraphs like 'Art. 60 Abs. 2 OR' when they exist."
- **Ceiling analysis**: 93% of missing train citations are recoverable
  via paragraph-aware matching (strip `Abs. X` suffix, treat
  article-level gold as satisfied by any paragraph-level prediction).
  This lifts theoretical F1 ceiling from 71% to 98%.

### Retrieval strategy

- Hybrid search, alpha tuned on train.csv 80/20 split
- Dynamic score threshold (not fixed k) — train citation counts range
  from 1 to 44, median 2, p99 21
- Paragraph-aware citation matcher during threshold tuning to get true
  F1 signal

## Roadmap

- ✅ v1 — Hybrid search baseline (this repo)
- 🔜 v2 — **Agentic retrieval**: multi-turn query decomposition, citation
  verification loop, iterative refinement. Building this before reranking
  because it's the harder and more foundational piece.
- 🔜 v3 — Cross-encoder reranking over the agent's candidate set
- 🔜 v4 — Fine-tuning bge-m3 on Swiss legal query/citation pairs

## Why I'm sharing this

I'm not eligible for the competition prize pool (residency constraints),
so this is a portfolio piece. The notebooks are all public on Kaggle with
full reasoning, including the dead-ends above — I think showing the
thought process matters as much as the final F1 score.

## Repository structure
swiss-legal-agentic-rag/
├── README.md                          # this file
├── notebooks/
│   ├── 1-wheels-and-models.ipynb      # utility: download packages + bge-m3
│   ├── 2-corpus.ipynb                 # utility: chunk + encode 2.9M texts
│   ├── 3-lancedb.ipynb                # utility: build hybrid index
│   ├── 4-data-exploration.ipynb       # portfolio: data findings + ceiling analysis
│   └── 5-retrieval.ipynb              # submission: hybrid search + alpha tuning
└── docs/
├── architecture.md                # detailed design notes
└── ceiling-analysis.md            # why F1 caps at 98%, not 100%
## License

MIT. Do whatever you want with it.
