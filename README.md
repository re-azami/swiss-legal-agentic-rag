# 🇨🇭 Swiss Legal Agentic RAG

Retrieval system for Swiss federal laws and court decisions, built for the
[LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)
Kaggle competition. Given a legal query (in German or English), returns
matching Swiss legal citations scored on macro F1.

## 🏗️ Architecture

Four-notebook pipeline, each a Kaggle utility script that plugs into the next:

1. **Corpus utility** — chunks 2.9M law articles and court considerations,
   encodes with bge-m3 in fp16 on dual T4 GPU (~4.5 hours)
2. **Weaviate utility** — pre-populated vector database with HNSW + BM25
   hybrid index
3. **Data exploration** — full investigation of queries, citation
   distributions, and format quirks
4. **Retrieval + submission** — hybrid search, dynamic score threshold,
   paragraph-aware matcher

## 🛣️ Roadmap

- ✅ **v1** — Hybrid search (BM25 + dense) baseline
- 🔜 **v2** — Agentic retrieval (query rewriting, multi-hop)
- 🔜 **v3** — Cross-encoder reranking
- 🔜 **v4** — Fine-tuning

## 💡 Key technical choices

- **bge-m3** multilingual embeddings for cross-lingual EN↔DE retrieval
- **fp16 encoding** on T4 tensor cores — 4.6× faster than fp32, no quality loss
- **Weaviate embedded** for one-call hybrid search
- **Dynamic k via score threshold** — fixed k fails because citation counts
  vary from 1 to 44 per query
- **Paragraph-aware matching** — the LeXam dataset cites at article-level
  but the corpus stores at paragraph-level; smart matching lifts the F1
  ceiling from 71% to 98%

📧 re.azami@gmail.com
