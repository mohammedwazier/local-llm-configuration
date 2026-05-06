# RAG 2.0 — Architecture & Feedback Loop

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       USER / CLIENT                                  │
│              (OpenCode, curl, any OpenAI client)                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ POST /v1/chat/completions
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG SERVER (:8090)                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   FEEDBACK LOOP                               │    │
│  │                                                               │    │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐  │    │
│  │   │ ANALYZE  │───▶│ RETRIEVE │───▶│ RERANK   │───▶│GEN   │  │    │
│  │   │          │    │          │    │          │    │ERATE │  │    │
│  │   └────┬─────┘    └────┬─────┘    └────┬─────┘    └───┬───┘  │    │
│  │        │               │               │              │      │    │
│  │        │  intent       │  ChromaDB     │  cross-      │ LLM  │    │
│  │        │  category     │               │  encoder     │ call │    │
│  │        │  keywords     │  filter       │  score       │      │    │
│  │        │               │               │  filter      │      │    │
│  │        └───────────────┴───────────────┴──────────────┘      │    │
│  │                                                               │    │
│  │                  ┌─────────────────┐                          │    │
│  │            ◀─────│   EVALUATE      │◀─────────────────────────┘    │
│  │            │     │                 │                               │
│  │            │     │  quality ≥ 0.35?│                               │
│  │            │     │  chunks ≥ 2?    │                               │
│  │            │     │  resp coverage? │                               │
│  │            │     └────────┬────────┘                               │
│  │            │              │                                        │
│  │            │         PASS │  FAIL                                  │
│  │            │              │  (up to 3 retries)                     │
│  │            │              ▼                                        │
│  │            │     ┌─────────────────┐                               │
│  │            └─────│   RE-ROUTE      │                               │
│  │                  │                 │                               │
│  │                  │  strategies:    │                               │
│  │                  │  1. standard    │                               │
│  │                  │  2. unfiltered  │                               │
│  │                  │  3. widen       │                               │
│  │                  │  4. expand      │                               │
│  │                  │  5. expand      │                               │
│  │                  └────────┬────────┘                               │
│  │                           │ RETRY                                  │
│  │                           └──────────▶ back to RETRIEVE            │
│  └─────────────────────────────────────────────────────────────────────┘
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐   │
│  │ Knowledge    │    │ Reranker     │    │ Auto-Updater          │   │
│  │ Layer        │    │ (cross-enc)  │    │ (background thread)   │   │
│  │              │    │              │    │                       │   │
│  │ ChromaDB ◀───┤    │ MiniLM-L6    │    │ every N seconds       │   │
│  │ Pinecone ◀───┤    │ or BGE       │    │ check Last-Modified   │   │
│  └──────────────┘    └──────────────┘    │ re-fetch if stale     │   │
│                                          └───────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ proxy to upstream
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLAMA SERVER (:8080)                               │
│                    (Qwen3, llama.cpp)                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Retry Strategies

| Strategy | Description | When Used |
|----------|-------------|-----------|
| `standard` | Normal retrieval with category filter | Initial attempt |
| `unfiltered` | Remove all category/domain filters | Retry 1 |
| `widen` | 2x top-k, no filters, include both tiers | Retry 1+ |
| `expand` | Query expansion with related terms | Retry 2+ |
| `expand` | Query expansion with related terms | Retry 2+ |

## Quality Evaluation

Score = weighted combination of:
1. **Avg reranker score** (40%) — cross-encoder relevance
2. **Chunk coverage** (30%) — how many chunks hit threshold
3. **Response overlap** (30%) — keyword coverage from query in answer

Pass if:
- `quality_score >= 0.35`
- `num_chunks >= 2`
- Response length >= 5 words

## Component Details

### Knowledge Layer (`knowledge_layer.py`)
- **ChromaDB**: Docker-based, local ANN search, metadata filtering

### Reranker (`reranker.py`)
- Cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Scores all retrieved chunks, filters by threshold, sorts
- Configurable via `config.py`: `RERANKER_MODEL`, `RERANK_SCORE_THRESHOLD`

### Auto-Updater (`auto_updater.py`)
- Background thread in `rag_server.py`
- Checks HTTP `Last-Modified`/`ETag` headers for stale URLs
- Re-fetches, re-chunks, re-embeds, and upserts

### Feedback Loop (`rag_orchestrator.py`)
- **ANALYZE**: Intent classification, category extraction, keyword spotting
- **RETRIEVE**: Multi-tier vector search with query expansion
- **RERANK**: Cross-encoder scoring
- **EVALUATE**: Quality scoring
- **RE-ROUTE**: Strategy selection for retry
- **RETRY**: Up to 3 attempts with escalating strategies

## Data Flow

```
Client Request
       │
       ▼
ANALYZE ──→ intent (question/task)
       │      category (software/rust, database/postgresql, …)
       │      keywords (rust, mqtt, kubernetes, …)
       ▼
RETRIEVE ──→ embed query with BGE-M3
       │      search ChromaDB
       ▼
RERANK ──→ cross-encoder scores each (query, chunk) pair
       │    filter by score threshold
       │    keep top-K
       ▼
GENERATE ──→ inject context into system prompt
       │     forward to llama-server
       ▼
EVALUATE ──→ compute quality score
       │
       ├── PASS → return response to client
       │
       └── FAIL → RE-ROUTE → select new strategy → RETRY → RETRIEVE
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_HOST` | localhost | ChromaDB Docker host |
| `CHROMA_PORT` | 8000 | ChromaDB REST port |

| `EMBEDDING_MODEL` | BAAI/bge-m3 | Embedding model |
| `RERANKER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker model |
| `RAG_TOP_K` | 10 | Initial retrieval count |
| `RAG_FINAL_TOP_K` | 3 | After re-ranking |
| `RAG_SCORE_THRESHOLD` | 0.35 | Quality pass threshold |
| `FEEDBACK_MAX_RETRIES` | 3 | Max feedback iterations |
| `AUTO_UPDATE_ENABLED` | false | Background auto-refresh |
| `AUTO_UPDATE_INTERVAL` | 86400 | Seconds between checks |

## Migration from Qdrant

If you have an existing Qdrant collection, run:

```bash
python migrate.py
```

This reads all points from Qdrant and re-inserts them into ChromaDB.
