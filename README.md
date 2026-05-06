# Local LLM Configuration

LLM server + RAG knowledge base running on a single RTX 3060 12GB.

## Stack

| Component | Tech |
|---|---|
| **LLM** | Qwen3.6-35B-A3B (MoE, Q4_K_M) via llama-cpp-turboquant |
| **KV Cache** | turbo4 (K) / turbo3 (V) |
| **Vector Store** | ChromaDB (Docker) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L6-v2 |
| **RAG Proxy** | FastAPI on :8090 with feedback loop |
| **Spec Decode** | ngram-mod (model-free) |
| **Context** | 262k tokens |

## Services

| Port | Service | Description |
|---|---|---|
| 8080 | `llama-qwen.service` | llama-server (OpenAI-compatible API) |
| 8090 | `rag-server.service` | RAG proxy with ChromaDB + reranker + feedback |
| 8000 | `chromadb.service` | ChromaDB (Docker) |

## Quick Start

```bash
# 1. Start ChromaDB
cd docker && docker compose up -d

# 2. Start LLM server
systemctl --user start llama-qwen.service

# 3. Start RAG proxy
systemctl --user start rag-server.service

# 4. Ingest documentation
python ingest.py

# 5. Query with RAG
python query.py "how does MQTT work?" --feedback
```

## Files

- `llama-qwen.service` — systemd unit for llama-server with turboquant
- `rag-server.service` — systemd unit for RAG proxy
- `docker-compose.yml` — ChromaDB Docker setup
- `config.py` — shared configuration
- `ingest.py` — documentation ingestion pipeline
- `rag_server.py` — RAG proxy with feedback loop
- `query.py` — CLI query tool with re-ranking
- `knowledge_layer.py` — ChromaDB vector store abstraction
- `reranker.py` — cross-encoder re-ranking
- `rag_orchestrator.py` — feedback loop (analyze → retrieve → rerank → evaluate → reroute → retry)
- `auto_updater.py` — background source refresh
- `setup.sh` — one-shot bootstrap
- `WORKFLOW.md` — architecture + flow diagram
