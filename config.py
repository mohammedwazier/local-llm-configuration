import os

# ── Vector Stores ─────────────────────────────────────────────────────────
# Tier 1: ChromaDB (Docker)
CHROMA_HOST        = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT        = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION", "tech_knowledge")

# ── Embeddings ────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBED_BATCH_SIZE   = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_DEVICE       = os.getenv("EMBED_DEVICE", "cpu")

# ── Re-ranker ─────────────────────────────────────────────────────────────
RERANKER_MODEL     = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
RERANKER_DEVICE    = os.getenv("RERANKER_DEVICE", "cpu")
RERANK_TOP_K       = int(os.getenv("RERANK_TOP_K", "5"))
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.1"))

# ── RAG Orchestrator (feedback loop) ──────────────────────────────────────
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "10"))
RAG_FINAL_TOP_K    = int(os.getenv("RAG_FINAL_TOP_K", "3"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.35"))
FEEDBACK_MAX_RETRIES = int(os.getenv("FEEDBACK_MAX_RETRIES", "3"))
FEEDBACK_MIN_CHUNKS = int(os.getenv("FEEDBACK_MIN_CHUNKS", "2"))

# ── Auto-update ───────────────────────────────────────────────────────────
AUTO_UPDATE_INTERVAL = int(os.getenv("AUTO_UPDATE_INTERVAL", "86400"))
AUTO_UPDATE_ENABLED  = os.getenv("AUTO_UPDATE_ENABLED", "false").lower() == "true"

# ── RAG Server ────────────────────────────────────────────────────────────
LLAMA_SERVER_URL   = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
RAG_SERVER_PORT    = int(os.getenv("RAG_SERVER_PORT", "8090"))
REQUEST_DELAY      = float(os.getenv("REQUEST_DELAY", "1.5"))
REQUEST_TIMEOUT    = int(os.getenv("REQUEST_TIMEOUT", "30"))

# ── Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Dataset (training) ────────────────────────────────────────────────────
HF_DATASET_ID      = os.getenv("HF_DATASET_ID", "Roman1111111/claude-sonnet-4.6-120000x")
MIN_GRADE          = float(os.getenv("MIN_GRADE", "8.0"))
MAX_EXAMPLES       = int(os.getenv("MAX_EXAMPLES", "0"))
RANDOM_SEED        = int(os.getenv("RANDOM_SEED", "42"))

# ── Model ─────────────────────────────────────────────────────────────────
BASE_MODEL_ID      = os.getenv("BASE_MODEL_ID", "unsloth/Qwen3.6-27B")
MAX_SEQ_LENGTH     = int(os.getenv("MAX_SEQ_LENGTH", "4096"))
LORA_RANK          = int(os.getenv("LORA_RANK", "16"))
LORA_ALPHA         = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT       = float(os.getenv("LORA_DROPOUT", "0.05"))
EPOCHS             = int(os.getenv("EPOCHS", "2"))
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM         = int(os.getenv("GRAD_ACCUM", "8"))
LEARNING_RATE      = float(os.getenv("LEARNING_RATE", "2e-4"))
LR_SCHEDULER       = os.getenv("LR_SCHEDULER", "cosine")
WARMUP_RATIO       = float(os.getenv("WARMUP_RATIO", "0.05"))
OPTIMIZER          = os.getenv("OPTIMIZER", "adamw_8bit")
PACKING            = os.getenv("PACKING", "true").lower() == "true"
BF16               = os.getenv("BF16", "true").lower() == "true"
LOAD_IN_4BIT       = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
GRAD_CHECKPOINT    = os.getenv("GRAD_CHECKPOINT", "true").lower() == "true"
LOG_STEPS          = int(os.getenv("LOG_STEPS", "10"))
SAVE_STEPS         = int(os.getenv("SAVE_STEPS", "200"))
SAVE_TOTAL_LIMIT   = int(os.getenv("SAVE_TOTAL_LIMIT", "2"))
OUTPUT_DIR         = os.getenv("OUTPUT_DIR", "./finetuned-qwen3.6-27b-claude")
GGUF_QUANT         = os.getenv("GGUF_QUANT", "q4_k_m")
EXPORT_GGUF        = os.getenv("EXPORT_GGUF", "true").lower() == "true"

# ── GGUF export ───────────────────────────────────────────────────────────
CONVERT_SCRIPT     = os.getenv("CONVERT_SCRIPT", "")
QUANTIZE_BIN       = os.getenv("QUANTIZE_BIN", "")
TRAINING_VENV      = os.getenv("TRAINING_VENV", "~/training-venv/bin/python")
