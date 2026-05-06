#!/usr/bin/env bash
# setup.sh — One-shot bootstrap for the RAG 2.0 knowledge base system.
#
# What it does:
#   1. Start ChromaDB via Docker Compose
#   2. Create systemd user service for ChromaDB
#   3. Create a Python venv (.venv) and install dependencies
#   4. Create a systemd user service for the RAG server (:8090)
#   5. Print next-step instructions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
CHROMA_DATA="$HOME/.chromadb-data"
SYSTEMD_USER="$HOME/.config/systemd/user"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
cmd()   { echo -e "${CYAN}[cmd]${NC}   $*"; }

# ─── 1. ChromaDB via Docker Compose ─────────────────────────────────────────
info "Pulling ChromaDB Docker image…"
docker compose -f "$SCRIPT_DIR/docker-compose.yml" pull

mkdir -p "$CHROMA_DATA"
info "ChromaDB data directory: $CHROMA_DATA"

# Stop existing container if any
docker compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true

info "Starting ChromaDB container (detached)…"
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d

# Wait for ChromaDB to be ready
info "Waiting for ChromaDB to be ready…"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
        info "ChromaDB is ready."
        break
    fi
    sleep 1
done

# ─── 2. Systemd user service for ChromaDB ──────────────────────────────────
mkdir -p "$SYSTEMD_USER"

cat > "$SYSTEMD_USER/chromadb.service" << EOF
[Unit]
Description=ChromaDB Vector Database (Docker)
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/docker compose -f $SCRIPT_DIR/docker-compose.yml start chromadb
ExecStop=/usr/bin/docker compose -f $SCRIPT_DIR/docker-compose.yml stop chromadb
Restart=on-failure

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable chromadb.service
info "ChromaDB systemd service enabled."

# ─── 3. Python venv ─────────────────────────────────────────────────────────
info "Creating Python virtual environment at $VENV_DIR …"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip

info "Installing CPU-only PyTorch…"
pip install -q torch --index-url https://download.pytorch.org/whl/cpu

info "Installing remaining dependencies…"
pip install -q -r "$SCRIPT_DIR/requirements.txt"
info "Python dependencies installed."

# ─── 4. systemd service for the RAG server ──────────────────────────────────
cat > "$SYSTEMD_USER/rag-server.service" << EOF
[Unit]
Description=RAG Knowledge Base Server 2.0 (OpenAI-compatible proxy on :8090)
After=network.target chromadb.service llama-qwen.service

[Service]
Type=simple
Environment=HOME=$HOME
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin
Environment=CHROMA_HOST=localhost
Environment=CHROMA_PORT=8000
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV_DIR/bin/python $SCRIPT_DIR/rag_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable rag-server.service
info "RAG server systemd service created."

# ─── 5. Summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup complete!  RAG 2.0${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  ChromaDB running at   http://localhost:8000"
echo "  Vector store          ChromaDB (Tier 1)"

echo "  Next steps:"
echo ""
cmd "1. Ingest documentation:"
echo "       cd $SCRIPT_DIR"
echo "       source .venv/bin/activate"
echo "       python ingest.py"
echo ""
cmd "2. Or ingest a single domain:"
echo "       python ingest.py --domain devops"
echo ""
cmd "3. Test retrieval:"
echo "       python query.py 'how to set up PostgreSQL replication?' --show-chunks"
echo ""
cmd "4. Test with feedback loop:"
echo "       python query.py 'how to set up PostgreSQL replication?' --feedback"
echo ""
cmd "5. Start the RAG server:"
echo "       systemctl --user start rag-server"
echo "       # OpenCode: set baseURL to http://127.0.0.1:8090/v1"
echo ""
cmd "6. Enable auto-update (in config.py):"
echo "       AUTO_UPDATE_ENABLED = True"
echo ""

echo "  Architecture:"
echo "    User → ANALYZE → RETRIEVE (ChromaDB)"
echo "                    → RE-RANK (cross-encoder)"
echo "                    → GENERATE (LLM)"
echo "                    → EVALUATE → PASS (return)"
echo "                               → FAIL → RE-ROUTE → RETRY"
echo ""
