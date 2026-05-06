import argparse
import json
import logging
import re
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBEDDING_DIM,
    EMBEDDING_MODEL, EMBED_DEVICE,
    LLAMA_SERVER_URL, RAG_SERVER_PORT, RAG_TOP_K, RAG_FINAL_TOP_K,
    RAG_SCORE_THRESHOLD, FEEDBACK_MAX_RETRIES, FEEDBACK_MIN_CHUNKS,
    RERANKER_MODEL, RERANKER_DEVICE, RERANK_TOP_K, RERANK_SCORE_THRESHOLD,
    AUTO_UPDATE_ENABLED, AUTO_UPDATE_INTERVAL,
)
from knowledge_layer import KnowledgeLayer
from reranker import Reranker
from rag_orchestrator import RAGOrchestrator
from auto_updater import AutoUpdater
from sources import SOURCES, DOMAINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_server")

_RAG_SYSTEM = (
    "You are an expert AI assistant for an IoT Platform Engineer and Team Lead. "
    "You have deep, practical expertise in: Rust, Go, IoT protocols (MQTT, CoAP, AMQP, OPC-UA, LwM2M, Modbus), "
    "network engineering, Linux infrastructure, Kubernetes & K3s for edge/IoT, "
    "GitLab CI/CD, DevSecOps, security (TLS/mTLS, JWT, OWASP), "
    "and engineering team leadership.\n\n"
    "You reason through problems carefully, give precise and actionable answers, "
    "and always consider production-grade reliability, security, and scalability.\n\n"
    "Use the following retrieved documentation to answer accurately. "
    "Supplement with your own knowledge where the documentation falls short.\n\n"
)

_embed_model: Optional[SentenceTransformer] = None
_knowledge_layer: Optional[KnowledgeLayer] = None
_reranker: Optional[Reranker] = None
_orchestrator: Optional[RAGOrchestrator] = None
_args: Optional[argparse.Namespace] = None
_http_client: Optional[httpx.AsyncClient] = None
_auto_updater: Optional[AutoUpdater] = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info("Loading embedding model %s (%s) …", EMBEDDING_MODEL, EMBED_DEVICE)
        _embed_model = SentenceTransformer(EMBEDDING_MODEL, device=EMBED_DEVICE)
        log.info("Embedding model ready.")
    return _embed_model


def get_knowledge_layer() -> KnowledgeLayer:
    global _knowledge_layer
    if _knowledge_layer is None:
        _knowledge_layer = KnowledgeLayer(
            chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT,
            collection_name=CHROMA_COLLECTION, embedding_dim=EMBEDDING_DIM,
        )
    return _knowledge_layer


def get_reranker() -> Optional[Reranker]:
    global _reranker
    if _reranker is None:
        try:
            _reranker = Reranker(
                model_name=RERANKER_MODEL,
                device=RERANKER_DEVICE,
                top_k=RERANK_TOP_K,
                score_threshold=RERANK_SCORE_THRESHOLD,
            )
        except Exception as exc:
            log.warning("Reranker failed to load: %s — running without re-ranking", exc)
    return _reranker


def get_orchestrator() -> RAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator(
            knowledge_layer=get_knowledge_layer(),
            embed_model=get_embed_model(),
            reranker=get_reranker(),
            llama_base_url=LLAMA_SERVER_URL,
            top_k=RAG_TOP_K,
            final_top_k=RAG_FINAL_TOP_K,
            score_threshold=RAG_SCORE_THRESHOLD,
            max_retries=FEEDBACK_MAX_RETRIES,
            min_chunks=FEEDBACK_MIN_CHUNKS,
        )
    return _orchestrator


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _http_client


@lru_cache(maxsize=256)
def _cached_embed(query: str) -> tuple:
    vec = get_embed_model().encode(query, normalize_embeddings=True)
    return tuple(vec.tolist())


def retrieve_chunks(
    query: str, top_k: int, category_filter: Optional[str] = None,
) -> List[dict]:
    vec = list(_cached_embed(query))
    where = None
    if category_filter:
        where = {"domain": category_filter} if "/" not in category_filter else {"category": category_filter}
    return get_knowledge_layer().search(query_vector=vec, top_k=top_k, where=where)


def build_context_block(chunks: List[dict]) -> str:
    lines = ["### RETRIEVED DOCUMENTATION\n"]
    for i, chunk in enumerate(chunks, 1):
        md = chunk.get("metadata", {})
        cat = md.get("category", "?")
        title = md.get("title", "?")
        url = md.get("url", "")
        text = (chunk.get("text") or md.get("text", "")).strip()
        if len(text) > 500:
            text = text[:500] + "…"
        lines.append(f"[{i}] {cat} — {title}")
        if url:
            lines.append(f"Source: {url}")
        lines.append(text)
        lines.append("")
    lines.append("### END DOCUMENTATION\n")
    return "\n".join(lines)


def augment_messages(messages: list, top_k: int, category_filter: Optional[str], use_feedback: bool) -> list:
    user_turns = [m for m in messages if m.get("role") == "user"][-2:]
    query_parts = []
    for m in user_turns:
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    query_parts.append(part.get("text", ""))
        elif isinstance(content, str):
            query_parts.append(content)
    query = " ".join(query_parts).strip()
    if not query:
        return messages

    try:
        if use_feedback:
            orchestrator = get_orchestrator()
            result = orchestrator.execute(messages)
            chunks = result.chunks
            log.info("FEEDBACK LOOP strategy=%s retries=%d quality=%.3f chunks=%d",
                      result.strategy, result.retries, result.quality_score, len(chunks))
        else:
            chunks = retrieve_chunks(query, top_k, category_filter)
            reranker = get_reranker()
            if chunks:
                chunks = reranker.rerank(query, chunks, top_k=RAG_FINAL_TOP_K)
    except Exception as exc:
        log.warning("RAG retrieval failed: %s", exc)
        return messages

    if not chunks:
        return messages

    context_block = build_context_block(chunks)
    rag_prefix = _RAG_SYSTEM + context_block

    new_messages: list = []
    injected = False
    for msg in messages:
        if msg.get("role") == "system" and not injected:
            existing = msg.get("content", "")
            new_messages.append({
                "role": "system",
                "content": rag_prefix + (("\n\n" + existing) if existing else ""),
            })
            injected = True
        else:
            new_messages.append(msg)
    if not injected:
        new_messages.insert(0, {"role": "system", "content": rag_prefix})

    log.info("RAG: %d chunk(s) injected | query: %.60s…", len(chunks), query)
    return new_messages


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_tool_calls(text: str) -> tuple[list, str]:
    tool_calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": obj.get("name", ""),
                    "arguments": json.dumps(obj.get("arguments", {})),
                },
            })
        except json.JSONDecodeError:
            pass
    return tool_calls, _TOOL_CALL_RE.sub("", text).strip()


def _translate_response(data: dict) -> dict:
    try:
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            if "<tool_call>" not in content:
                continue
            tool_calls, remaining = _extract_tool_calls(content)
            if not tool_calls:
                continue
            msg["tool_calls"] = tool_calls
            msg["content"] = remaining or None
            choice["finish_reason"] = "tool_calls"
            log.info("Translated %d tool_call(s)", len(tool_calls))
    except Exception as exc:
        log.warning("Tool call translation failed: %s", exc)
    return data


def _translate_stream_chunk(chunk_bytes: bytes) -> bytes:
    try:
        text = chunk_bytes.decode("utf-8")
    except Exception:
        return chunk_bytes
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        if not line.startswith("data: "):
            out.append(line)
            continue
        payload = line[6:].strip()
        if payload in ("[DONE]", ""):
            out.append(line)
            continue
        try:
            data = json.loads(payload)
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                if "<tool_call>" not in content:
                    continue
                tool_calls, remaining = _extract_tool_calls(content)
                if not tool_calls:
                    continue
                delta["tool_calls"] = [
                    {"index": i, "id": tc["id"], "type": "function",
                     "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                    for i, tc in enumerate(tool_calls)
                ]
                delta["content"] = remaining or None
                choice["finish_reason"] = "tool_calls"
            out.append("data: " + json.dumps(data) + "\n")
        except Exception:
            out.append(line)
    return "".join(out).encode("utf-8")


def _upstream_url(path: str) -> str:
    return LLAMA_SERVER_URL.rstrip("/") + path


def _prepare_body(body: dict) -> dict:
    ktw = body.get("chat_template_kwargs")
    if isinstance(ktw, dict):
        ktw["enable_thinking"] = False
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if not body.get("max_tokens") or body["max_tokens"] < 256:
        body["max_tokens"] = -1
    return body


async def _stream_upstream(url: str, body: dict, headers: dict) -> AsyncGenerator[bytes, None]:
    client = get_http_client()
    async with client.stream("POST", url, json=body, headers=headers) as resp:
        async for chunk in resp.aiter_bytes():
            yield _translate_stream_chunk(chunk)


def _clean_headers(raw: dict) -> dict:
    h = dict(raw)
    h.pop("host", None)
    h.pop("content-length", None)
    return h


app = FastAPI(title="RAG Proxy 2.0 — ChromaDB + Feedback Loop")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    headers = _clean_headers(dict(request.headers))

    if not (_args and _args.no_augment):
        top_k = _args.top_k if _args else RAG_TOP_K
        cat = getattr(_args, "category", None)
        use_fb = not getattr(_args, "no_feedback", False)
        body["messages"] = augment_messages(body.get("messages", []), top_k, cat, use_fb)

    body = _prepare_body(body)
    url = _upstream_url("/v1/chat/completions")
    streaming = body.get("stream", False)

    if streaming:
        return StreamingResponse(
            _stream_upstream(url, body, headers),
            media_type="text/event-stream",
        )

    client = get_http_client()
    resp = await client.post(url, json=body, headers=headers)
    return JSONResponse(content=_translate_response(resp.json()), status_code=resp.status_code)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    headers = _clean_headers(dict(request.headers))
    body = _prepare_body(body)
    client = get_http_client()
    resp = await client.post(_upstream_url("/v1/completions"), json=body, headers=headers)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/v1/models")
async def list_models(request: Request):
    client = get_http_client()
    resp = await client.get(_upstream_url("/v1/models"))
    return JSONResponse(content=resp.json())


@app.get("/health")
async def health():
    try:
        count = get_knowledge_layer().count()
        return {
            "status": "ok",
            "vectors": count,
            "store": "ChromaDB",
            "reranker": RERANKER_MODEL,
            "feedback_loop": "enabled" if not (_args and _args.no_feedback) else "disabled",
        }
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}


@app.get("/debug/collections")
async def debug_collections():
    kl = get_knowledge_layer()
    return {
        "chroma_collection": CHROMA_COLLECTION,
        "chroma_count": kl.count(),
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_all(request: Request, path: str):
    headers = _clean_headers(dict(request.headers))
    body = None
    if request.method in ("POST", "PUT"):
        body = await request.body()
    client = get_http_client()
    resp = await client.request(
        method=request.method,
        url=_upstream_url(f"/{path}"),
        headers=headers,
        content=body,
        params=dict(request.query_params),
    )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


def main() -> None:
    global _args, _auto_updater
    parser = argparse.ArgumentParser(description="RAG proxy 2.0")
    parser.add_argument("--port", type=int, default=RAG_SERVER_PORT)
    parser.add_argument("--top-k", type=int, default=RAG_TOP_K)
    parser.add_argument("--final-top-k", type=int, default=RAG_FINAL_TOP_K)
    parser.add_argument("--category", type=str, default=None, help="Restrict retrieval to domain or category")
    parser.add_argument("--no-augment", action="store_true", help="Disable RAG (pure proxy)")
    parser.add_argument("--no-feedback", action="store_true", help="Disable feedback loop (simple RAG)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder re-ranking")
    _args = parser.parse_args()

    get_embed_model()
    get_knowledge_layer()
    get_orchestrator()

    if AUTO_UPDATE_ENABLED:
        from ingest import ingest as ingest_fn
        _auto_updater = AutoUpdater(
            knowledge_layer=get_knowledge_layer(),
            ingest_fn=ingest_fn,
            interval=AUTO_UPDATE_INTERVAL,
            enabled=True,
        )
        _auto_updater.start()

    log.info("RAG server 2.0 on :%d  →  %s", _args.port, LLAMA_SERVER_URL)
    log.info("  Store     : ChromaDB")
    log.info("  Reranker  : %s", RERANKER_MODEL)
    log.info("  Feedback  : %s", "enabled" if not _args.no_feedback else "disabled")
    log.info("  Top-k     : %d → rerank → %d", _args.top_k, _args.final_top_k if not _args.no_rerank else _args.top_k)
    uvicorn.run(app, host="0.0.0.0", port=_args.port, log_level="warning")


if __name__ == "__main__":
    main()
