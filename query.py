import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBEDDING_DIM,
    EMBEDDING_MODEL, EMBED_DEVICE,
    LLAMA_SERVER_URL, RAG_TOP_K, RAG_FINAL_TOP_K,
    RERANKER_MODEL, RERANKER_DEVICE, RERANK_TOP_K, RERANK_SCORE_THRESHOLD,
    FEEDBACK_MAX_RETRIES, FEEDBACK_MIN_CHUNKS, RAG_SCORE_THRESHOLD,
)
from knowledge_layer import KnowledgeLayer
from reranker import Reranker
from rag_orchestrator import RAGOrchestrator


def retrieve(query: str, top_k: int, final_top_k: int, category: Optional[str],
             model: SentenceTransformer, kl: KnowledgeLayer, reranker: Reranker) -> list:
    vec = model.encode(query, normalize_embeddings=True).tolist()
    where = None
    if category:
        where = {"domain": category} if "/" not in category else {"category": category}

    raw_chunks = kl.search(query_vector=vec, top_k=top_k, where=where)
    return reranker.rerank(query, raw_chunks, top_k=final_top_k)


def ask_model(prompt: str, stream: bool = True) -> None:
    url = LLAMA_SERVER_URL.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": "Qwen3.6-27B",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": 0.6,
        "max_tokens": 2048,
    }
    if stream:
        with requests.post(url, json=body, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        token = delta.get("content") or delta.get("reasoning_content", "")
                        if token:
                            print(token, end="", flush=True)
                    except (json.JSONDecodeError, KeyError):
                        pass
        print()
    else:
        resp = requests.post(url, json=body, timeout=300)
        resp.raise_for_status()
        print(resp.json()["choices"][0]["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG query tool 2.0")
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--category", "-c", default=None,
                        help="Filter by domain (software|database|devops|network) or category (software/rust)")
    parser.add_argument("--top-k", "-k", type=int, default=RAG_TOP_K, help="Initial retrieval count")
    parser.add_argument("--final-top-k", type=int, default=RAG_FINAL_TOP_K, help="After re-ranking count")
    parser.add_argument("--show-chunks", "-v", action="store_true", help="Print chunks before answer")
    parser.add_argument("--no-model", action="store_true", help="Only retrieve, do not call model")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--feedback", action="store_true", help="Use feedback loop")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder re-ranking")
    args = parser.parse_args()

    kl = KnowledgeLayer(
        chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT,
        collection_name=CHROMA_COLLECTION, embedding_dim=EMBEDDING_DIM,
    )

    if args.stats:
        try:
            count = kl.count()
            print(f"Collection : {CHROMA_COLLECTION}")
            print(f"Vectors    : {count}")
            print(f"Store      : ChromaDB ({CHROMA_HOST}:{CHROMA_PORT})")
        except Exception as exc:
            print(f"Error: {exc}")
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    print(f"\nLoading embedding model ({EMBEDDING_MODEL})…")
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBED_DEVICE)

    reranker = Reranker(
        model_name=RERANKER_MODEL, device=RERANKER_DEVICE,
        top_k=RERANK_TOP_K, score_threshold=RERANK_SCORE_THRESHOLD,
    )

    if args.feedback:
        orchestrator = RAGOrchestrator(
            knowledge_layer=kl, embed_model=model, reranker=reranker,
            llama_base_url=LLAMA_SERVER_URL,
            top_k=args.top_k, final_top_k=args.final_top_k,
            score_threshold=RAG_SCORE_THRESHOLD,
            max_retries=FEEDBACK_MAX_RETRIES, min_chunks=FEEDBACK_MIN_CHUNKS,
        )
        messages = [{"role": "user", "content": args.query}]
        result = orchestrator.execute(messages)
        chunks = result.chunks
        print(f"\nFeedback loop: strategy={result.strategy} retries={result.retries} chunks={len(chunks)}")
    else:
        print(f"Searching for: {args.query!r}\n")
        chunks = retrieve(args.query, args.top_k, args.final_top_k, args.category, model, kl, reranker)

    if not chunks:
        print("No relevant chunks found.")
        if not args.no_model:
            print("\nAnswering from model base knowledge only…\n")
            ask_model(args.query)
        return

    context_lines = ["=== Retrieved Chunks ===\n"] if args.show_chunks else []
    rag_lines = []
    for i, c in enumerate(chunks, 1):
        md = c.get("metadata", {})
        cat = md.get("category", "?")
        title = md.get("title", "?")
        score = c.get("rerank_score", c.get("score", 0))
        text = c.get("text", "").strip()
        blurb = f"[{i}] {cat} | {title} (score={score:.3f})\n{text[:300]}…\n"
        if args.show_chunks:
            context_lines.append(blurb)
        rag_lines.append(f"[{i}] Source: {cat} — {title}\n{text}")

    if args.show_chunks:
        print("\n".join(context_lines))

    if args.no_model:
        return

    context_block = (
        "Use the following documentation to answer the question.\n\n"
        "--- DOCUMENTATION ---\n"
        + "\n\n".join(rag_lines)
        + "\n--- END ---\n\n"
        f"Question: {args.query}"
    )
    print("=== Model Response ===\n")
    ask_model(context_block)


if __name__ == "__main__":
    main()
