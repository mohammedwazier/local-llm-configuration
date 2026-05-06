import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from urllib3.util.retry import Retry
import trafilatura

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBEDDING_DIM,
    EMBEDDING_MODEL, EMBED_BATCH_SIZE, EMBED_DEVICE,
    CHUNK_SIZE, CHUNK_OVERLAP, REQUEST_DELAY, REQUEST_TIMEOUT,
)
from knowledge_layer import KnowledgeLayer
from sources import SOURCES, DOMAINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHECKPOINT_FILE = Path(__file__).parent / ".ingest_checkpoint.json"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        if isinstance(data, list):
            return {u: {"hash": "", "last_modified": "", "etag": ""} for u in data}
        return data
    return {}


def save_checkpoint(entries: dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(entries, indent=2, sort_keys=True))


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3, backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; rag-ingest/2.0; documentation crawler)",
    })
    return session


def fetch_and_extract(url: str, session: requests.Session) -> Optional[str]:
    try:
        log.debug("GET %s", url)
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        log.warning("Fetch failed: %s — %s", url, exc)
        return None
    text = trafilatura.extract(
        html, include_comments=False, include_tables=True,
        no_fallback=False, favor_precision=False,
    )
    if not text or len(text.strip()) < 200:
        log.warning("Low-content extraction from %s (%d chars)", url, len(text or ""))
        return None
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
                current = current[-overlap:].strip()
                current = (current + "\n\n" + para).strip() if current else para
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= chunk_size:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            chunks.append(current)
                            current = current[-overlap:].strip()
                            current = (current + " " + sent).strip() if current else sent
                        else:
                            words = sent.split()
                            for i in range(0, len(words), chunk_size // 6):
                                piece = " ".join(words[i: i + chunk_size // 6])
                                if piece:
                                    chunks.append(piece)
                            current = ""
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c.strip()) > 80]


def ingest(categories: List[str], force: bool = False) -> None:
    log.info("Loading embedding model: %s (%s)", EMBEDDING_MODEL, EMBED_DEVICE)
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBED_DEVICE)
    log.info("Embedding model loaded.")

    kl = KnowledgeLayer(
        chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT,
        collection_name=CHROMA_COLLECTION, embedding_dim=EMBEDDING_DIM,
    )

    session = make_session()
    checkpoint = {} if force else load_checkpoint()

    total_pages = sum(len(SOURCES[c]["pages"]) for c in categories)
    total_chunks = 0
    skipped_pages = 0

    with tqdm(total=total_pages, unit="page", desc="Ingesting") as pbar:
        for category in categories:
            source = SOURCES[category]
            cat_title = source["title"]
            pages = source["pages"]
            log.info("── Category: %s (%s) ──", category, cat_title)

            for url, page_title in pages:
                pbar.set_postfix({"url": url[-50:]})
                cached = checkpoint.get(url, {})
                cached_hash = cached.get("hash", "") if isinstance(cached, dict) else ""

                text = fetch_and_extract(url, session)
                if text is None:
                    pbar.update(1)
                    time.sleep(REQUEST_DELAY)
                    continue

                content_hash = hashlib.sha256(text.encode()).hexdigest()
                if url in checkpoint and cached_hash == content_hash and not force:
                    log.debug("Skip (unchanged): %s", url)
                    skipped_pages += 1
                    pbar.update(1)
                    continue

                chunks = chunk_text(text)
                if not chunks:
                    log.warning("No chunks from %s", url)
                    pbar.update(1)
                    time.sleep(REQUEST_DELAY)
                    continue

                domain = category.split("/")[0]
                subcategory = category.split("/")[1] if "/" in category else category

                ids = []
                embeddings = []
                metadatas = []
                documents = []
                for i, chunk in enumerate(chunks):
                    chunk_id = KnowledgeLayer.point_id(url, i)
                    vec = model.encode(chunk, normalize_embeddings=True).tolist()
                    ids.append(chunk_id)
                    embeddings.append(vec)
                    metadatas.append({
                        "url": url,
                        "title": page_title,
                        "category": category,
                        "domain": domain,
                        "subcategory": subcategory,
                        "chunk_index": i,
                    })
                    documents.append(chunk)

                kl.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
                total_chunks += len(chunks)

                checkpoint[url] = {
                    "hash": content_hash,
                    "last_modified": "",
                    "etag": "",
                }
                save_checkpoint(checkpoint)

                log.info("  [%s] %d chunks  ← %s", category, len(chunks), page_title)
                pbar.update(1)
                time.sleep(REQUEST_DELAY)

    count = kl.count()
    log.info("Done. New chunks: %d | Skipped pages: %d | Total vectors: %s", total_chunks, skipped_pages, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into ChromaDB.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--category", metavar="CAT", help="Single category (e.g. software/rust)")
    group.add_argument("--domain", metavar="DOMAIN", choices=list(DOMAINS.keys()), help="All categories in a domain")
    group.add_argument("--list", action="store_true", help="List categories and exit")
    group.add_argument("--stats", action="store_true", help="Show collection stats and exit")
    group.add_argument("--clear", action="store_true", help="Delete collection and exit")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if unchanged")
    args = parser.parse_args()

    kl = KnowledgeLayer(
        chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT,
        collection_name=CHROMA_COLLECTION, embedding_dim=EMBEDDING_DIM,
    )

    if args.list:
        for cat in sorted(SOURCES):
            src = SOURCES[cat]
            print(f"  {cat:<30s}  {src['title']}  ({len(src['pages'])} pages)")
        return

    if args.stats:
        count = kl.count()
        print(f"Collection : {CHROMA_COLLECTION}")
        print(f"Vectors    : {count}")
        print(f"Store      : ChromaDB ({CHROMA_HOST}:{CHROMA_PORT})")
        return

    if args.clear:
        kl.delete_collection()
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        log.info("Collection '%s' deleted.", CHROMA_COLLECTION)
        return

    if args.category:
        if args.category not in SOURCES:
            print(f"Unknown category '{args.category}'. Run --list to see options.")
            sys.exit(1)
        categories = [args.category]
    elif args.domain:
        categories = DOMAINS[args.domain]
    else:
        categories = sorted(SOURCES.keys())

    ingest(categories, force=args.force)


if __name__ == "__main__":
    main()
