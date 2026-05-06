import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBEDDING_DIM,
    EMBEDDING_MODEL, EMBED_BATCH_SIZE, EMBED_DEVICE,
    PINECONE_API_KEY, PINECONE_INDEX,
)
from knowledge_layer import KnowledgeLayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("migrate")


def main():
    log.info("This script migrates from Qdrant to ChromaDB.")
    log.info("Requires qdrant-client and an active Qdrant instance on localhost:6333.")

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        log.error("qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer

    qdrant = QdrantClient(url="http://localhost:6333")
    collections = qdrant.get_collections().collections
    if not collections:
        log.error("No Qdrant collections found.")
        sys.exit(1)

    old_name = collections[0].name
    log.info("Migrating from Qdrant collection: %s", old_name)

    kl = KnowledgeLayer(
        chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT,
        collection_name=CHROMA_COLLECTION, embedding_dim=EMBEDDING_DIM,
    )

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBED_DEVICE)

    offset = None
    migrated = 0
    while True:
        points, offset = qdrant.scroll(
            collection_name=old_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        batch_ids = []
        batch_embs = []
        batch_mds = []
        batch_docs = []
        for p in points:
            payload = p.payload or {}
            text = payload.get("text", "")
            if not text:
                continue
            batch_ids.append(str(p.id))
            batch_docs.append(text)
            batch_mds.append({
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "category": payload.get("category", ""),
                "domain": payload.get("domain", ""),
                "subcategory": payload.get("subcategory", ""),
                "chunk_index": payload.get("chunk_index", 0),
            })

        if batch_ids:
            embeddings = model.encode(batch_docs, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
            kl.upsert(ids=batch_ids, embeddings=embeddings.tolist(), metadatas=batch_mds, documents=batch_docs)
            migrated += len(batch_ids)

        log.info("Migrated %d points so far …", migrated)
        if offset is None:
            break

    log.info("Migration complete. %d points migrated to ChromaDB.", migrated)
    log.info("Collection '%s' has %d vectors.", CHROMA_COLLECTION, kl.count())


if __name__ == "__main__":
    main()
