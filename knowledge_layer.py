import hashlib
import logging
from typing import Any, Iterable, List, Optional
from uuid import uuid5, NAMESPACE_URL

import chromadb
from chromadb import HttpClient as ChromaHttpClient
from chromadb.config import Settings as ChromaSettings

log = logging.getLogger(__name__)


class KnowledgeLayer:
    """
    Vector store backed by ChromaDB (Docker).
    """

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        collection_name: str = "tech_knowledge",
        embedding_dim: int = 1024,
    ):
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        self._chroma = ChromaHttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None

    # ── collection access ──────────────────────────────────────────────────

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._chroma.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── collection access ──────────────────────────────────────────────────

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._chroma.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── point ID ───────────────────────────────────────────────────────────

    @staticmethod
    def point_id(url: str, idx: int) -> str:
        return str(uuid5(NAMESPACE_URL, f"{url}#{idx}"))

    # ── upsert ─────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        documents: List[str],
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    # ── search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> List[dict]:
        return self._search_chroma(query_vector, top_k, where)

    def _search_chroma(
        self, query_vector: List[float], top_k: int, where: Optional[dict] = None
    ) -> List[dict]:
        try:
            where_clause = self._to_chroma_where(where) if where else None
            raw = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            log.warning("ChromaDB query failed: %s", exc)
            return []

        return self._format_chroma_results(raw)

    # ── format helpers ─────────────────────────────────────────────────────

    def _format_chroma_results(self, raw: Any) -> List[dict]:
        if not raw or not raw["ids"]:
            return []
        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "score": 1.0 - raw["distances"][0][i] if raw.get("distances") else 0.0,
                "metadata": raw["metadatas"][0][i] if raw.get("metadatas") else {},
                "text": raw["documents"][0][i] if raw.get("documents") else "",
            })
        return results

    # ── filter translation ─────────────────────────────────────────────────

    @staticmethod
    def _to_chroma_where(filt: dict) -> dict:
        translated = {}
        for key, value in filt.items():
            if isinstance(value, str):
                translated[key] = {"$eq": value}
            elif isinstance(value, dict):
                translated[key] = value
            else:
                translated[key] = {"$eq": str(value)}
        return translated

    # ── scroll all (for training-data generation) ──────────────────────────

    def get_all_points(self, batch_size: int = 200) -> Iterable[dict]:
        offset = 0
        while True:
            raw = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas", "documents"],
            )
            if not raw or not raw["ids"]:
                break
            for i in range(len(raw["ids"])):
                md = raw["metadatas"][i] or {} if raw.get("metadatas") else {}
                doc = raw["documents"][i] if raw.get("documents") else ""
                yield {
                    "id": raw["ids"][i],
                    "metadata": md,
                    "text": doc,
                    **md,
                }
            offset += batch_size

    # ── stats ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        try:
            self._chroma.delete_collection(self._collection_name)
            self._collection = None
            log.info("Chroma collection '%s' deleted.", self._collection_name)
        except Exception as exc:
            log.warning("Delete failed: %s", exc)

    # ── content-hash helpers for auto-update ────────────────────────────────

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
