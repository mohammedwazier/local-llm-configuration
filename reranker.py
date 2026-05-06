import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        top_k: int = 5,
        score_threshold: float = 0.1,
    ):
        self._top_k = top_k
        self._score_threshold = score_threshold
        log.info("Loading reranker: %s (%s)", model_name, device)
        self._model = CrossEncoder(model_name, device=device)
        log.info("Reranker ready.")

    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        if not chunks:
            return []

        k = top_k or self._top_k
        pairs = [(query, c.get("text", c.get("metadata", {}).get("text", ""))) for c in chunks]
        scores = self._model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        chunks.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)

        filtered = [c for c in chunks if c.get("rerank_score", 0.0) >= self._score_threshold]
        return filtered[:k]
