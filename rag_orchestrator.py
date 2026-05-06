import logging
import re
from typing import Any, List, Optional

from knowledge_layer import KnowledgeLayer
from reranker import Reranker
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class Analysis:
    def __init__(self, query: str):
        self.query = query
        self.intent: str = "question"
        self.category: Optional[str] = None
        self.domain: Optional[str] = None
        self.keywords: List[str] = []
        self.expansions: List[str] = []


class RAGResult:
    def __init__(self, query: str):
        self.query = query
        self.chunks: List[dict] = []
        self.response: Optional[str] = None
        self.quality_score: float = 0.0
        self.passed: bool = False
        self.retries: int = 0
        self.strategy: str = "standard"


class RAGOrchestrator:
    FEEDBACK_DOMAINS = [
        "software", "database", "devops", "network", "iot", "security", "management",
    ]

    def __init__(
        self,
        knowledge_layer: KnowledgeLayer,
        embed_model: SentenceTransformer,
        reranker: Optional[Reranker],
        llama_base_url: str,
        top_k: int = 10,
        final_top_k: int = 3,
        score_threshold: float = 0.35,
        max_retries: int = 3,
        min_chunks: int = 2,
    ):
        self._kl = knowledge_layer
        self._embed = embed_model
        self._reranker = reranker
        self._llama_base_url = llama_base_url.rstrip("/")
        self._top_k = top_k
        self._final_top_k = final_top_k
        self._score_threshold = score_threshold
        self._max_retries = max_retries
        self._min_chunks = min_chunks

    # ── 1. ANALYZE ─────────────────────────────────────────────────────────

    def analyze(self, messages: List[dict]) -> Analysis:
        query = self._extract_query(messages)
        analysis = Analysis(query)

        query_lower = query.lower()

        known_domains = {
            "rust", "javascript", "python", "golang", "java", "zig", "bash", "c++", "c ",
            "postgresql", "postgres", "sql", "database", "replication",
            "terraform", "ansible", "docker", "kubernetes", "k3s", "proxmox", "gitlab",
            "network", "subnet", "dns", "bgp", "firewall", "vpn",
            "mqtt", "coap", "iot", "edge", "modbus", "opc-ua", "lwm2m",
            "security", "owasp", "tls", "devsecops", "vault",
        }
        found = [d for d in known_domains if d in query_lower]
        analysis.keywords = found

        category_map = {
            "rust": "software/rust",
            "javascript": "software/javascript",
            "python": "software/python",
            "golang": "software/golang",
            "java": "software/java",
            "zig": "software/zig",
            "bash": "software/bash",
            "c++": "software/cpp",
            "postgresql": "database/postgresql",
            "postgres": "database/postgresql",
            "replication": "database/replication",
            "terraform": "devops/terraform",
            "ansible": "devops/ansible",
            "docker": "devops/docker",
            "kubernetes": "devops/kubernetes",
            "k3s": "iot/edge",
            "proxmox": "devops/proxmox",
            "gitlab": "devops/gitlab",
            "mqtt": "iot/mqtt",
            "coap": "iot/coap",
            "iot": "iot/protocols",
            "modbus": "iot/protocols",
            "opc-ua": "iot/protocols",
            "lwm2m": "iot/protocols",
            "owasp": "security/general",
            "devsecops": "security/devsecops",
        }
        matched_cat = None
        for kw in found:
            if kw in category_map:
                matched_cat = category_map[kw]
                break
        if matched_cat:
            analysis.category = matched_cat
            analysis.domain = matched_cat.split("/")[0]

        if any(w in query_lower for w in ["how", "what", "why", "when", "where", "explain", "describe"]):
            analysis.intent = "question"
        elif any(w in query_lower for w in ["write", "create", "generate", "implement", "fix", "debug"]):
            analysis.intent = "task"
        else:
            analysis.intent = "question"

        return analysis

    @staticmethod
    def _extract_query(messages: List[dict]) -> str:
        user_turns = [m for m in messages if m.get("role") == "user"][-2:]
        parts = []
        for m in user_turns:
            content = m.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
        return " ".join(parts).strip()

    # ── 2. RETRIEVE ────────────────────────────────────────────────────────

    def retrieve(self, analysis: Analysis, strategy: str = "standard") -> List[dict]:
        query = analysis.query

        if strategy == "expand":
            query = self._expand_query(analysis)
        elif strategy == "unfiltered":
            analysis.category = None

        vec = self._embed.encode(query, normalize_embeddings=True).tolist()

        where = None
        if analysis.category:
            if "/" in analysis.category:
                where = {"category": analysis.category}
            else:
                where = {"domain": analysis.domain}

        chunks = self._kl.search(
            query_vector=vec,
            top_k=self._top_k,
            where=where,
        )

        if strategy == "widen":
            chunks = self._kl.search(
                query_vector=vec,
                top_k=self._top_k * 2,
                where=None,
            )

        if chunks:
            if self._reranker is not None:
                return self._reranker.rerank(query, chunks, top_k=self._final_top_k)
            return chunks[:self._final_top_k]

        return []

    @staticmethod
    def _expand_query(analysis: Analysis) -> str:
        expansion_map = {
            "rust": "Rust programming language memory safety concurrency",
            "python": "Python programming language functions modules",
            "kubernetes": "Kubernetes container orchestration pods deployments services",
            "docker": "Docker container images compose networking",
            "postgresql": "PostgreSQL database SQL queries indexes performance",
            "terraform": "Terraform infrastructure as code resources modules state",
            "mqtt": "MQTT protocol publish subscribe broker QoS topics",
        }
        expansions = []
        for kw in analysis.keywords:
            if kw in expansion_map:
                expansions.append(expansion_map[kw])
        if expansions:
            return analysis.query + " " + " ".join(expansions)
        return analysis.query

    # ── 3. EVALUATE ────────────────────────────────────────────────────────

    def evaluate(self, result: RAGResult) -> RAGResult:
        score = 0.0
        n = len(result.chunks)
        if n == 0:
            result.quality_score = 0.0
            result.passed = False
            return result

        avg_score = sum(c.get("rerank_score", c.get("score", 0)) for c in result.chunks) / n
        score += min(avg_score, 1.0) * 0.4

        coverage = min(n / self._min_chunks, 1.0)
        score += coverage * 0.3

        if result.response:
            resp_lower = result.response.lower()
            q_words = set(re.findall(r"\w+", result.query.lower()))
            resp_words = set(re.findall(r"\w+", resp_lower))
            if q_words:
                overlap = len(q_words & resp_words) / len(q_words)
                score += min(overlap, 1.0) * 0.3

            if len(result.response.strip().split()) < 5:
                score *= 0.5

        result.quality_score = round(score, 3)
        result.passed = score >= self._score_threshold and n >= self._min_chunks
        return result

    # ── FEEDBACK LOOP ──────────────────────────────────────────────────────

    def _next_strategy(self, current: str, retry: int) -> str:
        strategies = ["standard", "unfiltered", "widen", "expand"]
        if current == "standard":
            return "unfiltered" if retry < 2 else "widen"
        return strategies[min(retry + 1, len(strategies) - 1)]

    def execute(
        self,
        messages: List[dict],
    ) -> RAGResult:
        analysis = self.analyze(messages)
        log.info("ANALYZE  intent=%s category=%s keywords=%s", analysis.intent, analysis.category, analysis.keywords)

        result = RAGResult(analysis.query)

        for attempt in range(self._max_retries + 1):
            strategy = self._next_strategy(result.strategy, attempt) if attempt > 0 else "standard"
            result.strategy = strategy
            log.info("RETRIEVE attempt=%d strategy=%s", attempt + 1, strategy)

            chunks = self.retrieve(analysis, strategy=strategy)
            result.chunks = chunks
            result.retries = attempt

            log.info("RETRIEVE got %d chunks", len(chunks))

            result.passed = len(chunks) >= self._min_chunks
            result.quality_score = min(len(chunks) / self._min_chunks, 1.0)

            if result.passed:
                log.info("RETRIEVE passed — %d chunks (strategy=%s)", len(chunks), strategy)
                return result

            log.info("RE-ROUTE strategy=%s -> %s (attempt %d/%d)",
                      strategy, self._next_strategy(strategy, attempt), attempt + 1, self._max_retries)

        log.info("EXHAUSTED retries — returning best effort (%d chunks)", len(result.chunks))
        return result

    # ── context builder ───────────────────────────────────────────────────

    @staticmethod
    def build_context(chunks: List[dict], max_chars: int = 500) -> str:
        lines = ["### RETRIEVED DOCUMENTATION\n"]
        for i, c in enumerate(chunks, 1):
            md = c.get("metadata", {})
            cat = md.get("category", "?")
            title = md.get("title", "?")
            url = md.get("url", "")
            text = c.get("text", "").strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "…"
            lines.append(f"[{i}] {cat} — {title}")
            if url:
                lines.append(f"Source: {url}")
            lines.append(text)
            lines.append("")
        lines.append("### END DOCUMENTATION\n")
        return "\n".join(lines)
