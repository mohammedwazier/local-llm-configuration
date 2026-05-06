import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from knowledge_layer import KnowledgeLayer

log = logging.getLogger(__name__)


class AutoUpdater:
    CHECKPOINT_FILE = Path(__file__).parent / ".ingest_checkpoint.json"

    def __init__(
        self,
        knowledge_layer: KnowledgeLayer,
        ingest_fn: Callable,
        interval: int = 86400,
        enabled: bool = False,
    ):
        self._kl = knowledge_layer
        self._ingest_fn = ingest_fn
        self._interval = interval
        self._enabled = enabled
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._enabled:
            log.info("Auto-update disabled.")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-updater")
        self._thread.start()
        log.info("Auto-updater started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                log.info("Auto-update: checking sources for changes …")
                stale = self._find_stale_urls()
                if stale:
                    log.info("Auto-update: %d URLs may have changed — re-ingesting", len(stale))
                    self._ingest_fn(force=True)
                else:
                    log.info("Auto-update: no changes detected")
            except Exception as exc:
                log.warning("Auto-update error: %s", exc)

    def _find_stale_urls(self) -> list:
        checkpoint = self._load_checkpoint()
        if not checkpoint:
            return []
        session = self._make_session()
        stale = []
        for url in list(checkpoint):
            try:
                resp = session.head(url, timeout=10)
                last_mod = resp.headers.get("Last-Modified")
                etag = resp.headers.get("ETag")
                entry = checkpoint.get(url, {})
                if isinstance(entry, dict):
                    old_mod = entry.get("last_modified", "")
                    old_etag = entry.get("etag", "")
                    if last_mod and last_mod != old_mod:
                        stale.append(url)
                        continue
                    if etag and etag != old_etag:
                        stale.append(url)
                        continue
            except Exception:
                pass
        return stale

    @classmethod
    def _load_checkpoint(cls) -> dict:
        if cls.CHECKPOINT_FILE.exists():
            data = json.loads(cls.CHECKPOINT_FILE.read_text())
            if isinstance(data, list):
                return {u: {"last_modified": "", "etag": ""} for u in data}
            return data
        return {}

    @classmethod
    def save_checkpoint(cls, urls: set) -> None:
        existing = cls._load_checkpoint()
        for url in urls:
            if url not in existing:
                existing[url] = {"last_modified": "", "etag": ""}
        cls.CHECKPOINT_FILE.write_text(json.dumps(existing, indent=2, sort_keys=True))

    @staticmethod
    def _make_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(total=2, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "rag-auto-updater/1.0"})
        return session
