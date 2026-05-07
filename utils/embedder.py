"""Embedder factory: local SentenceTransformer or remote HTTP shim.

Selected by env var EMBED_BACKEND:
- unset / "local"      → SentenceTransformer (default, training paths)
- "remote_http"        → RemoteEmbedder pointing at EMBED_SERVER_URL
                         (used by holdout pipeline to share one GPU embedder)

Caller-visible API matches the subset of SentenceTransformer used in this repo:
- ``encode(texts, normalize_embeddings=True, show_progress_bar=False)``
  returning np.ndarray of shape (N, dim).
"""
import itertools
import os
import random
import threading
import time

import numpy as np
import requests


class RemoteEmbedder:
    """HTTP client for one or more embed servers.

    ``urls`` accepts a single URL or a comma-separated list. Requests are
    distributed round-robin across the pool to spread load. On a request
    failure we retry once on the *next* URL, then give up.
    """

    def __init__(self, urls, model_name: str, timeout: float = 120.0):
        if isinstance(urls, str):
            url_list = [u.strip().rstrip("/") for u in urls.split(",") if u.strip()]
        else:
            url_list = [str(u).rstrip("/") for u in urls if str(u).strip()]
        if not url_list:
            raise ValueError("RemoteEmbedder requires at least one URL")
        # Stagger the cycle starting offset per process so concurrent
        # workers don't all hit url_list[0] on their first request.
        offset = random.randint(0, len(url_list) - 1) if len(url_list) > 1 else 0
        self.urls = url_list[offset:] + url_list[:offset]
        self.model_name = model_name
        self.timeout = timeout
        self._cycle = itertools.cycle(self.urls)
        self._lock = threading.Lock()

    def _next_url(self) -> str:
        with self._lock:
            return next(self._cycle)

    def encode(self, texts, normalize_embeddings: bool = True, show_progress_bar: bool = False, **_):
        single = isinstance(texts, str)
        payload_texts = [texts] if single else list(texts)
        if not payload_texts:
            return np.zeros((0, 0), dtype=np.float32)
        last_err = None
        attempts = max(2, len(self.urls))
        for _ in range(attempts):
            url = self._next_url()
            try:
                resp = requests.post(
                    f"{url}/embed",
                    json={"texts": payload_texts, "normalize": bool(normalize_embeddings)},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                arr = np.asarray(resp.json()["embeddings"], dtype=np.float32)
                return arr[0] if single else arr
            except (requests.RequestException, ValueError) as exc:
                last_err = exc
                time.sleep(0.5)
        raise RuntimeError(f"RemoteEmbedder failed after {attempts} attempts: {last_err}") from last_err


def get_embedder(model_name: str):
    backend = os.environ.get("EMBED_BACKEND", "local").lower()
    if backend == "remote_http":
        url = os.environ.get("EMBED_SERVER_URL")
        if not url:
            raise RuntimeError("EMBED_BACKEND=remote_http requires EMBED_SERVER_URL")
        return RemoteEmbedder(url, model_name)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
