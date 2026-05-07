import math

import torch

from methods.base_method import BaseMethod
from utils.llm_client import generate as llm_generate

try:
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except ImportError:
    transformers_logging = None


class HistoryRAG(BaseMethod):
    name = "HistoryRAG"

    def __init__(
        self,
        top_k=3,
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        temperature=0.0,
        max_new_tokens=1024,
    ):
        super().__init__()
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._embedder = None
        self._memory_vectors = []

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from utils.embedder import get_embedder
        except ImportError as exc:
            raise RuntimeError(
                "HistoryRAG requires sentence-transformers. "
                "Please install it first: pip install sentence-transformers"
            ) from exc
        self._embedder = get_embedder(self.embedding_model_name)
        return self._embedder

    def _encode(self, texts):
        if not texts:
            return []
        embedder = self._load_embedder()
        vectors = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = [list(vec) for vec in vectors]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve_memory(self, task, **kwargs):
        if not self.memory:
            return []
        query = str(kwargs.get("query", ""))
        if not query.strip():
            return self.memory[-self.top_k :]
        query_vec = self._encode([query])[0]
        scored = []
        for idx, record in enumerate(self.memory):
            rec_task = record.get("task")
            if rec_task and rec_task != task.name:
                continue
            score = self._cosine(query_vec, self._memory_vectors[idx])
            scored.append((score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[: self.top_k]]

    def generate(self, task, prompt, **kwargs):
        retrieved = kwargs.get("memory", [])
        generation_prompt = self.build_memory_augmented_prompt(prompt, retrieved)
        return llm_generate(
            model_name=self.generation_model_name,
            prompt=generation_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

    def update_memory(self, task, output, **kwargs):
        record = kwargs.get("record")
        if not record:
            return
        retrieved = kwargs.get("memory") or []
        enriched = {"task": task.name, **record}
        self.memory.append(enriched)
        enriched["memory_qids"] = [
            r.get("qid") for r in self.memory if r.get("task") == task.name
        ]
        enriched["retrieved_qids"] = [r.get("qid") for r in retrieved]
        text = (
            f"Q: {enriched.get('question', '')}\n"
            f"A: {enriched.get('model_output', '')}\n"
            f"Feedback: {enriched.get('feedback', '')}"
        )
        self._memory_vectors.extend(self._encode([text]))

    def reset_task_state(self, task):
        # Keep memory across the stream of one dataset.
        return None

    def restore_state_from_memory(self, task):
        """Re-embed all preloaded memory records so retrieval works on resume.

        _memory_vectors must stay index-parallel with self.memory (retrieve_memory
        accesses self._memory_vectors[idx] where idx iterates over self.memory).
        Encodes in small batches to avoid GPU OOM when many records are loaded.

        Fast path: if env BWT_HRAG_VECTOR_CACHE points to an .npz with `qids` +
        `vectors` arrays, look up self.memory[i]["qid"] in the cache and skip
        re-encoding. The cache must have been built from the *same* training
        memory file with the *same* embedding model — BWT launcher precomputes
        it once per (model, task).
        """
        if not self.memory:
            return
        import os as _os
        cache_path = _os.environ.get("BWT_HRAG_VECTOR_CACHE")
        if cache_path and _os.path.isfile(cache_path):
            import numpy as _np
            blob = _np.load(cache_path, allow_pickle=False)
            cache_qids = list(blob["qids"])
            cache_vecs = blob["vectors"]
            qid_to_idx = {q: i for i, q in enumerate(cache_qids)}
            try:
                self._memory_vectors = [
                    list(cache_vecs[qid_to_idx[rec["qid"]]]) for rec in self.memory
                ]
            except KeyError as exc:
                raise RuntimeError(
                    f"BWT_HRAG_VECTOR_CACHE missing qid {exc}; rebuild cache"
                ) from exc
            print(
                f"[HistoryRAG Resume] Loaded {len(self._memory_vectors)} cached vectors "
                f"from {cache_path}"
            )
            return
        texts = [
            f"Q: {rec.get('question', '')}\n"
            f"A: {rec.get('model_output', '')}\n"
            f"Feedback: {rec.get('feedback', '')}"
            for rec in self.memory
        ]
        batch_size = 8
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            all_vectors.extend(self._encode(texts[i : i + batch_size]))
        self._memory_vectors = all_vectors
        print(f"[HistoryRAG Resume] Re-embedded {len(self._memory_vectors)} memory vectors")
