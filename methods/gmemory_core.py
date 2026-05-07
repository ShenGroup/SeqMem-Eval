from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx

from prompts.gmemory import PROMPTS, parse_numbered_list

# Avoid outbound telemetry noise in restricted/offline environments.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    try:
        return list(value)
    except TypeError:
        return [value]


def _to_first_nested_list(value):
    outer = _to_list(value)
    if not outer:
        return []
    return _to_list(outer[0])


def _to_float_list(value):
    return [float(x) for x in _to_list(value)]


@dataclass
class TaskRecordStore:
    persist_dir: Path
    embedding_model_name: str

    def __post_init__(self):
        self.persist_dir = Path(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._records_file = self.persist_dir / "records.json"

        self._embedder = None

        self._use_chroma = False
        self._collection = None
        self._records: list[dict] = []

        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))
            self._collection = client.get_or_create_collection(name="task_records")
            self._use_chroma = True
        except Exception:
            self._use_chroma = False

        if not self._use_chroma and self._records_file.exists():
            with self._records_file.open("r", encoding="utf-8") as f:
                self._records = json.load(f) or []

    def _save_local_records(self) -> None:
        if self._use_chroma:
            return
        with self._records_file.open("w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embedder is None:
            try:
                from utils.embedder import get_embedder
            except ImportError as exc:
                raise RuntimeError(
                    "AutoGen-GMemory requires sentence-transformers. "
                    "Please install it first: pip install sentence-transformers"
                ) from exc
            self._embedder = get_embedder(self.embedding_model_name)
        vectors = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Chroma type validation is strict in some versions; ensure native Python floats.
        return [[float(x) for x in vec] for vec in vectors]

    def count(self) -> int:
        if self._use_chroma:
            return int(self._collection.count())
        return len(self._records)

    def _to_storage_metadata(self, record: dict) -> dict:
        return {
            "record_id": str(record.get("record_id", "")),
            "task_name": str(record.get("task_name", "")),
            "task_main": str(record.get("task_main", "")),
            "task_description": str(record.get("task_description", "")),
            "task_trajectory": str(record.get("task_trajectory", "")),
            "label": bool(record.get("label", False)),
            "extra_fields": json.dumps(record.get("extra_fields", {}), ensure_ascii=False),
            "created_at": str(record.get("created_at", "")),
        }

    def _from_storage_metadata(self, metadata: dict, embedding: list[float] | None = None) -> dict:
        extra_raw = metadata.get("extra_fields", "{}")
        try:
            extra_fields = json.loads(extra_raw) if isinstance(extra_raw, str) else dict(extra_raw)
        except Exception:
            extra_fields = {}
        return {
            "record_id": str(metadata.get("record_id", "")),
            "task_name": str(metadata.get("task_name", "")),
            "task_main": str(metadata.get("task_main", "")),
            "task_description": str(metadata.get("task_description", "")),
            "task_trajectory": str(metadata.get("task_trajectory", "")),
            "label": bool(metadata.get("label", False)),
            "extra_fields": extra_fields,
            "created_at": str(metadata.get("created_at", "")),
            "embedding": _to_float_list(embedding) if embedding is not None else None,
        }

    def add_record(self, record: dict) -> dict:
        record = dict(record)
        record.setdefault("record_id", str(uuid.uuid4()))
        record.setdefault("created_at", str(time.time()))
        task_main = str(record.get("task_main", "")).strip()
        if not task_main:
            raise ValueError("task_main is required")
        embedding = self._encode([task_main])[0]
        record["embedding"] = embedding

        if self._use_chroma:
            self._collection.add(
                ids=[record["record_id"]],
                documents=[task_main],
                metadatas=[self._to_storage_metadata(record)],
                embeddings=[embedding],
            )
        else:
            self._records.append(record)
            self._save_local_records()
        return record

    def _query_chroma(
        self,
        query: str,
        k: int,
        label: bool | None = None,
    ) -> list[tuple[dict, float]]:
        if k <= 0:
            return []
        q_emb = self._encode([query])[0]
        where = None
        if label is not None:
            where = {"label": bool(label)}
        result = self._collection.query(
            query_embeddings=[q_emb],
            n_results=max(1, int(k)),
            where=where,
            include=["metadatas", "distances", "embeddings"],
        )
        metadatas = _to_first_nested_list(result.get("metadatas"))
        distances = _to_first_nested_list(result.get("distances"))
        embeddings = _to_first_nested_list(result.get("embeddings"))
        rows: list[tuple[dict, float]] = []
        for md, dist, emb in zip(metadatas, distances, embeddings):
            rec = self._from_storage_metadata(md, embedding=_to_float_list(emb) if emb is not None else None)
            rows.append((rec, 1.0 - float(dist)))
        return rows

    def _query_local(
        self,
        query: str,
        k: int,
        label: bool | None = None,
    ) -> list[tuple[dict, float]]:
        if k <= 0:
            return []
        q_emb = self._encode([query])[0]
        scored: list[tuple[dict, float]] = []
        for rec in self._records:
            if label is not None and bool(rec.get("label", False)) != bool(label):
                continue
            score = _cosine(q_emb, _to_float_list(rec.get("embedding")))
            scored.append((dict(rec), float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def similarity_search(
        self,
        query: str,
        k: int,
        label: bool | None = None,
    ) -> list[tuple[dict, float]]:
        query = str(query or "").strip()
        if not query:
            return []
        if self._use_chroma:
            return self._query_chroma(query, k, label)
        return self._query_local(query, k, label)

    def records_for_task_main(self, task_main: str) -> list[dict]:
        task_main = str(task_main or "").strip()
        if not task_main:
            return []

        if self._use_chroma:
            result = self._collection.get(
                where={"task_main": task_main},
                include=["metadatas", "embeddings"],
            )
            metadatas = _to_list(result.get("metadatas"))
            embeddings = _to_list(result.get("embeddings"))
            rows = [
                self._from_storage_metadata(md, embedding=_to_float_list(emb) if emb is not None else None)
                for md, emb in zip(metadatas, embeddings)
            ]
            return rows

        return [dict(rec) for rec in self._records if str(rec.get("task_main", "")) == task_main]

    def all_records(self) -> list[dict]:
        if self._use_chroma:
            result = self._collection.get(include=["metadatas", "embeddings"])
            metadatas = _to_list(result.get("metadatas"))
            embeddings = _to_list(result.get("embeddings"))
            return [
                self._from_storage_metadata(md, embedding=_to_float_list(emb) if emb is not None else None)
                for md, emb in zip(metadatas, embeddings)
            ]
        return [dict(rec) for rec in self._records]


@dataclass
class TaskLayer:
    persist_dir: Path
    task_store: TaskRecordStore
    similarity_threshold: float = 0.7

    def __post_init__(self):
        self.persist_dir = Path(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self.persist_dir / "task_graph.pkl"
        if self._graph_path.exists():
            with self._graph_path.open("rb") as f:
                self.graph = pickle.load(f)
        else:
            self.graph = nx.Graph()

    def _save(self) -> None:
        with self._graph_path.open("wb") as f:
            pickle.dump(self.graph, f)

    def add_task_node(self, task_main: str) -> None:
        task_main = str(task_main or "").strip()
        if not task_main:
            return
        if task_main not in self.graph:
            self.graph.add_node(task_main)

        neighbors = self.task_store.similarity_search(task_main, k=10)
        for rec, score in neighbors:
            neighbor = str(rec.get("task_main", "")).strip()
            if not neighbor or neighbor == task_main:
                continue
            if score < self.similarity_threshold:
                continue
            if neighbor not in self.graph:
                self.graph.add_node(neighbor)
            self.graph.add_edge(task_main, neighbor, weight=float(score))
        self._save()

    def retrieve_related_task(self, query_task: str, node_num: int, hop: int = 1) -> list[str]:
        query_task = str(query_task or "").strip()
        if not query_task:
            return []
        if query_task not in self.graph:
            self.graph.add_node(query_task)
            self._save()

        seeds = [
            str(rec.get("task_main", "")).strip()
            for rec, _ in self.task_store.similarity_search(query_task, k=max(1, int(node_num)))
            if str(rec.get("task_main", "")).strip()
        ]
        if not seeds:
            seeds = [query_task]

        related = set(seeds)
        for node in seeds:
            if node not in self.graph:
                continue
            for n in nx.single_source_shortest_path_length(self.graph, node, cutoff=max(1, int(hop))).keys():
                related.add(n)
        return list(related)

    def cluster_labels(self) -> dict[str, int]:
        labels: dict[str, int] = {}
        for idx, component in enumerate(nx.connected_components(self.graph)):
            for node in component:
                labels[node] = idx
        return labels


@dataclass
class InsightsManager:
    persist_dir: Path
    llm_call: Callable[[str, str, float, int], str]
    curator_model_name: str

    def __post_init__(self):
        self.persist_dir = Path(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._insights_file = self.persist_dir / "insights.json"
        if self._insights_file.exists():
            with self._insights_file.open("r", encoding="utf-8") as f:
                self.insights_memory = json.load(f) or []
        else:
            self.insights_memory = []

    def _save(self) -> None:
        with self._insights_file.open("w", encoding="utf-8") as f:
            json.dump(self.insights_memory, f, ensure_ascii=False, indent=2)

    def clear_insights(self) -> None:
        self.insights_memory = [r for r in self.insights_memory if float(r.get("score", 0.0)) > 0.0]

    def query_insights_with_score(self, task_mains: list[str], top_k: int | None = None) -> list[tuple[str, float]]:
        task_set = {str(t).strip() for t in task_mains if str(t).strip()}
        scored: list[tuple[str, float]] = []
        for row in self.insights_memory:
            rule = str(row.get("rule", "")).strip()
            if not rule:
                continue
            positives = set(str(t).strip() for t in row.get("positive_correlation_tasks", []))
            hit = len(task_set.intersection(positives))
            base = float(row.get("score", 0.0))
            total = hit + base * 0.01
            scored.append((rule, total))
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            scored = scored[: max(0, int(top_k))]
        return scored

    def backward(self, insight: str, reward: float) -> None:
        changed = False
        key = str(insight or "").strip()
        if not key:
            return
        for row in self.insights_memory:
            if key in str(row.get("rule", "")):
                row["score"] = float(row.get("score", 0.0)) + float(reward)
                changed = True
        if changed:
            self.clear_insights()
            self._save()

    def merge_insights(self, max_rules: int = 10) -> None:
        if not self.insights_memory:
            return
        current_rules = "\n".join(str(r.get("rule", "")) for r in self.insights_memory)
        user_prompt = PROMPTS.merge_rules_user_prompt.format(
            current_rules=current_rules,
            limited_number=max(1, int(max_rules)),
        )
        full_prompt = (
            f"[SYSTEM]\n{PROMPTS.merge_rules_system_prompt}\n\n"
            f"[USER]\n{user_prompt}"
        )
        raw = self.llm_call(self.curator_model_name, full_prompt, 0.0, 512)
        merged = parse_numbered_list(raw)
        if not merged:
            seen = set()
            merged = []
            for row in self.insights_memory:
                rule = str(row.get("rule", "")).strip()
                if not rule or rule in seen:
                    continue
                seen.add(rule)
                merged.append(rule)
                if len(merged) >= max(1, int(max_rules)):
                    break

        all_tasks = set()
        for row in self.insights_memory:
            all_tasks.update(str(t).strip() for t in row.get("positive_correlation_tasks", []))

        self.insights_memory = [
            {
                "rule": str(rule).strip(),
                "score": 2.0,
                "positive_correlation_tasks": sorted(all_tasks),
                "negative_correlation_tasks": [],
            }
            for rule in merged
            if str(rule).strip()
        ]
        self._save()

    def _normalize_rule_text(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", str(text or "").strip(" -\n\t"))
        if not clean:
            return ""
        if not clean.endswith("."):
            clean = f"{clean}."
        return clean

    def _parse_operations(self, llm_text: str) -> list[tuple[str, str]]:
        matches = re.findall(r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+)?):\s*(.*)", str(llm_text or ""))
        operations: list[tuple[str, str]] = []
        for op, text in matches:
            cleaned = self._normalize_rule_text(text)
            if not cleaned:
                continue
            head = op.strip()
            op_type = head.split(" ")[0]
            if op_type not in {"REMOVE", "EDIT", "ADD", "AGREE"}:
                continue
            operations.append((head, cleaned))
        return operations[:4]

    def _retrieve_rule_index(self, operation_rule_text: str) -> int:
        needle = str(operation_rule_text or "").strip()
        for i, row in enumerate(self.insights_memory):
            base = str(row.get("rule", "")).strip()
            if not base:
                continue
            if base in needle or needle in base:
                return i
        return -1

    def _is_existing_rule(self, operation_rule_text: str) -> bool:
        return self._retrieve_rule_index(operation_rule_text) >= 0

    def _apply_operations(self, relative_tasks: list[str], operations: list[tuple[str, str]], max_rules_num: int) -> None:
        if not operations:
            return

        valid_ops: list[tuple[str, str]] = []
        for op, rule_text in operations:
            op_type = op.split(" ")[0]
            rule_num = int(op.split(" ")[1]) if " " in op else None
            if op_type == "ADD":
                if self._is_existing_rule(rule_text):
                    continue
                valid_ops.append((op, rule_text))
                continue

            if op_type in {"EDIT", "REMOVE", "AGREE"}:
                if rule_num is None or rule_num <= 0 or rule_num > len(self.insights_memory):
                    continue
                valid_ops.append((op, rule_text))

        list_full = len(self.insights_memory) >= max(1, int(max_rules_num))
        rel_tasks = sorted({str(t).strip() for t in relative_tasks if str(t).strip()})

        for op, rule_text in valid_ops:
            op_type = op.split(" ")[0]
            if op_type == "ADD":
                if len(self.insights_memory) >= max(1, int(max_rules_num)):
                    continue
                self.insights_memory.append(
                    {
                        "rule": rule_text,
                        "score": 2.0,
                        "positive_correlation_tasks": list(rel_tasks),
                        "negative_correlation_tasks": [],
                    }
                )
                continue

            idx = int(op.split(" ")[1]) - 1
            row = self.insights_memory[idx]
            if op_type == "REMOVE":
                row["score"] = float(row.get("score", 0.0)) - (3.0 if list_full else 1.0)
                neg = set(str(t).strip() for t in row.get("negative_correlation_tasks", []))
                neg.update(rel_tasks)
                row["negative_correlation_tasks"] = sorted(neg)
            elif op_type == "AGREE":
                row["score"] = float(row.get("score", 0.0)) + 1.0
                pos = set(str(t).strip() for t in row.get("positive_correlation_tasks", []))
                pos.update(rel_tasks)
                row["positive_correlation_tasks"] = sorted(pos)
            elif op_type == "EDIT":
                row["rule"] = rule_text
                row["score"] = float(row.get("score", 0.0)) + 1.0
                pos = set(str(t).strip() for t in row.get("positive_correlation_tasks", []))
                pos.update(rel_tasks)
                row["positive_correlation_tasks"] = sorted(pos)

        self.clear_insights()
        self._save()

    def finetune_insights(
        self,
        successful_records: list[dict],
        failed_records: list[dict],
        max_rules_num: int = 20,
    ) -> None:
        if not successful_records and not failed_records:
            return

        existing_rules = [str(r.get("rule", "")).strip() for r in self.insights_memory if str(r.get("rule", "")).strip()]
        if not existing_rules:
            existing_rules = ["(empty)"]
        existing_rules_text = "\n".join(f"{i}. {r}" for i, r in enumerate(existing_rules, start=1))

        operations: list[tuple[str, str]] = []

        pairs = []
        for i, failed in enumerate(failed_records):
            if i >= len(successful_records):
                break
            pairs.append((successful_records[i], failed))

        for success_rec, fail_rec in pairs:
            user_prompt = PROMPTS.critique_compare_rules_user_prompt.format(
                task1=success_rec.get("task_description", ""),
                task1_trajectory=success_rec.get("task_trajectory", ""),
                task2=fail_rec.get("task_description", ""),
                task2_trajectory=fail_rec.get("task_trajectory", ""),
                fail_reason=fail_rec.get("extra_fields", {}).get("fail_reason", ""),
                existing_rules=existing_rules_text,
            )
            full_prompt = (
                f"[SYSTEM]\n{PROMPTS.critique_compare_rules_system_prompt}\n\n"
                f"[USER]\n{user_prompt}"
            )
            raw = self.llm_call(self.curator_model_name, full_prompt, 0.0, 512)
            operations.extend(self._parse_operations(raw))

        if successful_records:
            chunks = []
            src = list(successful_records)
            random.shuffle(src)
            chunk_size = max(1, min(5, len(src)))
            for i in range(0, len(src), chunk_size):
                chunks.append(src[i : i + chunk_size])
            for chunk in chunks:
                success_history = []
                for i, item in enumerate(chunk, start=1):
                    success_history.append(
                        f"task{i}:\n{item.get('task_description', '')}\n{item.get('extra_fields', {}).get('key_steps', '')}"
                    )
                user_prompt = PROMPTS.critique_success_rules_user_prompt.format(
                    success_history="\n".join(success_history),
                    existing_rules=existing_rules_text,
                )
                full_prompt = (
                    f"[SYSTEM]\n{PROMPTS.critique_success_rules_system_prompt}\n\n"
                    f"[USER]\n{user_prompt}"
                )
                raw = self.llm_call(self.curator_model_name, full_prompt, 0.0, 512)
                operations.extend(self._parse_operations(raw))

        relative_tasks = [
            str(item.get("task_main", "")).strip()
            for item in successful_records + failed_records
            if str(item.get("task_main", "")).strip()
        ]
        self._apply_operations(relative_tasks, operations, max_rules_num=max_rules_num)


@dataclass
class GMemoryEngine:
    persist_dir: Path
    embedding_model_name: str
    llm_call: Callable[[str, str, float, int], str]
    curator_model_name: str
    hop: int = 1
    start_insights_threshold: int = 5
    rounds_per_insights: int = 5
    insights_point_num: int = 5

    def __post_init__(self):
        self.persist_dir = Path(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.task_store = TaskRecordStore(
            persist_dir=self.persist_dir,
            embedding_model_name=self.embedding_model_name,
        )
        self.task_layer = TaskLayer(
            persist_dir=self.persist_dir,
            task_store=self.task_store,
        )
        self.insights_layer = InsightsManager(
            persist_dir=self.persist_dir,
            llm_call=self.llm_call,
            curator_model_name=self.curator_model_name,
        )
        self.insights_cache: list[str] = []

    @property
    def memory_size(self) -> int:
        return self.task_store.count()

    def _extract_mas_record(self, record: dict) -> dict:
        out = dict(record)
        trajectory = str(out.get("task_trajectory", ""))

        clean_traj = re.sub(r"\d+", "", trajectory)
        out.setdefault("extra_fields", {})
        out["extra_fields"]["clean_traj"] = clean_traj

        extract_user = PROMPTS.extract_true_traj_user_prompt.format(
            task=out.get("task_description", ""),
            trajectory=clean_traj,
        )
        extract_prompt = (
            f"[SYSTEM]\n{PROMPTS.extract_true_traj_system_prompt}\n\n"
            f"[USER]\n{extract_user}"
        )
        key_steps = self.llm_call(self.curator_model_name, extract_prompt, 0.1, 512)
        out["extra_fields"]["key_steps"] = str(key_steps or "").strip()

        if not bool(out.get("label", False)):
            detect_user = PROMPTS.detect_mistakes_user_prompt.format(
                task=out.get("task_description", ""),
                trajectory=clean_traj,
            )
            detect_prompt = (
                f"[SYSTEM]\n{PROMPTS.detect_mistakes_system_prompt}\n\n"
                f"[USER]\n{detect_user}"
            )
            fail_reason = self.llm_call(self.curator_model_name, detect_prompt, 0.0, 256)
            out["extra_fields"]["fail_reason"] = str(fail_reason or "").strip()

        return out

    def add_memory(self, record: dict, max_num_rules: int = 20) -> dict:
        cooked = self._extract_mas_record(record)
        cooked = self.task_store.add_record(cooked)
        self.task_layer.add_task_node(cooked.get("task_main", ""))

        if (
            self.memory_size >= max(1, int(self.start_insights_threshold))
            and self.memory_size % max(1, int(self.rounds_per_insights)) == 0
        ):
            all_records = self.task_store.all_records()
            sample_num = max(1, min(int(self.insights_point_num), len(all_records)))
            for _ in range(sample_num):
                center = random.choice(all_records)
                query_task = str(center.get("task_main", "")).strip()
                if not query_task:
                    continue
                success = [dict(rec) for rec, _ in self.task_store.similarity_search(query_task, k=3, label=True)]
                fail = [dict(rec) for rec, _ in self.task_store.similarity_search(query_task, k=1, label=False)]
                if bool(center.get("label", False)):
                    success.append(dict(center))
                else:
                    fail.append(dict(center))
                self.insights_layer.finetune_insights(success, fail, max_rules_num=max_num_rules)

        if self.memory_size > 0 and self.memory_size % 20 == 0:
            self.insights_layer.merge_insights(max_rules=max_num_rules)

        return cooked

    def _retrieve_memory_raw(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
        insight_windows: int = 10,
        threshold: float = 0.3,
    ) -> tuple[list[dict], list[dict], list[str]]:
        true_tasks_doc: list[tuple[dict, float]] = []
        false_tasks_doc: list[tuple[dict, float]] = []

        related_point_num = max((int(successful_topk) + int(failed_topk)) // 2, 1)
        task_mains = self.task_layer.retrieve_related_task(query_task, node_num=related_point_num, hop=self.hop)

        seen_ids = set()
        for task_main in task_mains:
            cand = self.task_store.similarity_search(task_main, k=1)
            if not cand:
                continue
            rec, _ = cand[0]
            rid = rec.get("record_id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            if bool(rec.get("label", False)):
                true_tasks_doc.append((rec, 0.0))
            else:
                false_tasks_doc.append((rec, 0.0))

        if len(true_tasks_doc) < successful_topk:
            for rec, score in self.task_store.similarity_search(query_task, k=successful_topk, label=True):
                rid = rec.get("record_id")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                true_tasks_doc.append((rec, score))

        if len(false_tasks_doc) < failed_topk:
            for rec, score in self.task_store.similarity_search(query_task, k=failed_topk, label=False):
                rid = rec.get("record_id")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                false_tasks_doc.append((rec, score))

        q_emb = self.task_store._encode([query_task])[0]

        def sort_and_filter(items: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
            rows: list[tuple[dict, float]] = []
            for rec, _ in items:
                emb = _to_float_list(rec.get("embedding"))
                if not emb:
                    emb = self.task_store._encode([str(rec.get("task_main", ""))])[0]
                sim = _cosine(q_emb, emb)
                if sim >= float(threshold):
                    rows.append((rec, sim))
            rows.sort(key=lambda x: x[1], reverse=True)
            return rows

        true_sorted = sort_and_filter(true_tasks_doc)[: max(0, int(successful_topk))]
        false_sorted = sort_and_filter(false_tasks_doc)[: max(0, int(failed_topk))]
        true_records = [rec for rec, _ in true_sorted]
        false_records = [rec for rec, _ in false_sorted]

        insight_scores = self.insights_layer.query_insights_with_score(
            task_mains=task_mains + [query_task],
            top_k=max(1, int(insight_windows)),
        )
        insights = [rule for rule, _ in insight_scores][: max(0, int(insight_windows))]
        return true_records, false_records, insights

    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 2,
        failed_topk: int = 1,
        insight_topk: int = 10,
        threshold: float = 0.3,
    ) -> tuple[list[dict], list[dict], list[str]]:
        succ, fail, insights = self._retrieve_memory_raw(
            query_task=query_task,
            successful_topk=2 * max(0, int(successful_topk)),
            failed_topk=2 * max(0, int(failed_topk)),
            insight_windows=2 * max(0, int(insight_topk)),
            threshold=threshold,
        )

        # Optional LLM rerank on successful trajectories (generative relevance scoring).
        scored_success: list[tuple[float, dict]] = []
        for rec in succ:
            prompt_user = PROMPTS.generative_task_user_prompt.format(
                trajectory=f"{rec.get('task_description', '')}\n{rec.get('task_trajectory', '')}",
                query_scenario=query_task,
            )
            full_prompt = (
                f"[SYSTEM]\n{PROMPTS.generative_task_system_prompt}\n\n"
                f"[USER]\n{prompt_user}"
            )
            raw = self.llm_call(self.curator_model_name, full_prompt, 0.0, 64)
            m = re.search(r"\d+", str(raw or ""))
            score = float(m.group()) if m else 0.0
            scored_success.append((score, rec))
        scored_success.sort(key=lambda x: x[0], reverse=True)

        top_success = [rec for _, rec in scored_success[: max(0, int(successful_topk))]]
        top_fail = fail[: max(0, int(failed_topk))]
        top_insights = insights[: max(0, int(insight_topk))]
        self.insights_cache = list(top_insights)
        return top_success, top_fail, top_insights

    def backward(self, reward: bool) -> None:
        delta = 1.0 if bool(reward) else -2.0
        for insight in self.insights_cache:
            self.insights_layer.backward(insight, delta)
        self.insights_cache = []

    def project_insights(
        self,
        raw_insights: list[str],
        role: str | None,
        task_traj: str | None = None,
    ) -> list[str]:
        if not role:
            return list(raw_insights)
        raw_insights_text = "\n".join(raw_insights)
        if not task_traj:
            user = PROMPTS.project_insights_user_prompt.format(role=role, insights=raw_insights_text)
            full_prompt = (
                f"[SYSTEM]\n{PROMPTS.project_insights_system_prompt}\n\n"
                f"[USER]\n{user}"
            )
        else:
            user = PROMPTS.project_insights_with_traj_user_prompt.format(
                role=role,
                insights=raw_insights_text,
                trajectory=task_traj,
            )
            full_prompt = (
                f"[SYSTEM]\n{PROMPTS.project_insights_with_traj_system_prompt}\n\n"
                f"[USER]\n{user}"
            )
        raw = self.llm_call(self.curator_model_name, full_prompt, 0.0, 256)
        parsed = parse_numbered_list(raw)
        return parsed if parsed else list(raw_insights)
