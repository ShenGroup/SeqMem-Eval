import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


_NOISY_KEYS = {"embedding", "embeddings", "vector", "vectors"}


def _strip_noisy_fields(value):
    """Recursively drop high-volume / non-human-readable fields (e.g. raw
    embedding float arrays) so eval_log stays human-inspectable."""
    if isinstance(value, dict):
        return {
            k: _strip_noisy_fields(v)
            for k, v in value.items()
            if k not in _NOISY_KEYS
        }
    if isinstance(value, list):
        return [_strip_noisy_fields(v) for v in value]
    return value


def _serialize_for_log(value):
    """Best-effort JSON-safe conversion for eval-log payloads."""
    value = _strip_noisy_fields(value)
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except (TypeError, ValueError):
        return str(value)


def append_eval_log(
    output_dir: Path,
    task_name: str,
    method_name: str,
    payload: dict,
    entries_accumulated: list,
) -> None:
    """Append one payload to eval_log_readable.json (indented JSON array,
    rewritten atomically per sample).

    Written from the runner after every sample regardless of update_memory /
    seeded flags so holdout runs always produce per-sample inspection records
    (question, retrieved memory, model output, score).
    """
    entries_accumulated.append(payload)
    readable_path = (
        Path(output_dir) / f"{task_name}_{method_name}_eval_log_readable.json"
    )
    tmp_path = readable_path.with_suffix(readable_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(entries_accumulated, f, ensure_ascii=False, indent=2, default=str)
    tmp_path.replace(readable_path)

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):
        return iterable


def write_live_stats(
    output_dir: Path,
    method_name: str,
    task_name: str,
    total: int,
    success: int,
    timer,
    total_steps_executed: int = 0,
    counted_step_samples: int = 0,
    total_hallucinated: int = 0,
    counted_hallucination_samples: int = 0,
) -> None:
    """Dump current accuracy + timing averages for (method, task) to stats.json.

    Called per-sample so a killed run still has its latest numbers on disk.
    """
    stats = {
        "task": task_name,
        "method": method_name,
        "total": total,
        "success": success,
        "accuracy": (success / total) if total else 0.0,
    }
    if counted_step_samples > 0:
        stats["avg_steps_executed"] = total_steps_executed / counted_step_samples
        stats["total_steps_executed"] = total_steps_executed
        stats["counted_step_samples"] = counted_step_samples
    if counted_hallucination_samples > 0:
        stats["hallucination_rate"] = (
            total_hallucinated / counted_hallucination_samples
        )
        stats["total_hallucinated"] = total_hallucinated
        stats["counted_hallucination_samples"] = counted_hallucination_samples

    if getattr(timer, "enabled", False):
        stage_avg = {}
        per_sample_sum = 0.0
        sample_total_avg = 0.0
        for key, values in timer.records.items():
            if not values:
                continue
            parts = key.split("/", 2)
            if len(parts) != 3:
                continue
            m, t, stage = parts
            if m != method_name or t != task_name:
                continue
            mean = sum(values) / len(values)
            stage_avg[stage] = mean
            if stage in {"retrieve", "generate", "update"}:
                per_sample_sum += mean
            if stage == "sample_total":
                sample_total_avg = mean
        stats["stage_avg_seconds"] = stage_avg
        stats["per_sample_avg_seconds"] = per_sample_sum
        stats["sample_total_avg_seconds"] = sample_total_avg

    stats_path = Path(output_dir) / f"{task_name}_{method_name}_stats.json"
    tmp_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    tmp_path.replace(stats_path)


class Runner:
    def __init__(self, method, tasks, timer, output_dir, max_parallel_samples=1):
        self.method = method
        self.tasks = tasks
        self.timer = timer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel_samples = max(1, int(max_parallel_samples))

    def run(
        self,
        update_memory=True,
        run_first_k=None,
        resume=False,
        memory_seed_file=None,
    ):
        results = []
        for task in self.tasks:
            task_name = task.name
            self.method.reset_task_state(task)
            total = 0
            success = 0
            total_steps_executed = 0
            counted_step_samples = 0
            total_hallucinated = 0
            counted_hallucination_samples = 0
            skip_n = 0
            seeded = False  # True when memory_seed_file was applied for this task
            eval_log_entries: list = []  # accumulator for readable eval_log JSON

            # --- Memory seed: load state from a training run for holdout eval. ---
            # Differs from --resume in that qids from the seed file are NOT added
            # to the "already completed" set, so the holdout loop runs all
            # holdout entries fresh. update_memory is typically False for this
            # flow, and export_task_memory is suppressed so the seed records do
            # not leak into the holdout output dir.
            if memory_seed_file and not resume:
                seed_path = Path(memory_seed_file)
                if not seed_path.exists():
                    raise FileNotFoundError(
                        f"--memory-seed-file not found: {seed_path}"
                    )
                with seed_path.open() as f:
                    seed_records = [json.loads(line) for line in f if line.strip()]
                for rec in seed_records:
                    legacy_memory = rec.pop("memory", None)
                    if legacy_memory and "memory_qids" not in rec:
                        rec["memory_qids"] = [
                            m.get("qid")
                            for m in legacy_memory
                            if isinstance(m, dict)
                        ]
                self.method.memory.extend(seed_records)
                self.method.restore_state_from_memory(task)
                seeded = True
                print(
                    f"[Seed] {task_name}: loaded {len(seed_records)} records "
                    f"from {seed_path}; running holdout fresh (no skip)."
                )

            # --- Resume: reload existing records and skip already-done entries ---
            if resume:
                output_path = self.output_dir / task.memory_filename(self.method.name)
                if output_path.exists():
                    with output_path.open() as f:
                        existing = [json.loads(line) for line in f if line.strip()]
                    if existing:
                        # Legacy format compaction: convert any pre-qid `memory`
                        # snapshot field (O(N) per record) into `memory_qids` so
                        # the next export_task_memory shrinks the jsonl in place.
                        for rec in existing:
                            legacy_memory = rec.pop("memory", None)
                            if legacy_memory and "memory_qids" not in rec:
                                rec["memory_qids"] = [
                                    m.get("qid")
                                    for m in legacy_memory
                                    if isinstance(m, dict)
                                ]
                        self.method.memory.extend(existing)
                        # Some methods (e.g. AWM-Online) emit auxiliary rows
                        # like {"memory_event": {"type": "workflow_induced"}}
                        # that are NOT processed samples. Count only primary
                        # sample rows so skip_n matches the runner's per-sample
                        # outer loop iterations.
                        def _is_sample_row(r):
                            ev = r.get("memory_event")
                            if not isinstance(ev, dict):
                                return True
                            t = ev.get("type")
                            if t is None:
                                return True
                            return t == "sample_observed"

                        sample_rows = [r for r in existing if _is_sample_row(r)]
                        skip_n = len(sample_rows)
                        for rec in sample_rows:
                            total += 1
                            s = rec.get("score", 0.0)
                            try:
                                success += int(float(s) >= 1.0)
                            except (TypeError, ValueError):
                                pass
                        self.method.restore_state_from_memory(task)
                        print(
                            f"[Resume] {task_name}: loaded {skip_n} records, "
                            f"accuracy so far {success}/{total} = "
                            f"{(success / total if total else 0):.3f}"
                        )

            limit = None
            if run_first_k is not None:
                limit = max(0, int(run_first_k))

            expected_total = task.total_entries()
            if limit is not None and expected_total is not None:
                expected_total = min(expected_total, limit)
            elif limit is not None and expected_total is None:
                expected_total = limit
            progress = tqdm(
                task.iter_entries(),
                desc=f"{self.method.name}/{task_name}",
                unit="sample",
                total=expected_total,
                initial=skip_n,
                dynamic_ncols=True,
            )
            if not TQDM_AVAILABLE:
                print(
                    f"[Progress disabled] Install tqdm to show progress bar: "
                    f"{self.method.name}/{task_name}"
                )
            # --- Intra-task sample-level concurrency (opt-in) -----------------
            # Methods that opt in via `supports_parallel_samples = True`
            # (Baseline only, today) can run their per-sample retrieve+generate
            # in a thread pool. Interactive tasks (ALFWorld → has `step_episode`)
            # are forced sequential because their per-task env state is not
            # thread-safe. Evaluation, memory updates, eval-log writes, and
            # stats writes always run on the main thread in submission order
            # to preserve `task._last_eval_info` semantics and per-sample
            # determinism on disk.
            method_opt_in = bool(getattr(self.method, "supports_parallel_samples", False))
            task_is_interactive = hasattr(task, "step_episode")
            use_parallel = (
                self.max_parallel_samples > 1
                and method_opt_in
                and not task_is_interactive
            )
            iter_state = {
                "total": total,
                "success": success,
                "total_steps_executed": total_steps_executed,
                "counted_step_samples": counted_step_samples,
                "total_hallucinated": total_hallucinated,
                "counted_hallucination_samples": counted_hallucination_samples,
                "skipped": 0,
            }
            if use_parallel:
                self._run_entries_parallel(
                    task, task_name, progress, skip_n, limit,
                    update_memory, seeded, eval_log_entries, iter_state,
                )
            else:
                self._run_entries_sequential(
                    task, task_name, progress, skip_n, limit,
                    update_memory, seeded, eval_log_entries, iter_state,
                )
            total = iter_state["total"]
            success = iter_state["success"]
            total_steps_executed = iter_state["total_steps_executed"]
            counted_step_samples = iter_state["counted_step_samples"]
            total_hallucinated = iter_state["total_hallucinated"]
            counted_hallucination_samples = iter_state["counted_hallucination_samples"]

            accuracy = (success / total) if total else 0.0
            self.method.finalize_task_state(task)
            results.append(
                {
                    "task": task_name,
                    "method": self.method.name,
                    "total": total,
                    "success": success,
                    "accuracy": accuracy,
                    "avg_steps_executed": (
                        (total_steps_executed / counted_step_samples)
                        if counted_step_samples > 0
                        else None
                    ),
                    "total_steps_executed": total_steps_executed,
                    "counted_step_samples": counted_step_samples,
                    "hallucination_rate": (
                        (total_hallucinated / counted_hallucination_samples)
                        if counted_hallucination_samples > 0
                        else None
                    ),
                    "total_hallucinated": total_hallucinated,
                    "counted_hallucination_samples": counted_hallucination_samples,
                }
            )

            if not seeded and update_memory:
                output_path = self.output_dir / task.memory_filename(self.method.name)
                self.method.export_task_memory(task_name, output_path)
        return results

    # ------------------------------------------------------------------
    # Per-sample helpers used by both sequential and parallel paths.
    # ------------------------------------------------------------------
    def _finalize_sample(
        self,
        *,
        task,
        task_name,
        entry,
        score,
        retrieved_memory_for_log,
        model_output_for_log,
        question_for_log,
        last_eval_info,
        token_totals,
        seeded,
        update_memory,
        eval_log_entries,
        iter_state,
    ):
        """Common post-generation bookkeeping: update counters, write eval-log,
        and persist live stats. Always runs on the main thread."""
        iter_state["success"] += int(score >= 1.0)
        if isinstance(last_eval_info, dict) and "steps_executed" in last_eval_info:
            try:
                step_count = int(last_eval_info.get("steps_executed", 0))
                iter_state["total_steps_executed"] += max(0, step_count)
                iter_state["counted_step_samples"] += 1
            except Exception:
                pass
        if isinstance(last_eval_info, dict) and "hallucinated" in last_eval_info:
            try:
                iter_state["total_hallucinated"] += int(
                    bool(last_eval_info.get("hallucinated", 0))
                )
                iter_state["counted_hallucination_samples"] += 1
            except Exception:
                pass

        qid = entry.get("qid") if isinstance(entry, dict) else None
        eval_payload = {
            "qid": qid if qid is not None else iter_state["total"],
            "task": task_name,
            "method": self.method.name,
            "question": _serialize_for_log(question_for_log),
            "retrieved_memory": _serialize_for_log(retrieved_memory_for_log),
            "model_output": _serialize_for_log(model_output_for_log),
            "score": score,
            "correct": bool(score >= 1.0) if score is not None else None,
            "reasoning_tokens_total":  token_totals.get("reasoning_tokens_total", 0),
            "completion_tokens_total": token_totals.get("completion_tokens_total", 0),
        }
        try:
            extra_ctx = self.method.export_eval_context(
                task, entry, retrieved_memory_for_log
            ) or {}
        except Exception as exc:
            extra_ctx = {"_export_eval_context_error": str(exc)}
        if extra_ctx:
            eval_payload["extra_context"] = _serialize_for_log(extra_ctx)
        append_eval_log(
            self.output_dir, task_name, self.method.name,
            eval_payload, eval_log_entries,
        )

        if not seeded and update_memory:
            output_path = self.output_dir / task.memory_filename(self.method.name)
            self.method.export_task_memory(task_name, output_path)
        write_live_stats(
            self.output_dir,
            self.method.name,
            task_name,
            iter_state["total"],
            iter_state["success"],
            self.timer,
            iter_state["total_steps_executed"],
            iter_state["counted_step_samples"],
            iter_state["total_hallucinated"],
            iter_state["counted_hallucination_samples"],
        )

    def _run_entries_sequential(
        self, task, task_name, progress, skip_n, limit,
        update_memory, seeded, eval_log_entries, iter_state,
    ):
        from utils.llm_client import (
            reset_call_accumulators, get_call_totals,
        )

        for entry in progress:
            if iter_state["skipped"] < skip_n:
                iter_state["skipped"] += 1
                continue
            if limit is not None and iter_state["total"] >= limit:
                break
            with self.timer.track(f"{self.method.name}/{task_name}/sample_total"):
                iter_state["total"] += 1
                inputs = task.build_inputs(entry)
                try:
                    reset_call_accumulators()
                except Exception:
                    pass

                retrieved_memory_for_log = None
                model_output_for_log = None
                question_for_log = None

                score = self.method.run_trial(
                    task=task, entry=entry, inputs=inputs,
                    timer=self.timer, update_memory=update_memory,
                )
                if score is None:
                    prompt = task.build_prompt(entry)
                    query = task.get_query(entry, inputs)
                    question_for_log = query
                    with self.timer.track(f"{self.method.name}/{task_name}/retrieve"):
                        memory = self.method.retrieve_memory(
                            task, query=query, entry=entry, **inputs
                        )
                    retrieved_memory_for_log = memory
                    with self.timer.track(f"{self.method.name}/{task_name}/generate"):
                        output = self.method.generate(
                            task, prompt, memory=memory, entry=entry, **inputs
                        )
                    model_output_for_log = output
                    score = task.evaluate_entry(output, entry)
                    feedback = "success" if score >= 1.0 else "failure"
                    if update_memory:
                        record = task.build_memory_record(
                            entry, output, feedback, score
                        )
                        with self.timer.track(f"{self.method.name}/{task_name}/update"):
                            self.method.update_memory(
                                task, output, record=record, entry=entry,
                                memory=memory, **inputs
                            )
                last_eval_info = getattr(task, "_last_eval_info", None)
                try:
                    token_totals = get_call_totals()
                except Exception:
                    token_totals = {}
                self._finalize_sample(
                    task=task, task_name=task_name, entry=entry, score=score,
                    retrieved_memory_for_log=retrieved_memory_for_log,
                    model_output_for_log=model_output_for_log,
                    question_for_log=question_for_log,
                    last_eval_info=last_eval_info,
                    token_totals=token_totals,
                    seeded=seeded, update_memory=update_memory,
                    eval_log_entries=eval_log_entries, iter_state=iter_state,
                )

    def _run_entries_parallel(
        self, task, task_name, progress, skip_n, limit,
        update_memory, seeded, eval_log_entries, iter_state,
    ):
        """Parallel per-sample dispatch for opt-in stateless methods on
        non-interactive tasks. Workers run retrieve+generate concurrently;
        the main thread evaluates, updates memory (if enabled), and writes
        the eval-log + stats in submission order."""
        from utils.llm_client import (
            reset_call_accumulators, get_call_totals,
        )

        # Materialize the entry list (post skip / pre limit) so we can submit
        # all jobs to the pool at once. For our six tasks the entry counts are
        # small (<= ~1170) and entries are plain dicts.
        entries: list = []
        for entry in progress:
            if iter_state["skipped"] < skip_n:
                iter_state["skipped"] += 1
                continue
            if limit is not None and len(entries) >= limit:
                break
            entries.append(entry)
        if not entries:
            return

        def _worker(entry):
            t_sample_start = time.perf_counter()
            inputs = task.build_inputs(entry)
            try:
                reset_call_accumulators()
            except Exception:
                pass
            prompt = task.build_prompt(entry)
            query = task.get_query(entry, inputs)
            t0 = time.perf_counter()
            memory = self.method.retrieve_memory(
                task, query=query, entry=entry, **inputs
            )
            t_retrieve = time.perf_counter() - t0
            t1 = time.perf_counter()
            output = self.method.generate(
                task, prompt, memory=memory, entry=entry, **inputs
            )
            t_generate = time.perf_counter() - t1
            try:
                token_totals = get_call_totals()
            except Exception:
                token_totals = {}
            return {
                "entry": entry,
                "inputs": inputs,
                "prompt": prompt,
                "query": query,
                "memory": memory,
                "output": output,
                "token_totals": token_totals,
                "t_retrieve": t_retrieve,
                "t_generate": t_generate,
                "t_sample_total": time.perf_counter() - t_sample_start,
            }

        with ThreadPoolExecutor(max_workers=self.max_parallel_samples) as ex:
            futures = [ex.submit(_worker, e) for e in entries]
            for fut in futures:
                res = fut.result()
                iter_state["total"] += 1
                # Inject worker timings into the (non-thread-safe) Timer
                # from the main thread so aggregation stays consistent with
                # the sequential path.
                if self.timer.enabled:
                    self.timer.records.setdefault(
                        f"{self.method.name}/{task_name}/retrieve", []
                    ).append(res["t_retrieve"])
                    self.timer.records.setdefault(
                        f"{self.method.name}/{task_name}/generate", []
                    ).append(res["t_generate"])
                # Evaluate on the main thread so `task._last_eval_info` is safe.
                eval_start = time.perf_counter()
                score = task.evaluate_entry(res["output"], res["entry"])
                last_eval_info = getattr(task, "_last_eval_info", None)
                if update_memory:
                    feedback = "success" if score >= 1.0 else "failure"
                    record = task.build_memory_record(
                        res["entry"], res["output"], feedback, score
                    )
                    t_upd = time.perf_counter()
                    self.method.update_memory(
                        task, res["output"], record=record, entry=res["entry"],
                        memory=res["memory"], **res["inputs"]
                    )
                    if self.timer.enabled:
                        self.timer.records.setdefault(
                            f"{self.method.name}/{task_name}/update", []
                        ).append(time.perf_counter() - t_upd)
                if self.timer.enabled:
                    sample_total = (
                        res["t_sample_total"]
                        + (time.perf_counter() - eval_start)
                    )
                    self.timer.records.setdefault(
                        f"{self.method.name}/{task_name}/sample_total", []
                    ).append(sample_total)
                self._finalize_sample(
                    task=task, task_name=task_name, entry=res["entry"],
                    score=score,
                    retrieved_memory_for_log=res["memory"],
                    model_output_for_log=res["output"],
                    question_for_log=res["query"],
                    last_eval_info=last_eval_info,
                    token_totals=res["token_totals"],
                    seeded=seeded, update_memory=update_memory,
                    eval_log_entries=eval_log_entries, iter_state=iter_state,
                )
