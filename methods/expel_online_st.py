import math
import re
from pathlib import Path

from methods.base_method import BaseMethod
from utils.llm_client import generate as llm_generate

try:
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except ImportError:
    transformers_logging = None


class ExpelOnlineST(BaseMethod):
    # Explicitly mark this method as the online single-try variant.
    name = "ExpeL-Online-ST"
    _PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts" / "expel" / "alfworld"
    _SHARED_ALFWORLD_SYSTEM_PATH = (
        Path(__file__).resolve().parents[1] / "prompts" / "alfworld" / "system_instruction.txt"
    )
    _DEFAULT_SYSTEM_INSTRUCTION_PATH = (
        _SHARED_ALFWORLD_SYSTEM_PATH
        if _SHARED_ALFWORLD_SYSTEM_PATH.is_file()
        else _PROMPT_ROOT / "system_instruction.txt"
    )
    _DEFAULT_HUMAN_INSTRUCTION_TEMPLATE_PATH = _PROMPT_ROOT / "human_instruction_template.txt"
    _DEFAULT_TASK_TEMPLATE_PATH = _PROMPT_ROOT / "task_template.txt"
    _DEFAULT_RULE_TEMPLATE_PATH = _PROMPT_ROOT / "rule_template.txt"

    _RETRIEVED_TRAJ_PREFIX = (
        "Additional retrieved successful trajectories from prior tasks "
        "(for reference only):"
    )

    def __init__(
        self,
        top_k=3,
        batch_update_size=8,
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        curator_model_name=None,
        temperature=0.0,
        max_new_tokens=1024,
        insights_init="(empty)",
        max_num_rules=20,
        **kwargs,
    ):
        super().__init__()
        self.top_k = max(0, int(top_k))
        self.batch_update_size = max(1, int(batch_update_size))
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.curator_model_name = curator_model_name or generation_model_name
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.insights_init = str(insights_init or "(empty)").strip() or "(empty)"
        self.max_num_rules = max(1, int(max_num_rules))

        self._embedder = None
        self._task_rule_items_with_count = {}
        self._task_recent_success = {}
        self._task_experience_pool = {}
        self._task_experience_vectors = {}
        self._task_pending_rollout = {}
        self._task_seeded = {}
        self._alfworld_system_instruction = self._read_prompt_text(self._DEFAULT_SYSTEM_INSTRUCTION_PATH)
        self._alfworld_human_instruction_template = self._read_prompt_text(
            self._DEFAULT_HUMAN_INSTRUCTION_TEMPLATE_PATH
        )
        self._alfworld_task_template = self._read_prompt_text(self._DEFAULT_TASK_TEMPLATE_PATH)
        self._alfworld_rule_template = self._read_prompt_text(self._DEFAULT_RULE_TEMPLATE_PATH)

    def _read_prompt_text(self, path):
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise RuntimeError(f"ExpeL-Online-ST failed to read prompt file: {path}") from exc

    def _supports_interactive_task(self, task):
        return all(hasattr(task, name) for name in ("start_episode", "step_episode", "finish_episode"))

    def _build_step_prompt_text(self, task_prompt, current_observation, step_rows,
                                initial_observation=None):
        lines = [f"{task_prompt}"]
        init_obs = str(initial_observation or "").strip()
        if init_obs:
            lines.append(init_obs)
        current_obs = str(current_observation or "").strip()
        if current_obs and current_obs != init_obs:
            lines.append(current_obs)
        for row in step_rows[-12:]:
            action = str(row.get("action", "")).strip()
            observation = str(row.get("observation", "")).strip()
            if action:
                lines.append(f"Action: {action}")
            if observation:
                lines.append(observation)
        return "\n".join(lines).strip()

    def _extract_single_action(self, output, task=None):
        text = str(output or "").strip()
        if not text:
            return ""
        if task is not None and hasattr(task, "_extract_actions"):
            try:
                actions = list(task._extract_actions(text))
            except Exception:
                actions = []
            if actions:
                return str(actions[0]).strip()
        trajectory = self._extract_tag_content(text, "trajectory")
        source = trajectory if trajectory else text
        for raw in source.splitlines():
            line = str(raw).strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("> think") or low.startswith("think:") or low.startswith("thought:"):
                continue
            if line.startswith(">"):
                line = re.sub(r"^>\s*", "", line).strip()
            line = re.sub(r"^\d+\.\s*", "", line).strip()
            if line.lower().startswith("action:"):
                line = line.split(":", 1)[1].strip()
            if not line:
                continue
            if line.lower().startswith("thought:"):
                continue
            if re.match(
                r"^(go to|open|close|take|put|move|use|heat|cool|look|clean|inventory|examine|drop|slice)\b",
                line,
                flags=re.IGNORECASE,
            ):
                return line
        return ""

    def _build_trial_model_output(self, step_rows, success):
        lines = []
        for row in step_rows:
            action = str(row.get("action", "")).strip()
            observation = str(row.get("observation", "")).strip()
            if action:
                lines.append(f"Action: {action}")
            if observation:
                lines.append(observation)
        return "\n".join(lines).strip()

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from utils.embedder import get_embedder
        except ImportError as exc:
            raise RuntimeError(
                "ExpeL-Online-ST requires sentence-transformers. "
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
        return [list(vec) for vec in vectors]

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _extract_tag_content(self, text, tag_name):
        if not text:
            return ""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"
        if start_tag not in text or end_tag not in text:
            return ""
        try:
            return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
        except Exception:
            return ""

    def _normalize_rule_text(self, text):
        clean = str(text or "").strip().strip("- ").strip()
        if not clean:
            return ""
        clean = re.sub(r"\s+", " ", clean)
        if not clean.endswith("."):
            clean = f"{clean}."
        return clean

    def _current_rule_items(self, task_name):
        return self._task_rule_items_with_count.get(task_name, [])

    def export_eval_context(self, task, entry, memory):
        return {
            "insights_rendered": self._current_insights(task.name),
            "rules_count": len(self._current_rule_items(task.name)),
        }

    def _current_insights(self, task_name):
        items = self._current_rule_items(task_name)
        if not items:
            return self.insights_init
        return "\n".join(f"{idx}. {rule}" for idx, (rule, _) in enumerate(items, start=1))

    def _build_retrieved_cases_block(self, retrieved):
        if not retrieved:
            return "(empty)"
        lines = [
            "### RETRIEVED SUCCESS CASES (START)",
            "These are previous successful cases. Use for guidance, not copy.",
            "",
        ]
        for idx, item in enumerate(retrieved, start=1):
            score = item.get("retrieval_score")
            score_text = f" (Similarity: {score:.2f})" if isinstance(score, float) else ""
            lines.append(f"[Case {idx}]{score_text}")
            lines.append(f"Question: {item.get('question', '')}")
            lines.append(f"Trajectory: {item.get('trajectory', item.get('model_output', ''))}")
            lines.append("---")
        lines.append("### RETRIEVED SUCCESS CASES (END)")
        return "\n".join(lines).strip()

    def _build_manual_fewshots_block(self, manual_fewshots, limit=1):
        shots = [str(s).strip() for s in (manual_fewshots or []) if str(s).strip()]
        if not shots:
            return "(none)"
        return "\n\n".join(shots[: max(1, int(limit))]).strip()

    def _build_fewshot_examples_block(self, retrieved):
        if not retrieved:
            return "(none)"
        lines = []
        for idx, item in enumerate(retrieved, start=1):
            score = item.get("retrieval_score")
            score_text = f" (Similarity: {score:.2f})" if isinstance(score, float) else ""
            lines.append(f"### Example {idx}{score_text}")
            lines.append(f"Task: {item.get('question', '')}")
            lines.append(f"Trajectory:\n{item.get('trajectory', item.get('model_output', ''))}")
            lines.append("")
        return "\n".join(lines).strip()

    def _build_solver_prompt(
        self,
        prompt,
        insights,
        retrieved_cases,
        action_mode=False,
        max_steps=None,
        manual_fewshots=None,
    ):
        fewshots_block = self._build_fewshot_examples_block(retrieved_cases)
        if action_mode:
            max_steps_txt = int(max_steps) if max_steps is not None else 30
            manual_block = self._build_manual_fewshots_block(manual_fewshots, limit=1)
            human_instruction = self._alfworld_human_instruction_template.format(
                instruction=f"{self._alfworld_system_instruction}\n\n",
                max_steps=max_steps_txt,
            )
            task_block = self._alfworld_task_template.format(task=prompt)
            rules_block = self._alfworld_rule_template.format(rules=insights)
            return (
                f"{human_instruction}\n"
                f"{manual_block}\n\n"
                "(END OF EXAMPLES)\n\n"
                f"{self._RETRIEVED_TRAJ_PREFIX}\n"
                f"{fewshots_block}\n\n"
                f"{rules_block}\n\n"
                f"{task_block}\n"
            )
        return (
            "# ROLE\n"
            "You are an advanced reasoning agent following a compact rulebook.\n\n"
            "# OBJECTIVE\n"
            "Solve the current problem correctly in one attempt.\n"
            "Use retrieved successful trajectories as references.\n\n"
            "# RULEBOOK (EXISTING INSIGHTS)\n"
            f"{insights}\n\n"
            "# FEWSHOT EXAMPLES (DYNAMIC, RETRIEVED FROM SUCCESS POOL)\n"
            f"{fewshots_block}\n\n"
            "# REASONING TRACE (OPTIONAL, FOR METHOD-INTERNAL BOOKKEEPING)\n"
            "You may wrap your reasoning in <trajectory>...</trajectory>. "
            "Follow the output format specified by the task below for the final answer.\n\n"
            "# CURRENT PROBLEM\n"
            f"{prompt}\n"
        )

    def _parse_fewshot_seed(self, shot_text):
        text = str(shot_text or "").strip()
        if not text:
            return None
        lines = [ln.rstrip() for ln in text.splitlines()]
        if not lines:
            return None
        question = ""
        trajectory_lines = []
        for ln in lines:
            clean = ln.strip()
            if not clean:
                continue
            if not question:
                question = clean
                continue
            trajectory_lines.append(clean)
        trajectory = "\n".join(trajectory_lines).strip() or text
        return {
            "question": question or "fewshot_seed",
            "trajectory": trajectory,
            "model_output": trajectory,
            "score": 1.0,
            "seed_demo": True,
        }

    def _seed_experience_pool_from_manual_fewshots(self, task_name, manual_fewshots):
        if self._task_seeded.get(task_name, False):
            return
        for shot in (manual_fewshots or []):
            parsed = self._parse_fewshot_seed(shot)
            if not parsed:
                continue
            self._append_success_case(task_name, parsed)
        self._task_seeded[task_name] = True

    def _build_operation_format_instruction(self):
        return (
            "<OPERATION> <RULE NUMBER>: <RULE>\n\n"
            "Allowed operations:\n"
            "- AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\n"
            "- REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\n"
            "- EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\n"
            "- ADD <NEW RULE NUMBER>: <NEW RULE>\n\n"
            "Constraints:\n"
            "- Return at most 4 operations.\n"
            "- Each operation on a new line.\n"
            "- Only output operations, no extra commentary.\n"
            "- Rules should be general, concise, reusable.\n"
        )

    def _build_batch_update_prompt(self, current_insights, recent_success):
        lines = [
            "# RULE CURATOR (ALL SUCCESS BATCH)",
            "",
            "Update rules based on successful trajectories only.",
            "",
            "## EXISTING RULES",
            str(current_insights),
            "",
            "## SUCCESSFUL TRAJECTORIES",
        ]
        for idx, item in enumerate(recent_success, start=1):
            lines.extend(
                [
                    f"[Success {idx}]",
                    f"Question: {item.get('question', '')}",
                    f"Trajectory: {item.get('trajectory', '')}",
                    "",
                ]
            )
        lines.extend(
            [
                "Apply rule operations in this format:",
                self._build_operation_format_instruction(),
            ]
        )
        return "\n".join(lines).strip()

    def _parse_rules_ops(self, text):
        pattern = r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+)?):\s*(.*)"
        matches = re.findall(pattern, str(text or ""))
        operations = []
        for operation, raw_rule in matches:
            cleaned = self._normalize_rule_text(raw_rule)
            if not cleaned:
                continue
            operation_head = operation.strip()
            op_type = operation_head.split(" ")[0]
            if op_type not in {"REMOVE", "EDIT", "ADD", "AGREE"}:
                continue
            operations.append((operation_head, cleaned))
        return operations

    def _retrieve_rule_index(self, rules, operation_rule_text):
        for i, (rule, _) in enumerate(rules):
            if rule in operation_rule_text or operation_rule_text in rule:
                return i
        return None

    def _is_existing_rule(self, rules, operation_rule_text):
        return self._retrieve_rule_index(rules, operation_rule_text) is not None

    def _apply_rule_operations(self, task_name, operations):
        if not operations:
            return False
        rules = list(self._current_rule_items(task_name))
        changed = False

        # Validate and normalize operations to avoid invalid edits.
        filtered = []
        for operation, operation_rule_text in operations:
            op_type = operation.split(" ")[0]
            rule_num = int(operation.split(" ")[1]) if " " in operation else None
            if op_type == "ADD":
                if self._is_existing_rule(rules, operation_rule_text):
                    continue
                filtered.append((operation, operation_rule_text))
                continue

            if op_type == "EDIT":
                if self._is_existing_rule(rules, operation_rule_text):
                    matched = self._retrieve_rule_index(rules, operation_rule_text)
                    if matched is None:
                        continue
                    filtered.append((f"AGREE {matched + 1}", rules[matched][0]))
                    continue
                if rule_num is None or rule_num < 1 or rule_num > len(rules):
                    continue
                filtered.append((operation, operation_rule_text))
                continue

            if op_type in {"REMOVE", "AGREE"}:
                if not self._is_existing_rule(rules, operation_rule_text):
                    continue
                filtered.append((operation, operation_rule_text))

        for op in ["REMOVE", "AGREE", "EDIT", "ADD"]:
            for operation, operation_rule_text in filtered:
                op_type = operation.split(" ")[0]
                if op_type != op:
                    continue
                if op_type == "REMOVE":
                    rule_index = self._retrieve_rule_index(rules, operation_rule_text)
                    if rule_index is None:
                        continue
                    remove_strength = 3 if len(rules) >= self.max_num_rules else 1
                    rules[rule_index] = (rules[rule_index][0], rules[rule_index][1] - remove_strength)
                    changed = True
                elif op_type == "AGREE":
                    rule_index = self._retrieve_rule_index(rules, operation_rule_text)
                    if rule_index is None:
                        continue
                    rules[rule_index] = (rules[rule_index][0], rules[rule_index][1] + 1)
                    changed = True
                elif op_type == "EDIT":
                    rule_index = int(operation.split(" ")[1]) - 1
                    if rule_index < 0 or rule_index >= len(rules):
                        continue
                    rules[rule_index] = (operation_rule_text, rules[rule_index][1] + 1)
                    changed = True
                elif op_type == "ADD":
                    rules.append((operation_rule_text, 2))
                    changed = True

        # Drop dead rules and keep sorted by importance.
        rules = [r for r in rules if r[1] > 0]
        rules.sort(key=lambda x: x[1], reverse=True)
        if len(rules) > self.max_num_rules:
            rules = rules[: self.max_num_rules]
        self._task_rule_items_with_count[task_name] = rules
        return changed

    def _update_rules_from_batch(self, task_name, batch_success):
        prompt = self._build_batch_update_prompt(
            current_insights=self._current_insights(task_name),
            recent_success=batch_success,
        )
        raw = llm_generate(
            model_name=self.curator_model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=max(self.max_new_tokens, 512),
        )
        return self._apply_rule_operations(task_name, self._parse_rules_ops(raw))

    def _append_success_case(self, task_name, success_case):
        self._task_experience_pool.setdefault(task_name, []).append(success_case)
        question = str(success_case.get("question", "")).strip()
        self._task_experience_vectors.setdefault(task_name, []).extend(self._encode([question]))

    def retrieve_memory(self, task, **kwargs):
        query = str(kwargs.get("query", "")).strip()
        task_name = task.name
        self._seed_experience_pool_from_manual_fewshots(
            task_name=task_name,
            manual_fewshots=kwargs.get("fewshots", []),
        )
        pool = self._task_experience_pool.get(task_name, [])
        vectors = self._task_experience_vectors.get(task_name, [])
        if self.top_k <= 0 or not pool or not vectors:
            return []
        if not query:
            import os as _os
            if _os.environ.get("BWT_DISABLE_RECENCY_FALLBACK"):
                return []
            return [
                {
                    "question": c.get("question", ""),
                    "model_output": c.get("model_output", ""),
                    "trajectory": c.get("trajectory", ""),
                    "retrieval_score": None,
                }
                for c in pool[-self.top_k :]
            ]
        query_vec = self._encode([query])[0]
        similarities = [self._cosine(query_vec, vec) for vec in vectors]
        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )[: self.top_k]
        return [
            {
                "question": pool[i].get("question", ""),
                "model_output": pool[i].get("model_output", ""),
                "trajectory": pool[i].get("trajectory", ""),
                "retrieval_score": float(similarities[i]),
            }
            for i in top_indices
        ]

    def run_trial(self, task, entry, inputs, timer, update_memory=True):
        if not self._supports_interactive_task(task):
            return None

        task_name = task.name
        prompt = task.build_prompt(entry)
        query = task.get_query(entry, inputs)

        with timer.track(f"{self.name}/{task_name}/retrieve"):
            memory = self.retrieve_memory(task, query=query, entry=entry, **inputs)

        insights_before = self._current_insights(task_name)
        step_rows = []
        current_observation = task.start_episode(entry)
        initial_observation = current_observation
        max_steps = max(1, int(getattr(task, "max_steps", 30)))

        for step_idx in range(max_steps):
            step_prompt = self._build_step_prompt_text(
                prompt, current_observation, step_rows,
                initial_observation=initial_observation,
            )
            generation_prompt = self._build_solver_prompt(
                prompt=step_prompt,
                insights=insights_before,
                retrieved_cases=memory,
                action_mode=True,
                max_steps=max_steps,
                manual_fewshots=inputs.get("fewshots", []),
            )
            with timer.track(f"{self.name}/{task_name}/generate"):
                raw_output = llm_generate(
                    model_name=self.generation_model_name,
                    prompt=generation_prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                )
            action = self._extract_single_action(raw_output, task=task)
            if not action:
                break
            step_result = task.step_episode(
                action,
                raw_action=action,
                meta={"agent_name": "expel_st"},
            )
            row = {
                "step": step_idx + 1,
                "action": str(step_result.get("executed_action", action)),
                "observation": str(step_result.get("raw_observation", step_result.get("observation", ""))),
                "reward": str(step_result.get("reward", "")),
                "done": bool(step_result.get("done", False)),
                "won": bool(step_result.get("won", False)),
            }
            step_rows.append(row)
            current_observation = str(step_result.get("observation", ""))
            if bool(step_result.get("done", False)):
                break

        final = task.finish_episode()
        score = float(final.get("score", 0.0))
        feedback = "success" if score >= 1.0 else "failure"
        model_output = self._build_trial_model_output(step_rows, success=(score >= 1.0))
        record = task.build_memory_record(entry, model_output, feedback, score)

        self._task_pending_rollout[task_name] = {
            "question": query,
            "retrieved_cases": memory,
            "trajectory": model_output,
            "raw_output": model_output,
            "insights_before": insights_before,
            "rollout_type": "online_single_try_interactive",
        }

        if update_memory:
            with timer.track(f"{self.name}/{task_name}/update"):
                self.update_memory(task, model_output, record=record, entry=entry, **inputs)
        return score

    def generate(self, task, prompt, **kwargs):
        retrieved = kwargs.get("memory", [])
        task_name = task.name
        insights_before = self._current_insights(task_name)
        action_mode = bool(getattr(task, "action_mode", False))
        generation_prompt = self._build_solver_prompt(
            prompt=prompt,
            insights=insights_before,
            retrieved_cases=retrieved,
            action_mode=action_mode,
            max_steps=getattr(task, "max_steps", 30),
            manual_fewshots=kwargs.get("fewshots", []),
        )
        raw_output = llm_generate(
            model_name=self.generation_model_name,
            prompt=generation_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        trajectory = self._extract_tag_content(raw_output, "trajectory")
        if not trajectory:
            trajectory = str(raw_output).strip()
        self._task_pending_rollout[task_name] = {
            "question": str(kwargs.get("query", "")).strip(),
            "retrieved_cases": retrieved,
            "trajectory": trajectory,
            "raw_output": str(raw_output),
            "insights_before": insights_before,
            "rollout_type": "online_single_try",
        }
        return str(raw_output)

    def update_memory(self, task, output, **kwargs):
        record = kwargs.get("record")
        if not record:
            return

        task_name = task.name
        rollout = self._task_pending_rollout.pop(task_name, {})
        question = str(record.get("question", rollout.get("question", ""))).strip()
        final_output = str(record.get("model_output", output))
        trajectory = str(rollout.get("trajectory", final_output))
        is_success = str(record.get("feedback", "")).lower() == "success"
        insights_before = rollout.get("insights_before", self._current_insights(task_name))
        batch_updated = False

        if is_success:
            success_case = {
                "question": question,
                "trajectory": trajectory,
                "model_output": trajectory,
                "score": record.get("score"),
            }
            self._append_success_case(task_name, success_case)
            self._task_recent_success.setdefault(task_name, []).append(success_case)

            recent_success = self._task_recent_success.get(task_name, [])
            if len(recent_success) >= self.batch_update_size:
                batch_updated = self._update_rules_from_batch(
                    task_name=task_name,
                    batch_success=recent_success,
                )
                self._task_recent_success[task_name] = []

        insights_after = self._current_insights(task_name)
        enriched = {
            "task": task_name,
            **record,
            "memory": {
                "insights": insights_after,
                "retrieved_success_cases": rollout.get("retrieved_cases", []),
                "rollout": {
                    "max_tries": 1,
                    "attempts": [
                        {
                            "try": 1,
                            "trajectory": trajectory,
                        }
                    ],
                    "reflections": [],
                },
            },
            "memory_meta": {
                "method_variant": "ExpeL-Online-ST",
                "rollout_type": rollout.get("rollout_type", "online_single_try"),
                "insights_before": insights_before,
                "insights_after": insights_after,
                "insights_pair_updated": False,
                "insights_batch_updated": batch_updated,
                "rules_count": len(self._current_rule_items(task_name)),
                "experience_pool_size": len(self._task_experience_pool.get(task_name, [])),
                "recent_success_size": len(self._task_recent_success.get(task_name, [])),
            },
        }
        self.memory.append(enriched)

    def reset_task_state(self, task):
        task_name = task.name
        self._task_rule_items_with_count.setdefault(task_name, [])
        self._task_recent_success.setdefault(task_name, [])
        self._task_experience_pool.setdefault(task_name, [])
        self._task_experience_vectors.setdefault(task_name, [])
        self._task_seeded.setdefault(task_name, False)
        return None

    def restore_state_from_memory(self, task):
        """Rebuild rule library + experience pool from `self.memory` records.

        See ExpeL-Online-MT for the rationale — without this the holdout
        harness would start ExpeL with empty rules/pool, producing an
        incomparable prompt at the seed checkpoint.
        """
        task_name = task.name
        records = [r for r in self.memory if r.get("task") == task_name]
        if not records:
            return

        # Rules: parse last record's memory_meta.insights_after (rendered text).
        last = records[-1]
        meta = last.get("memory_meta") or {}
        insights_after = meta.get("insights_after")
        parsed_rules: list[tuple[str, int]] = []
        if isinstance(insights_after, str) and insights_after.strip():
            for line in insights_after.splitlines():
                m = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
                if not m:
                    continue
                text = self._normalize_rule_text(m.group(1))
                if text:
                    parsed_rules.append((text, 1))
        self._task_rule_items_with_count[task_name] = parsed_rules

        # Experience pool: replay success cases from score>=1 records.
        for rec in records:
            try:
                score = float(rec.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < 1.0:
                continue
            mem = rec.get("memory") or {}
            rollout = mem.get("rollout") or {}
            attempts = rollout.get("attempts") or []
            question = str(rec.get("question", "")).strip()
            final_output = str(rec.get("model_output", ""))
            # ST records a single attempt row whose trajectory is the
            # full trial output.
            trajectory = final_output
            if attempts:
                trajectory = str(
                    attempts[0].get("trajectory", attempts[0].get("model_output", final_output))
                )
            success_case = {
                "question": question,
                "trajectory": trajectory,
                "model_output": trajectory,
                "score": rec.get("score"),
            }
            self._append_success_case(task_name, success_case)

        self._task_seeded[task_name] = True
        print(
            f"[ExpeL-Online-ST Restore] {task_name}: "
            f"rules={len(self._task_rule_items_with_count.get(task_name, []))}, "
            f"experience_pool={len(self._task_experience_pool.get(task_name, []))}, "
            f"vectors={len(self._task_experience_vectors.get(task_name, []))}"
        )
