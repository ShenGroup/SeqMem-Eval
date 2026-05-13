import math
import re
from pathlib import Path

from methods.base_method import BaseMethod
from utils.llm_client import generate as llm_generate

try:
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    HumanMessage = None
    SystemMessage = None

try:
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except ImportError:
    transformers_logging = None


class ExpelOnlineMT(BaseMethod):
    # Explicitly mark this method as the online multi-try variant.
    name = "ExpeL-Online-MT"
    _PROMPT_ROOT_BASE = Path(__file__).resolve().parents[1] / "prompts" / "expel"
    _PROMPT_DIR_BY_TASK = {
        "ALFWorld": "alfworld",
    }
    _DEFAULT_PROMPT_DIR = "alfworld"

    _RETRIEVED_TRAJ_PREFIX = (
        "Additional retrieved successful trajectories from prior tasks "
        "(for reference only):"
    )

    def __init__(
        self,
        top_k=3,
        max_tries=3,
        batch_update_size=8,
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        reflection_model_name=None,
        curator_model_name=None,
        temperature=0.0,
        max_new_tokens=1024,
        insights_init="(empty)",
        max_num_rules=20,
        buffer_retrieve_ratio=5,
        **kwargs,
    ):
        super().__init__()
        self.top_k = max(0, int(top_k))
        self.max_tries = max(1, int(max_tries))
        self.batch_update_size = max(1, int(batch_update_size))
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.reflection_model_name = reflection_model_name or generation_model_name
        self.curator_model_name = curator_model_name or generation_model_name
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.insights_init = str(insights_init or "(empty)").strip() or "(empty)"
        self.max_num_rules = max(1, int(max_num_rules))
        self.buffer_retrieve_ratio = max(1, int(buffer_retrieve_ratio))

        self._embedder = None
        self._task_rule_items_with_count = {}
        self._task_recent_success = {}
        self._task_experience_pool = {}
        self._task_retrieval_docs = {}
        self._task_retrieval_vectors = {}
        self._task_pending_rollout = {}
        self._task_seeded = {}
        self._prompts_cache = {}

    def _read_prompt_text(self, path):
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise RuntimeError(f"ExpeL-Online-MT failed to read prompt file: {path}") from exc

    def _read_reflection_fewshots(self, path):
        content = self._read_prompt_text(path)
        shots = [s.strip() for s in content.split("===REFLECTION_FEWSHOT===") if s.strip()]
        return shots

    def _get_task_prompts(self, task):
        """Lazy-load + cache the per-task prompt template set.

        Dispatches by ``task.name`` via ``_PROMPT_DIR_BY_TASK``; unknown tasks
        fall back to the ALFWorld templates so behavior is preserved for tasks
        that haven't been explicitly mapped.
        """
        key = getattr(task, "name", None) or self._DEFAULT_PROMPT_DIR
        cached = self._prompts_cache.get(key)
        if cached is not None:
            return cached
        subdir = self._PROMPT_DIR_BY_TASK.get(key, self._DEFAULT_PROMPT_DIR)
        root = self._PROMPT_ROOT_BASE / subdir
        shared_alfworld_system = (
            self._PROMPT_ROOT_BASE.parent / "alfworld" / "system_instruction.txt"
        )
        system_instruction_path = (
            shared_alfworld_system
            if subdir == "alfworld" and shared_alfworld_system.is_file()
            else root / "system_instruction.txt"
        )
        bundle = {
            "system_instruction": self._read_prompt_text(system_instruction_path),
            "human_instruction_template": self._read_prompt_text(
                root / "human_instruction_template.txt"
            ),
            "human_reflection_template": self._read_prompt_text(
                root / "human_reflection_instruction_template.txt"
            ),
            "task_template": self._read_prompt_text(root / "task_template.txt"),
            "rule_template": self._read_prompt_text(root / "rule_template.txt"),
            "reflection_prefix": self._read_prompt_text(root / "reflection_prefix.txt"),
            "reflection_fewshots": self._read_reflection_fewshots(
                root / "reflection_fewshots.txt"
            ),
        }
        self._prompts_cache[key] = bundle
        return bundle

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
        for row in step_rows:
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
        # Delegate to the task when it exposes a custom parser, if available.
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
                "ExpeL-Online requires sentence-transformers. "
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
            matched = item.get("matched_types", [])
            if matched:
                lines.append(f"Matched Fields: {', '.join(matched)}")
            lines.append(f"Question: {item.get('question', '')}")
            lines.append(
                f"Trajectory: {item.get('trajectory', item.get('model_output', ''))}"
            )
            lines.append("---")
        lines.append("### RETRIEVED SUCCESS CASES (END)")
        return "\n".join(lines).strip()

    def _build_fewshot_examples_block(self, retrieved):
        if not retrieved:
            return "(none)"
        lines = []
        for idx, item in enumerate(retrieved, start=1):
            lines.append(f"### Example {idx}")
            lines.append(f"Task: {item.get('question', '')}")
            lines.append(
                f"Trajectory:\n{item.get('trajectory', item.get('model_output', ''))}"
            )
            score = item.get("retrieval_score")
            if isinstance(score, float):
                lines.append(f"(Similarity: {score:.2f})")
            lines.append("")
        return "\n".join(lines).strip()

    def _build_manual_fewshots_block(self, manual_fewshots, limit=1):
        shots = [str(s).strip() for s in (manual_fewshots or []) if str(s).strip()]
        if not shots:
            return "(none)"
        return "\n\n".join(shots[: max(1, int(limit))]).strip()

    def _build_reflection_fewshots_block(self, task, limit=1):
        shots = self._get_task_prompts(task)["reflection_fewshots"]
        if not shots:
            return "(none)"
        return "\n\n".join(shots[: max(1, int(limit))]).strip()

    def _format_reflection_memory(self, accumulated_reflection):
        text = str(accumulated_reflection or "").strip()
        if not text:
            return ""
        return f"Your memory for the task below:\nTrial 0:\n{text}\n\n"

    def _to_chat_messages(self, system_text, human_text):
        if HumanMessage is not None and SystemMessage is not None:
            return [SystemMessage(content=system_text), HumanMessage(content=human_text)]
        return [
            {"role": "system", "content": system_text},
            {"role": "human", "content": human_text},
        ]

    def _render_chat_messages(self, messages):
        rendered = []
        for message in messages:
            role = getattr(message, "type", None) or str(message.get("role", "message")).upper()
            content = getattr(message, "content", None)
            if content is None:
                content = str(message.get("content", ""))
            rendered.append(f"[{str(role).upper()}]\n{content}")
        return "\n\n".join(rendered).strip()

    def _build_solver_prompt(
        self,
        prompt,
        insights,
        retrieved_cases,
        reflection_note,
        try_idx,
        action_mode=False,
        max_steps=None,
        manual_fewshots=None,
        task=None,
    ):
        fewshots_block = self._build_fewshot_examples_block(retrieved_cases)

        if action_mode:
            max_steps_txt = int(max_steps) if max_steps is not None else 30
            manual_block = self._build_manual_fewshots_block(manual_fewshots, limit=1)
            templates = self._get_task_prompts(task)
            system_text = templates["system_instruction"]
            human_instruction = templates["human_instruction_template"].format(
                instruction=f"{system_text}\n\n",
                max_steps=max_steps_txt,
            )
            task_block = templates["task_template"].format(task=prompt)
            rules_block = templates["rule_template"].format(rules=insights)
            human_text = (
                f"{human_instruction}\n"
                f"{manual_block}\n\n"
                "(END OF EXAMPLES)\n\n"
                f"{self._RETRIEVED_TRAJ_PREFIX}\n"
                f"{fewshots_block}\n\n"
                f"{rules_block}\n\n"
                "Your memory for the task below:\n"
                f"{reflection_note or '(empty)'}\n\n"
                f"# CURRENT ATTEMPT: {try_idx}\n\n"
                f"{task_block}"
            )
            return self._render_chat_messages(self._to_chat_messages(system_text, human_text))

        system_text = (
            "You are an advanced reasoning agent. "
            "Solve accurately and reuse valid prior experiences and rules."
        )
        human_text = (
            "# FEWSHOT EXAMPLES (DYNAMIC, RETRIEVED FROM SUCCESS POOL)\n"
            f"{fewshots_block}\n\n"
            "# EXISTING RULES\n"
            f"{insights}\n\n"
            "# PREVIOUS FAILURE REFLECTIONS FOR THIS SAME TASK\n"
            f"{reflection_note or '(empty)'}\n\n"
            f"# CURRENT ATTEMPT: {try_idx}\n\n"
            "# REASONING TRACE (OPTIONAL, FOR METHOD-INTERNAL BOOKKEEPING)\n"
            "You may wrap your reasoning in <trajectory>...</trajectory>. "
            "Follow the output format specified by the task below for the final answer.\n\n"
            "# CURRENT PROBLEM\n"
            f"{prompt}"
        )
        return self._render_chat_messages(self._to_chat_messages(system_text, human_text))

    def _build_reflection_prompt(self, prompt, failed_output, accumulated_reflection, action_mode=False, task=None):
        if action_mode:
            templates = self._get_task_prompts(task)
            reflection_fewshots_block = self._build_reflection_fewshots_block(task)
            reflection_memory = self._format_reflection_memory(accumulated_reflection)
            return (
                f"{templates['human_reflection_template']}\n"
                f"{reflection_fewshots_block}\n\n"
                "(END OF EXAMPLES)\n\n"
                f"{reflection_memory}"
                f"Previous trial:\n{prompt}\n{failed_output}\n"
                f"{templates['reflection_prefix']}\n"
                "Return only <reflection>...</reflection>."
            )
        else:
            task_type_hint = "arithmetic mistakes, parsing mistakes, assumption errors, final-answer format"
        return (
            "# SELF-REFLECTION\n"
            "Analyze why the previous attempt failed and provide short actionable guidance for the next retry.\n"
            f"Focus on: {task_type_hint}.\n\n"
            "# PREVIOUS REFLECTIONS\n"
            f"{accumulated_reflection or '(empty)'}\n\n"
            "# PROBLEM\n"
            f"{prompt}\n\n"
            "# FAILED ATTEMPT OUTPUT\n"
            f"{failed_output}\n\n"
            "Return ONLY one concise reflection wrapped in <reflection>...</reflection>."
        )

    def _build_pair_update_prompt(self, current_insights, success_case, fail_case):
        return (
            "# RULE CURATOR (SUCCESS vs FAILURE)\n\n"
            "Update the existing ExpeL rule set by comparing one success and one failure trajectory.\n"
            "Keep rules concise, reusable, and implementation-agnostic.\n\n"
            "## EXISTING RULES\n"
            f"{current_insights or '(empty)'}\n\n"
            "## SUCCESS CASE\n"
            f"Question: {success_case.get('question', '')}\n"
            f"Trajectory: {success_case.get('trajectory', success_case.get('model_output', ''))}\n\n"
            "## FAILED CASE\n"
            f"Question: {fail_case.get('question', '')}\n"
            f"Trajectory: {fail_case.get('trajectory', fail_case.get('model_output', ''))}\n\n"
            "Apply rule operations in this format:\n"
            f"{self._build_operation_format_instruction()}"
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
                    f"Trajectory: {item.get('trajectory', item.get('model_output', ''))}",
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

        rules = [r for r in rules if r[1] > 0]
        rules.sort(key=lambda x: x[1], reverse=True)
        if len(rules) > self.max_num_rules:
            rules = rules[: self.max_num_rules]
        self._task_rule_items_with_count[task_name] = rules
        return changed

    def _update_insights_from_pair(self, task_name, success_case, fail_case):
        prompt = self._build_pair_update_prompt(
            current_insights=self._current_insights(task_name),
            success_case=success_case,
            fail_case=fail_case,
        )
        raw = llm_generate(
            model_name=self.curator_model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=max(self.max_new_tokens, 512),
        )
        return self._apply_rule_operations(task_name, self._parse_rules_ops(raw))

    def _update_insights_from_batch(self, task_name, batch_success):
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

    def _extract_structured_fields(self, text):
        raw = str(text or "").strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        thoughts = []
        actions = []
        steps = []
        for ln in lines:
            low = ln.lower()
            if low.startswith("> think") or low.startswith("think:"):
                thought = re.sub(r"^>?\s*think:\s*", "", ln, flags=re.IGNORECASE).strip()
                if thought:
                    thoughts.append(thought)
            elif ln.startswith(">"):
                action = re.sub(r"^>\s*", "", ln).strip()
                if action:
                    actions.append(action)
            steps.append(ln)
        if not thoughts and raw:
            thoughts = [raw[:300]]
        if not steps and raw:
            steps = [raw]
        return thoughts, actions, steps

    def _build_case_retrieval_docs(self, case):
        docs = []
        question = str(case.get("question", "")).strip()
        trajectory = str(case.get("trajectory", case.get("model_output", ""))).strip()
        reflections = case.get("reflections", [])
        reflection_text = "\n".join([str(r).strip() for r in reflections if str(r).strip()])
        thoughts, actions, steps = self._extract_structured_fields(trajectory)
        if question:
            docs.append({"type": "task", "text": question})
        if thoughts:
            docs.append({"type": "thought", "text": "\n".join(thoughts[-3:])})
        if steps:
            docs.append({"type": "step", "text": "\n".join(steps[-5:])})
        if actions:
            docs.append({"type": "action", "text": "\n".join(actions[-5:])})
        if reflection_text:
            docs.append({"type": "reflection", "text": reflection_text})
        return docs

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
            "reflections": [],
            "seed_demo": True,
        }

    def _seed_experience_pool_from_manual_fewshots(self, task_name, manual_fewshots):
        if self._task_seeded.get(task_name, False):
            return
        shots = manual_fewshots or []
        seeded = False
        for shot in shots:
            parsed = self._parse_fewshot_seed(shot)
            if not parsed:
                continue
            self._append_success_case(task_name, parsed)
            seeded = True
        self._task_seeded[task_name] = True
        if not seeded:
            return

    def _append_success_case(self, task_name, success_case):
        pool = self._task_experience_pool.setdefault(task_name, [])
        case_index = len(pool)
        pool.append(success_case)

        retrieval_docs = self._task_retrieval_docs.setdefault(task_name, [])
        texts = []
        pending_docs = []
        for doc in self._build_case_retrieval_docs(success_case):
            text = str(doc.get("text", "")).strip()
            if not text:
                continue
            pending_docs.append({"type": doc.get("type", "task"), "text": text, "case_index": case_index})
            texts.append(text)
        if not pending_docs:
            return
        vectors = self._encode(texts)
        self._task_retrieval_vectors.setdefault(task_name, []).extend(vectors)
        retrieval_docs.extend(pending_docs)

    def retrieve_memory(self, task, **kwargs):
        query = str(kwargs.get("query", "")).strip()
        task_name = task.name
        self._seed_experience_pool_from_manual_fewshots(
            task_name=task_name,
            manual_fewshots=kwargs.get("fewshots", []),
        )
        pool = self._task_experience_pool.get(task_name, [])
        docs = self._task_retrieval_docs.get(task_name, [])
        vectors = self._task_retrieval_vectors.get(task_name, [])
        if self.top_k <= 0 or not pool or not vectors or not docs:
            return []
        if not query:
            import os as _os
            if _os.environ.get("BWT_DISABLE_RECENCY_FALLBACK"):
                return []
            return [
                {
                    "question": c.get("question", ""),
                    "model_output": c.get("model_output", ""),
                    "trajectory": c.get("trajectory", c.get("model_output", "")),
                    "matched_types": ["recent"],
                    "retrieval_score": None,
                }
                for c in pool[-self.top_k :]
            ]
        query_vec = self._encode([query])[0]
        similarities = [self._cosine(query_vec, vec) for vec in vectors]
        top_doc_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )[: max(self.top_k * self.buffer_retrieve_ratio, self.top_k)]

        by_case = {}
        for doc_idx in top_doc_indices:
            doc = docs[doc_idx]
            case_idx = doc.get("case_index")
            if case_idx is None or case_idx < 0 or case_idx >= len(pool):
                continue
            score = float(similarities[doc_idx])
            item = by_case.setdefault(
                case_idx,
                {"best_score": score, "matched_types": set()},
            )
            if score > item["best_score"]:
                item["best_score"] = score
            item["matched_types"].add(str(doc.get("type", "task")))

        ranked_cases = sorted(by_case.items(), key=lambda x: x[1]["best_score"], reverse=True)[: self.top_k]
        results = []
        for case_idx, meta in ranked_cases:
            case = pool[case_idx]
            results.append(
                {
                    "question": case.get("question", ""),
                    "model_output": case.get("model_output", ""),
                    "trajectory": case.get("trajectory", case.get("model_output", "")),
                    "matched_types": sorted(list(meta["matched_types"])),
                    "retrieval_score": float(meta["best_score"]),
                }
            )
        return results

    def _extract_trajectory(self, raw_output):
        raw = str(raw_output or "").strip()
        trajectory = self._extract_tag_content(raw, "trajectory")
        if not trajectory:
            trajectory = raw
        return trajectory

    def run_trial(self, task, entry, inputs, timer, update_memory=True):
        if not self._supports_interactive_task(task):
            return None

        task_name = task.name
        prompt = task.build_prompt(entry)
        query = task.get_query(entry, inputs)

        with timer.track(f"{self.name}/{task_name}/retrieve"):
            memory = self.retrieve_memory(task, query=query, entry=entry, **inputs)

        insights_before = self._current_insights(task_name)
        reflection_note = ""
        attempts = []
        reflections = []
        final_output = ""
        final_trajectory = ""
        final_score = 0.0

        max_steps = max(1, int(getattr(task, "max_steps", 30)))
        for try_idx in range(1, self.max_tries + 1):
            current_observation = task.start_episode(entry)
            initial_observation = current_observation
            step_rows = []
            for step_idx in range(max_steps):
                step_prompt = self._build_step_prompt_text(
                    prompt, current_observation, step_rows,
                    initial_observation=initial_observation,
                )
                generation_prompt = self._build_solver_prompt(
                    prompt=step_prompt,
                    insights=insights_before,
                    retrieved_cases=memory,
                    reflection_note=reflection_note,
                    try_idx=try_idx,
                    action_mode=True,
                    max_steps=max_steps,
                    manual_fewshots=inputs.get("fewshots", []),
                    task=task,
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
                    meta={"agent_name": "expel_mt", "try": try_idx},
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
            is_success = score >= 1.0
            candidate_raw = self._build_trial_model_output(step_rows, success=is_success)
            candidate_trajectory = self._extract_trajectory(candidate_raw)
            final_output = candidate_raw
            final_trajectory = candidate_trajectory
            final_score = score

            attempts.append(
                {
                    "try": try_idx,
                    "model_output": candidate_raw,
                    "raw_output": candidate_raw,
                    "trajectory": candidate_trajectory,
                    "score": score,
                    "feedback": "success" if is_success else "failure",
                }
            )
            if is_success:
                break

            reflection_prompt = self._build_reflection_prompt(
                prompt=prompt,
                failed_output=candidate_raw,
                accumulated_reflection=reflection_note,
                action_mode=True,
                task=task,
            )
            with timer.track(f"{self.name}/{task_name}/generate"):
                reflection_raw = llm_generate(
                    model_name=self.reflection_model_name,
                    prompt=reflection_prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                )
            reflection = self._extract_tag_content(reflection_raw, "reflection") or str(reflection_raw).strip()
            if reflection:
                reflections.append(reflection)
                reflection_note = f"{reflection_note}\n- {reflection}".strip() if reflection_note else f"- {reflection}"

        feedback = "success" if final_score >= 1.0 else "failure"
        record = task.build_memory_record(entry, final_output, feedback, final_score)
        self._task_pending_rollout[task_name] = {
            "question": query,
            "retrieved_cases": memory,
            "attempts": attempts,
            "reflections": reflections,
            "insights_before": insights_before,
            "rollout_type": "online_multi_try_interactive",
            "trajectory": final_trajectory,
        }
        if update_memory:
            with timer.track(f"{self.name}/{task_name}/update"):
                self.update_memory(task, final_output, record=record, entry=entry, **inputs)
        return final_score

    def generate(self, task, prompt, **kwargs):
        entry = kwargs.get("entry")
        query = str(kwargs.get("query", "")).strip()
        retrieved = kwargs.get("memory", [])
        task_name = task.name
        insights_before = self._current_insights(task_name)
        action_mode = bool(getattr(task, "action_mode", False))

        reflection_note = ""
        attempts = []
        reflections = []
        final_output = ""
        final_trajectory = ""

        for try_idx in range(1, self.max_tries + 1):
            generation_prompt = self._build_solver_prompt(
                prompt=prompt,
                insights=insights_before,
                retrieved_cases=retrieved,
                reflection_note=reflection_note,
                try_idx=try_idx,
                action_mode=action_mode,
                max_steps=getattr(task, "max_steps", 30),
                manual_fewshots=kwargs.get("fewshots", []),
                task=task,
            )
            candidate_raw = llm_generate(
                model_name=self.generation_model_name,
                prompt=generation_prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            candidate_trajectory = self._extract_trajectory(candidate_raw)
            candidate_for_eval = str(candidate_raw)
            final_output = candidate_for_eval
            final_trajectory = candidate_trajectory

            score = task.evaluate_entry(candidate_for_eval, entry)
            is_success = score >= 1.0
            attempt_record = {
                "try": try_idx,
                "model_output": candidate_for_eval,
                "raw_output": str(candidate_raw),
                "trajectory": candidate_trajectory,
                "score": score,
                "feedback": "success" if is_success else "failure",
            }
            attempts.append(attempt_record)
            if is_success:
                break

            reflection_prompt = self._build_reflection_prompt(
                prompt=prompt,
                failed_output=str(candidate_raw),
                accumulated_reflection=reflection_note,
                action_mode=action_mode,
                task=task,
            )
            reflection_raw = llm_generate(
                model_name=self.reflection_model_name,
                prompt=reflection_prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            reflection = self._extract_tag_content(reflection_raw, "reflection") or str(reflection_raw).strip()
            if reflection:
                reflections.append(reflection)
                reflection_note = (
                    f"{reflection_note}\n- {reflection}".strip()
                    if reflection_note
                    else f"- {reflection}"
                )

        self._task_pending_rollout[task_name] = {
            "question": query,
            "retrieved_cases": retrieved,
            "attempts": attempts,
            "reflections": reflections,
            "insights_before": insights_before,
            "rollout_type": "online_multi_try",
            "trajectory": final_trajectory,
        }
        return final_output

    def update_memory(self, task, output, **kwargs):
        record = kwargs.get("record")
        if not record:
            return

        task_name = task.name
        rollout = self._task_pending_rollout.pop(task_name, {})
        attempts = rollout.get("attempts", [])
        reflections = rollout.get("reflections", [])
        insights_before = rollout.get("insights_before", self._current_insights(task_name))

        question = str(record.get("question", rollout.get("question", ""))).strip()
        final_output = str(record.get("model_output", output))
        trajectory = str(rollout.get("trajectory", final_output))
        is_success = str(record.get("feedback", "")).lower() == "success"

        success_case = None
        failed_attempts = []
        for attempt in attempts:
            if attempt.get("feedback") == "success":
                success_case = {
                    "question": question,
                    "model_output": attempt.get("model_output", final_output),
                    "trajectory": attempt.get("trajectory", attempt.get("model_output", final_output)),
                    "score": attempt.get("score"),
                    "reflections": list(reflections),
                }
            else:
                failed_attempts.append(
                    {
                        "question": question,
                        "model_output": attempt.get("model_output", ""),
                        "trajectory": attempt.get("trajectory", attempt.get("model_output", "")),
                        "score": attempt.get("score"),
                    }
                )

        pair_updated = False
        pair_update_count = 0
        batch_updated = False
        if is_success:
            if success_case is None:
                success_case = {
                    "question": question,
                    "model_output": final_output,
                    "trajectory": trajectory,
                    "score": record.get("score"),
                    "reflections": list(reflections),
                }
            self._append_success_case(task_name, success_case)
            self._task_recent_success.setdefault(task_name, []).append(success_case)

            # Official-like full pairing in online setting:
            # one success trajectory is compared against all failed tries.
            if failed_attempts:
                for fail_case in failed_attempts:
                    updated = self._update_insights_from_pair(
                        task_name=task_name,
                        success_case=success_case,
                        fail_case=fail_case,
                    )
                    if updated:
                        pair_update_count += 1
                pair_updated = pair_update_count > 0

            recent_success = self._task_recent_success.get(task_name, [])
            if len(recent_success) >= self.batch_update_size:
                batch_updated = self._update_insights_from_batch(
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
                    "max_tries": self.max_tries,
                    "attempts": attempts,
                    "reflections": reflections,
                },
            },
            "memory_meta": {
                "method_variant": "ExpeL-Online-MT",
                "rollout_type": rollout.get("rollout_type", "online_multi_try"),
                "insights_before": insights_before,
                "insights_after": insights_after,
                "insights_pair_updated": pair_updated,
                "insights_pair_update_count": pair_update_count,
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
        self._task_retrieval_docs.setdefault(task_name, [])
        self._task_retrieval_vectors.setdefault(task_name, [])
        self._task_seeded.setdefault(task_name, False)
        return None

    def restore_state_from_memory(self, task):
        """Rebuild rule library + experience pool from `self.memory` records.

        Used by the holdout harness after seeding `self.memory` with a
        training-run JSONL. Without this, rules and pool restart empty and
        the retrieval step produces an empty prompt, making holdout numbers
        incomparable to the training run at the seed checkpoint.
        """
        task_name = task.name
        records = [r for r in self.memory if r.get("task") == task_name]
        if not records:
            return

        # 1. Rules — parse the last record's memory_meta.insights_after string.
        #    Counts are not persisted; assign 1 each (fine for read-only eval
        #    since update_memory is skipped and counts only matter for curator
        #    pruning in future updates).
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

        # 2. Experience pool — replay success cases from score==1 records.
        for rec in records:
            try:
                score = float(rec.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < 1.0:
                continue
            mem = rec.get("memory") or {}
            rollout = mem.get("rollout") or {}
            reflections = list(rollout.get("reflections") or [])
            attempts = rollout.get("attempts") or []
            question = str(rec.get("question", "")).strip()
            final_output = str(rec.get("model_output", ""))

            success_case = None
            for att in attempts:
                if att.get("feedback") == "success":
                    success_case = {
                        "question": question,
                        "model_output": att.get("model_output", final_output),
                        "trajectory": att.get(
                            "trajectory", att.get("model_output", final_output)
                        ),
                        "score": att.get("score"),
                        "reflections": reflections,
                    }
                    break
            if success_case is None:
                success_case = {
                    "question": question,
                    "model_output": final_output,
                    "trajectory": final_output,
                    "score": rec.get("score"),
                    "reflections": reflections,
                }
            self._append_success_case(task_name, success_case)

        # Mark as seeded so manual fewshots are not re-seeded on top of the
        # restored pool when retrieve_memory first runs.
        self._task_seeded[task_name] = True

        print(
            f"[ExpeL-Online-MT Restore] {task_name}: "
            f"rules={len(self._task_rule_items_with_count.get(task_name, []))}, "
            f"experience_pool={len(self._task_experience_pool.get(task_name, []))}, "
            f"retrieval_docs={len(self._task_retrieval_docs.get(task_name, []))}"
        )
