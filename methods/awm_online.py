from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import torch

from methods.base_method import BaseMethod
from utils.llm_client import generate as llm_generate


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-zA-Z0-9_]+", str(a or "").lower()))
    tb = set(re.findall(r"[a-zA-Z0-9_]+", str(b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


class AwmOnline(BaseMethod):
    name = "AWM-Online"

    def __init__(
        self,
        top_k=3,
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        curator_model_name=None,
        temperature=0.0,
        max_new_tokens=1024,
        induce_steps=1,
        workflow_init="",
        awminstruction_path=None,
        awm_oneshot_path=None,
        history_char_limit=6000,
        online_exemplar_top_k=2,
        static_exemplar_top_k=1,
        **kwargs,
    ):
        super().__init__()
        self.top_k = max(0, int(top_k))
        self.embedding_model_name = str(embedding_model_name or "").strip()
        self.generation_model_name = generation_model_name
        self.curator_model_name = curator_model_name or generation_model_name
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.induce_steps = max(1, int(induce_steps))
        self.workflow_init = str(workflow_init or "").strip()
        self.history_char_limit = max(500, int(history_char_limit))
        self.online_exemplar_top_k = max(0, int(online_exemplar_top_k))
        self.static_exemplar_top_k = max(0, int(static_exemplar_top_k))

        self._awminstruction_path = str(awminstruction_path).strip() if awminstruction_path else None
        self._awm_oneshot_path = str(awm_oneshot_path).strip() if awm_oneshot_path else None

        self._assets_dir = Path(__file__).resolve().parents[1] / "prompts" / "awm"
        self._alfworld_assets_dir = self._assets_dir / "alfworld"

        self._task_workflow_records = {}
        self._task_success_examples = {}
        self._task_sample_counter = {}
        self._task_success_counter = {}

        self._profile_prompt_cache = {}
        self._static_exemplars_cache = {}

        self._embedder = None
        self._embedder_disabled = False

    def _task_profile(self, task_name):
        low = str(task_name or "").lower()
        if "alfworld" in low:
            return "action"
        if "apibench" in low:
            return "api_call"
        return "math"

    def _is_alfworld_task(self, task_name):
        return "alfworld" in str(task_name or "").lower()

    def _supports_interactive_task(self, task):
        return all(hasattr(task, name) for name in ("start_episode", "step_episode", "finish_episode"))

    def _default_induction_instruction(self, profile):
        if profile == "action":
            return (
                "Given solved interaction trajectories, extract reusable workflows that can be applied "
                "to similar tasks. Each workflow must be a non-overlapping sub-routine with at least two steps."
            )
        if profile == "tool_call":
            return (
                "Given solved multi-turn function-calling trajectories, extract reusable workflows that can "
                "be applied to similar tasks. Each workflow must be a non-overlapping sub-routine of at "
                "least two function calls, expressed at the level of intent (what to do) rather than "
                "literal arguments."
            )
        if profile == "api_call":
            return (
                "Given solved API-selection traces (natural-language task -> chosen library API call), "
                "extract reusable workflows that map task intent to API family and argument choices. "
                "Each workflow must be a non-overlapping sub-routine with at least two steps: "
                "(1) infer the task's domain, (2) select the API family / entry-point, and optionally "
                "(3) fill in the required arguments (repo_or_dir, model, etc.)."
            )
        return (
            "Given solved reasoning traces, extract reusable workflows that can be applied to similar tasks. "
            "Each workflow must be a non-overlapping sub-routine with at least two steps."
        )

    def _default_induction_one_shot(self, profile):
        if profile == "action":
            return (
                "Website: Household Environment\n"
                "## Query 1: Put a clean mug on the counter.\n"
                "Actions and Environments:\n"
                "open cabinet 1\n"
                "take mug 2 from cabinet 1\n"
                "clean mug 2 with sink 1\n"
                "put mug 2 in/on counter 1\n\n"
                "Summary Workflows:\n"
                "## retrieve_object_then_place\n"
                "Open relevant container, take target object, and place it at destination.\n\n"
                "## clean_then_place\n"
                "When object must be clean first, clean it before final placement."
            )
        if profile == "tool_call":
            return (
                "Task Group: Multi-Turn Function Calling\n"
                "## Query 1: Authenticate the user, post a tweet, then search for replies.\n"
                "Trajectory:\n"
                "Step 1: authenticate_twitter(username='user123', password='secret')\n"
                "Step 2: post_tweet(content='Hello world')\n"
                "Step 3: search_tweets(query='Hello world')\n\n"
                "Summary Workflows:\n"
                "## authenticate_then_act\n"
                "When an API requires authentication, call the auth function first before any state-changing call.\n\n"
                "## act_then_verify\n"
                "After a mutating call (post/create/write), follow up with a read call to verify the effect."
            )
        if profile == "api_call":
            return (
                "Task Group: API Selection\n"
                "## Query 1: What is an API to classify sports activities in videos?\n"
                "Trajectory:\n"
                "Step 1: Identify domain -> Video Classification.\n"
                "Step 2: Pick API family -> torch.hub.load on a Kinetics-pretrained 3D ResNet.\n"
                "Step 3: Fill arguments -> repo_or_dir='facebookresearch/pytorchvideo', model='slow_r50', pretrained=True.\n"
                "Final API Call: torch.hub.load(repo_or_dir='facebookresearch/pytorchvideo', model='slow_r50', pretrained=True)\n\n"
                "Summary Workflows:\n"
                "## classify_domain_then_select_api\n"
                "First map the task description to a concrete domain keyword; only then pick an API family whose domain matches.\n\n"
                "## fill_required_args_from_pool\n"
                "Once the API family is fixed, populate its required keyword arguments (e.g. repo_or_dir / model) from the pretrained-model pool rather than inventing names."
            )
        return (
            "Task Group: Quantitative Reasoning\n"
            "## Query 1: A student buys 3 notebooks at $4 each and 2 pens at $1 each. What is the total cost?\n"
            "Trajectory:\n"
            "Step 1: Parse quantities and unit prices.\n"
            "Step 2: Compute notebook cost = 3 * 4 = 12.\n"
            "Step 3: Compute pen cost = 2 * 1 = 2.\n"
            "Step 4: Sum = 12 + 2 = 14.\n"
            "Final Answer: 14\n\n"
            "Summary Workflows:\n"
            "## parse_and_formalize\n"
            "Identify entities, quantities, constraints, and target variable before solving.\n\n"
            "## stepwise_compute_and_check\n"
            "Perform computations in verifiable steps and check arithmetic consistency."
        )

    def _load_text_or_default(self, path, default_text):
        if not path:
            return default_text
        p = Path(path)
        if not p.exists():
            return default_text
        text = p.read_text(encoding="utf-8").strip()
        return text if text else default_text

    def _get_profile_prompts(self, profile, task_name=None):
        if self._is_alfworld_task(task_name):
            variant = "alfworld"
        else:
            variant = "default"
        cache_key = f"{profile}:{variant}"
        cached = self._profile_prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        if profile == "action" and self._is_alfworld_task(task_name):
            default_instruction_path = self._alfworld_assets_dir / "instruction_action.txt"
            default_one_shot_path = self._alfworld_assets_dir / "one_shot_action.txt"
        else:
            default_instruction_path = self._assets_dir / f"instruction_{profile}.txt"
            default_one_shot_path = self._assets_dir / f"one_shot_{profile}.txt"

        instruction_default = self._load_text_or_default(
            str(default_instruction_path),
            self._default_induction_instruction(profile),
        )
        one_shot_default = self._load_text_or_default(
            str(default_one_shot_path),
            self._default_induction_one_shot(profile),
        )

        instruction = self._load_text_or_default(self._awminstruction_path, instruction_default)
        one_shot = self._load_text_or_default(self._awm_oneshot_path, one_shot_default)

        out = {
            "instruction": instruction,
            "one_shot": one_shot,
        }
        self._profile_prompt_cache[cache_key] = out
        return out

    def _alfworld_step_instruction(self):
        default_text = (
            "You are a large language model trained to solve ALFWorld household tasks. "
            "Output the next action and wait for the next observation.\n"
            "Action space examples:\n"
            "1. go to <receptacle>\n"
            "2. open <container>\n"
            "3. take <object> from <receptacle>\n"
            "4. put <object> in/on <receptacle>\n"
            "5. clean/heat/cool <object> with <tool>\n"
            "6. use/examine/look/inventory\n"
            "Return exactly one executable action line."
        )
        shared_path = Path(__file__).resolve().parents[1] / "prompts" / "alfworld" / "system_instruction.txt"
        return self._load_text_or_default(str(shared_path), default_text)

    def _load_static_exemplars(self, profile):
        cached = self._static_exemplars_cache.get(profile)
        if cached is not None:
            return cached

        default_data = []
        json_path = self._assets_dir / f"exemplars_{profile}.json"
        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    default_data = [x for x in loaded if isinstance(x, dict)]
            except Exception:
                default_data = []

        self._static_exemplars_cache[profile] = default_data
        return default_data

    def _ensure_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if self._embedder_disabled:
            return None
        try:
            from utils.embedder import get_embedder

            self._embedder = get_embedder(self.embedding_model_name)
            return self._embedder
        except Exception:
            self._embedder_disabled = True
            return None

    def _encode_text(self, text):
        text = str(text or "").strip()
        if not text:
            return None
        embedder = self._ensure_embedder()
        if embedder is None:
            return None
        try:
            vec = embedder.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            result = [float(x) for x in vec]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result
        except Exception:
            self._embedder_disabled = True
            self._embedder = None
            return None

    def _truncate_text(self, text, max_chars):
        # Truncation disabled: per-experiment config keeps full history in the
        # prompt so the solver can see every prior step. history_char_limit
        # is retained on the instance for backwards-compat but is ignored here.
        return str(text or "")

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

    def _extract_workflow(self, text):
        for tag in ("workflow", "workflows", "cheatsheet"):
            content = self._extract_tag_content(text, tag)
            if content:
                return content
        return str(text or "").strip()

    def _similarity(self, query, target, query_embedding=None, target_embedding=None):
        if query_embedding is not None and target_embedding is not None:
            return _cosine(query_embedding, target_embedding)
        return _lexical_similarity(query, target)

    def _sample_trajectory_text(self, sample):
        parsed_actions = sample.get("parsed_actions")
        if isinstance(parsed_actions, list) and parsed_actions:
            lines = [f"Step {i}: {str(act).strip()}" for i, act in enumerate(parsed_actions, start=1)]
            return "\n".join(lines)

        executed_actions = sample.get("executed_actions")
        if isinstance(executed_actions, list) and executed_actions:
            lines = [f"Step {i}: {str(act).strip()}" for i, act in enumerate(executed_actions, start=1)]
            return "\n".join(lines)

        return str(sample.get("model_output", "")).strip()

    def _build_induction_prompt(self, task_name, profile, sample):
        prompts = self._get_profile_prompts(profile, task_name=task_name)
        question = str(sample.get("question", "")).strip()
        trajectory = self._sample_trajectory_text(sample)

        lines = [
            prompts["instruction"],
            "",
            prompts["one_shot"],
            "",
            f"Task: {task_name}",
            "",
            "New successful example:",
            f"Query: {question}",
            "Trajectory:",
            trajectory,
            "",
            "Summary Workflows:",
        ]
        return self._truncate_text("\n".join(lines).strip(), self.history_char_limit * 2)

    def _extract_single_action(self, task, raw_output):
        text = str(raw_output or "").strip()
        if not text:
            return ""

        if hasattr(task, "_extract_actions"):
            try:
                actions = list(task._extract_actions(text))
            except Exception:
                actions = []
            if actions:
                return str(actions[0]).strip()

        lines = []
        for row in text.splitlines():
            line = str(row).strip()
            if not line:
                continue
            if line.startswith("```"):
                continue
            if line.startswith(">"):
                line = re.sub(r"^>\s*", "", line).strip()
            low = line.lower()
            if low.startswith("action:"):
                line = line.split(":", 1)[1].strip()
                low = line.lower()
            if low.startswith("think:") or low.startswith("thought:") or low.startswith("observation:"):
                continue
            if line:
                lines.append(line)
        return lines[0] if lines else ""

    def _build_alfworld_step_prompt(
        self,
        task_prompt,
        current_observation,
        step_rows,
        workflows,
        exemplars,
        manual_fewshots,
        step_idx,
        max_steps,
        initial_observation=None,
    ):
        recent = step_rows
        history_lines = []
        init_obs = str(initial_observation or "").strip()
        if init_obs:
            history_lines.append(init_obs)
        for row in recent:
            history_lines.append(f"Action: {row.get('action', '')}")
            obs = str(row.get("observation", "")).strip()
            if obs:
                history_lines.append(obs)
        history_block = "\n".join(history_lines).strip() if history_lines else "(none)"
        fewshot_text = "\n\n".join(str(x).strip() for x in manual_fewshots if str(x).strip())

        lines = [
            self._alfworld_step_instruction(),
            "",
            "## Retrieved Workflows",
            self._format_workflow_block(workflows),
        ]
        if fewshot_text:
            lines.extend(
                [
                    "",
                    "## Few-shot Examples",
                    fewshot_text,
                ]
            )
        lines.extend(
            [
                "",
                "## Current Task",
                str(task_prompt).strip(),
                "",
                "Recent trajectory:",
                history_block,
                "",
                f"## Current Step ({step_idx}/{max_steps})",
                "Current observation:",
                str(current_observation or "").strip(),
                "",
                "Respond with exactly one step in the required format:",
                "Thought: <your thoughts>",
                "Action: <your next action>",
            ]
        )
        return self._truncate_text("\n".join(lines).strip(), self.history_char_limit * 3)

    def _build_trial_model_output(self, step_rows, success):
        lines = []
        for row in step_rows:
            action = str(row.get("action", "")).strip()
            obs = str(row.get("observation", "")).strip()
            if action:
                lines.append(f"Action: {action}")
            if obs:
                lines.append(obs)
        return "\n".join(lines).strip()

    def _induce_workflow(self, task_name, sample):
        profile = self._task_profile(task_name)
        prompt = self._build_induction_prompt(task_name, profile, sample)
        induced_raw = llm_generate(
            model_name=self.curator_model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=max(self.max_new_tokens, 512),
        )
        return self._extract_workflow(induced_raw)

    def _new_workflow_record(self, task_name, workflow_text, source_sample):
        workflow_text = str(workflow_text or "").strip()
        if not workflow_text:
            return None

        source_qid = str(source_sample.get("qid", "")).strip()
        record = {
            "record_id": str(uuid.uuid4()),
            "task": task_name,
            "workflow_text": workflow_text,
            "query_text": str(source_sample.get("question", "")).strip(),
            "source_qid": source_qid,
            "source_score": source_sample.get("score"),
            "source_feedback": source_sample.get("feedback", ""),
            "created_at": str(time.time()),
            "profile": self._task_profile(task_name),
            "embedding": self._encode_text(workflow_text),
        }
        return record

    def _new_init_workflow_record(self, task_name):
        if not self.workflow_init:
            return None
        return {
            "record_id": f"init-{task_name}",
            "task": task_name,
            "workflow_text": self.workflow_init,
            "query_text": "",
            "source_qid": "",
            "source_score": None,
            "source_feedback": "init",
            "created_at": "0",
            "profile": self._task_profile(task_name),
            "embedding": self._encode_text(self.workflow_init),
        }

    def _workflow_public_view(self, record):
        return {
            "record_id": record.get("record_id", ""),
            "workflow_text": record.get("workflow_text", ""),
            "query_text": record.get("query_text", ""),
            "source_qid": record.get("source_qid", ""),
            "source_score": record.get("source_score"),
            "source_feedback": record.get("source_feedback", ""),
            "profile": record.get("profile", ""),
            "created_at": record.get("created_at", ""),
            "retrieval_score": float(record.get("retrieval_score", 0.0)),
        }

    def _retrieve_workflows(self, task_name, query):
        records = self._task_workflow_records.get(task_name, [])
        if not records or self.top_k <= 0:
            return []

        query = str(query or "").strip()
        if not query:
            import os as _os
            if _os.environ.get("BWT_DISABLE_RECENCY_FALLBACK"):
                return []
            recent = records[-self.top_k :]
            return [self._workflow_public_view(dict(r, retrieval_score=0.0)) for r in recent]

        query_embedding = self._encode_text(query)
        scored = []
        for row in records:
            score = self._similarity(
                query,
                row.get("workflow_text", ""),
                query_embedding=query_embedding,
                target_embedding=row.get("embedding"),
            )
            scored.append((row, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_rows = scored[: self.top_k]
        return [self._workflow_public_view(dict(row, retrieval_score=score)) for row, score in top_rows]

    def _score_examples(self, query, examples, text_key):
        query = str(query or "").strip()
        if not examples:
            return []
        if not query:
            import os as _os
            if _os.environ.get("BWT_DISABLE_RECENCY_FALLBACK"):
                return []
            return [(ex, 0.0) for ex in examples]

        query_embedding = self._encode_text(query)
        scored = []
        for ex in examples:
            text = str(ex.get(text_key, "")).strip()
            target_embedding = ex.get("embedding")
            if target_embedding is None:
                target_embedding = self._encode_text(text)
                ex["embedding"] = target_embedding
            score = self._similarity(
                query,
                text,
                query_embedding=query_embedding,
                target_embedding=target_embedding,
            )
            scored.append((ex, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _retrieve_exemplars(self, task_name, query):
        profile = self._task_profile(task_name)
        collected = []
        seen = set()

        online_examples = self._task_success_examples.get(task_name, [])
        for ex, score in self._score_examples(query, online_examples, "question")[: self.online_exemplar_top_k]:
            key = (str(ex.get("question", "")).strip(), str(ex.get("trajectory", "")).strip())
            if key in seen:
                continue
            seen.add(key)
            collected.append(
                {
                    "source": "online",
                    "query": str(ex.get("question", "")).strip(),
                    "trajectory": str(ex.get("trajectory", "")).strip(),
                    "score": float(score),
                }
            )

        static_examples = self._load_static_exemplars(profile)
        for ex, score in self._score_examples(query, static_examples, "query")[: self.static_exemplar_top_k]:
            key = (str(ex.get("query", "")).strip(), str(ex.get("trajectory", "")).strip())
            if key in seen:
                continue
            seen.add(key)
            collected.append(
                {
                    "source": "static",
                    "query": str(ex.get("query", "")).strip(),
                    "trajectory": str(ex.get("trajectory", "")).strip(),
                    "score": float(score),
                }
            )

        return collected

    def _format_workflow_block(self, workflows):
        if not workflows:
            return "(empty)"
        lines = []
        for i, wf in enumerate(workflows, start=1):
            score = float(wf.get("retrieval_score", 0.0))
            lines.append(f"### Workflow #{i} (score={score:.3f})")
            lines.append(str(wf.get("workflow_text", "")).strip())
            lines.append("")
        return self._truncate_text("\n".join(lines).strip(), self.history_char_limit)

    def _format_exemplar_block(self, exemplars):
        if not exemplars:
            return "(empty)"
        lines = []
        for i, ex in enumerate(exemplars, start=1):
            lines.append(f"### Exemplar #{i} [{ex.get('source', 'unknown')}] (score={float(ex.get('score', 0.0)):.3f})")
            lines.append(f"Query: {ex.get('query', '')}")
            lines.append("Trajectory:")
            lines.append(str(ex.get("trajectory", "")).strip())
            lines.append("")
        return self._truncate_text("\n".join(lines).strip(), self.history_char_limit)

    def _build_generation_prompt(self, prompt, workflows, exemplars):
        workflow_block = self._format_workflow_block(workflows)
        exemplar_block = self._format_exemplar_block(exemplars)
        return (
            "You are a large language model trained to solve tasks with reusable workflows. "
            "First adapt relevant workflows, then follow the current input requirements.\n\n"
            "## Retrieved Workflows\n"
            f"{workflow_block}\n\n"
            "## Concrete Exemplars\n"
            f"{exemplar_block}\n\n"
            "## Current Input\n"
            f"{prompt}\n"
        )

    def retrieve_memory(self, task, **kwargs):
        task_name = task.name
        query = str(kwargs.get("query", "")).strip()
        workflows = self._retrieve_workflows(task_name, query)
        exemplars = self._retrieve_exemplars(task_name, query)
        return {
            "query": query,
            "profile": self._task_profile(task_name),
            "workflows": workflows,
            "exemplars": exemplars,
        }

    def run_trial(self, task, entry, inputs, timer, update_memory=True):
        if not self._supports_interactive_task(task):
            return None

        task_name = task.name
        if self._is_alfworld_task(task_name):
            step_prompt_builder = self._build_alfworld_step_prompt
        else:
            return None

        prompt = task.build_prompt(entry)
        query = task.get_query(entry, inputs)

        with timer.track(f"{self.name}/{task_name}/retrieve"):
            memory = self.retrieve_memory(task, query=query, entry=entry, **inputs)

        workflows = list(memory.get("workflows", []))
        exemplars = list(memory.get("exemplars", []))
        manual_fewshots = list(inputs.get("fewshots", []) or [])
        step_rows = []
        current_observation = task.start_episode(entry)
        initial_observation = current_observation
        max_steps = max(1, int(getattr(task, "max_steps", 20)))

        for step_idx in range(max_steps):
            generation_prompt = step_prompt_builder(
                task_prompt=prompt,
                current_observation=current_observation,
                step_rows=step_rows,
                workflows=workflows,
                exemplars=exemplars,
                manual_fewshots=manual_fewshots,
                step_idx=step_idx + 1,
                max_steps=max_steps,
                initial_observation=initial_observation,
            )
            with timer.track(f"{self.name}/{task_name}/generate"):
                raw_output = llm_generate(
                    model_name=self.generation_model_name,
                    prompt=generation_prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                )
            action = self._extract_single_action(task, raw_output)
            if not action:
                break
            step_result = task.step_episode(
                action,
                raw_action=action,
                meta={"agent_name": "awm_online"},
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

        if update_memory:
            with timer.track(f"{self.name}/{task_name}/update"):
                self.update_memory(task, model_output, record=record, entry=entry, **inputs)
        return score

    def generate(self, task, prompt, **kwargs):
        memory = kwargs.get("memory") or {}
        workflows = memory.get("workflows", [])
        exemplars = memory.get("exemplars", [])

        generation_prompt = self._build_generation_prompt(
            prompt=prompt,
            workflows=workflows,
            exemplars=exemplars,
        )
        return llm_generate(
            model_name=self.generation_model_name,
            prompt=generation_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

    def update_memory(self, task, output, **kwargs):
        task_name = task.name
        record = kwargs.get("record") or {}
        question = str(record.get("question") or kwargs.get("query") or "").strip()
        output_text = str(record.get("model_output", output))
        feedback = str(record.get("feedback", ""))
        score = record.get("score")
        qid = str(record.get("qid", "")).strip()

        self._task_sample_counter[task_name] = self._task_sample_counter.get(task_name, 0) + 1
        is_success = feedback.lower() == "success"

        sample = {
            "qid": qid,
            "question": question,
            "model_output": output_text,
            "feedback": feedback,
            "score": score,
            "parsed_actions": record.get("parsed_actions", []),
            "executed_actions": record.get("executed_actions", []),
            "embedding": self._encode_text(question),
        }
        sample["trajectory"] = self._sample_trajectory_text(sample)

        induced_workflow = ""
        new_workflow_record = None
        induce_triggered = False
        if is_success:
            self._task_success_counter[task_name] = self._task_success_counter.get(task_name, 0) + 1
            self._task_success_examples.setdefault(task_name, []).append(sample)

            # success-only, online update: each successful sample can induce a new workflow record.
            if self._task_success_counter[task_name] % self.induce_steps == 0:
                induce_triggered = True
                induced_workflow = self._induce_workflow(task_name, sample)
                if induced_workflow.strip():
                    new_workflow_record = self._new_workflow_record(task_name, induced_workflow, sample)
                    if new_workflow_record is not None:
                        self._task_workflow_records.setdefault(task_name, []).append(new_workflow_record)

        workflow_count = len(self._task_workflow_records.get(task_name, []))

        sample_event = {
            "task": task_name,
            **record,
            "memory_event": {
                "type": "sample_observed",
                "is_success": is_success,
                "workflow_count": workflow_count,
                "profile": self._task_profile(task_name),
                "sample_counter": self._task_sample_counter[task_name],
                "success_counter": self._task_success_counter.get(task_name, 0),
                "induce_triggered": induce_triggered,
                "induce_steps": self.induce_steps,
            },
        }
        self.memory.append(sample_event)

        if new_workflow_record is not None:
            self.memory.append(
                {
                    "task": task_name,
                    "qid": qid,
                    "question": question,
                    "memory_event": {
                        "type": "workflow_induced",
                        "workflow_id": new_workflow_record.get("record_id", ""),
                        "profile": new_workflow_record.get("profile", ""),
                        "workflow_text": new_workflow_record.get("workflow_text", ""),
                        "source_feedback": new_workflow_record.get("source_feedback", ""),
                        "source_score": new_workflow_record.get("source_score"),
                    },
                }
            )

    def reset_task_state(self, task):
        task_name = task.name
        self._task_workflow_records.setdefault(task_name, [])
        self._task_success_examples.setdefault(task_name, [])
        self._task_sample_counter.setdefault(task_name, 0)
        self._task_success_counter.setdefault(task_name, 0)

        if not self._task_workflow_records[task_name] and self.workflow_init:
            init_record = self._new_init_workflow_record(task_name)
            if init_record is not None:
                self._task_workflow_records[task_name].append(init_record)
        return None

    def restore_state_from_memory(self, task):
        """Rebuild induced-workflow pool, success-example pool, and counters
        from preloaded jsonl records so --resume preserves retrieval fidelity."""
        task_name = task.name
        task_rows = [r for r in self.memory if r.get("task") == task_name]
        if not task_rows:
            return

        sample_counter = 0
        success_counter = 0
        success_examples = []
        workflow_records = []

        for row in task_rows:
            event = row.get("memory_event") or {}
            ev_type = event.get("type")
            if ev_type == "sample_observed":
                sample_counter += 1
                is_success = bool(event.get("is_success"))
                if is_success:
                    success_counter += 1
                    question = str(row.get("question") or "").strip()
                    sample = {
                        "qid": str(row.get("qid", "")).strip(),
                        "question": question,
                        "model_output": str(row.get("model_output", "")),
                        "feedback": str(row.get("feedback", "")),
                        "score": row.get("score"),
                        "parsed_actions": row.get("parsed_actions", []),
                        "executed_actions": row.get("executed_actions", []),
                        "embedding": None,  # lazily re-embedded on first retrieval
                    }
                    sample["trajectory"] = self._sample_trajectory_text(sample)
                    success_examples.append(sample)
            elif ev_type == "workflow_induced":
                workflow_text = str(event.get("workflow_text", "")).strip()
                if not workflow_text:
                    continue
                workflow_records.append(
                    {
                        "record_id": event.get("workflow_id", str(uuid.uuid4())),
                        "task": task_name,
                        "workflow_text": workflow_text,
                        "query_text": str(row.get("question", "")).strip(),
                        "source_qid": str(row.get("qid", "")).strip(),
                        "source_score": event.get("source_score"),
                        "source_feedback": event.get("source_feedback", ""),
                        "created_at": "",
                        "profile": event.get("profile", self._task_profile(task_name)),
                        "embedding": self._encode_text(workflow_text),
                    }
                )

        self._task_sample_counter[task_name] = sample_counter
        self._task_success_counter[task_name] = success_counter
        self._task_success_examples[task_name] = success_examples
        self._task_workflow_records[task_name] = workflow_records
        print(
            f"[AWM Resume] {task_name}: restored {sample_counter} samples "
            f"({success_counter} success), {len(workflow_records)} workflows"
        )
