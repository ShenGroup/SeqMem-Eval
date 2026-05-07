from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from prompts.gmemory import (
    PROMPTS,
    format_task_context,
    format_task_prompt_with_insights,
)
from methods.base_method import BaseMethod
from methods.gmemory_core import GMemoryEngine
from utils.llm_client import generate as llm_generate


class AutoGenGMemoryOnline(BaseMethod):
    name = "AutoGen-GMemory"

    def __init__(
        self,
        top_k=1,
        successful_topk=None,
        failed_topk=1,
        insights_topk=3,
        threshold=0.0,
        use_projector=False,
        hop=1,
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        curator_model_name=None,
        temperature=0.0,
        max_new_tokens=1024,
        persist_dir=None,
        start_insights_threshold=5,
        rounds_per_insights=5,
        insights_point_num=5,
        max_num_rules=20,
        fidelity="high",
        snapshot_every=100,
        reset_persist=False,
        **kwargs,
    ):
        super().__init__()
        self.successful_topk = max(
            0, int(successful_topk if successful_topk is not None else top_k)
        )
        self.failed_topk = max(0, int(failed_topk))
        self.insights_topk = max(0, int(insights_topk))
        self.threshold = float(threshold)
        self.use_projector = bool(use_projector)
        self.hop = max(1, int(hop))

        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.curator_model_name = curator_model_name or generation_model_name
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)

        self.start_insights_threshold = max(1, int(start_insights_threshold))
        self.rounds_per_insights = max(1, int(rounds_per_insights))
        self.insights_point_num = max(1, int(insights_point_num))
        self.max_num_rules = max(1, int(max_num_rules))
        self.fidelity = str(fidelity or "high").strip().lower()
        self.snapshot_every = max(1, int(snapshot_every))

        self.persist_dir = Path(persist_dir) if persist_dir else Path("outputs") / ".gmemory"

        if reset_persist and self.persist_dir.exists():
            shutil.rmtree(self.persist_dir, ignore_errors=True)
            print(f"[AutoGen-GMemory] reset persist dir: {self.persist_dir}")

        self._task_engines: dict[str, GMemoryEngine] = {}
        self._task_pending_context: dict[str, dict] = {}
        self._task_recent_outputs: dict[str, list[str]] = {}
        self._task_action_history: dict[str, list[str]] = {}
        self._task_snapshot_counter: dict[str, int] = {}
        self._task_system_suffix_cache: dict[str, str] = {}

    _PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"
    _TASK_SYSTEM_SUFFIX_FILES: dict[str, Path] = {
        "ALFWorld": _PROMPTS_ROOT / "alfworld" / "system_instruction.txt",
        "APIBench-HF": _PROMPTS_ROOT / "apibench" / "system_instruction.txt",
        "APIBench-TF": _PROMPTS_ROOT / "apibench" / "system_instruction.txt",
        "APIBench-TH": _PROMPTS_ROOT / "apibench" / "system_instruction.txt",
    }

    def _task_system_suffix(self, task) -> str:
        task_name = getattr(task, "name", "") or ""
        if task_name in self._task_system_suffix_cache:
            return self._task_system_suffix_cache[task_name]
        path = self._TASK_SYSTEM_SUFFIX_FILES.get(task_name)
        text = ""
        if path is not None and path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                text = ""
        self._task_system_suffix_cache[task_name] = text
        return text

    def _task_persist_dir(self, task_name: str) -> Path:
        safe_task = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(task_name or "unknown"))
        return self.persist_dir / safe_task / self.name

    def _task_snapshot_root(self, task_name: str) -> Path:
        safe_task = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(task_name or "unknown"))
        return self.persist_dir / "_snapshots" / safe_task / self.name

    def _snapshot_state(self, task_name: str, sample_idx: int, final: bool = False) -> str:
        src = self._task_persist_dir(task_name)
        if not src.exists():
            return ""
        snap_root = self._task_snapshot_root(task_name)
        snap_root.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        tag = "final" if final else "step"
        dst = snap_root / f"{tag}_{int(sample_idx):06d}_{stamp}"
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        return str(dst)

    def _llm_call(self, model_name: str, prompt: str, temperature: float, max_new_tokens: int) -> str:
        return llm_generate(
            model_name=model_name,
            prompt=prompt,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
        )

    def _engine_for_task(self, task_name: str) -> GMemoryEngine:
        if task_name in self._task_engines:
            return self._task_engines[task_name]
        engine = GMemoryEngine(
            persist_dir=self._task_persist_dir(task_name),
            embedding_model_name=self.embedding_model_name,
            llm_call=self._llm_call,
            curator_model_name=self.curator_model_name,
            hop=self.hop,
            start_insights_threshold=self.start_insights_threshold,
            rounds_per_insights=self.rounds_per_insights,
            insights_point_num=self.insights_point_num,
        )
        self._task_engines[task_name] = engine
        return engine

    def _extract_tag_content(self, text: str, tag_name: str) -> str:
        s = str(text or "")
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"
        if start_tag not in s or end_tag not in s:
            return ""
        try:
            return s.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
        except Exception:
            return ""

    def _build_memory_fewshots(self, successful_trajectories: list[dict]) -> list[str]:
        shots = []
        for traj in successful_trajectories:
            shots.append(
                format_task_context(
                    task_description=str(traj.get("task_description", "")),
                    task_traj=str(traj.get("task_trajectory", "")),
                    key_steps=str(traj.get("extra_fields", {}).get("key_steps", "")),
                )
            )
        return shots

    def _build_output_contract(self, action_mode: bool) -> str:
        # The <trajectory>...</trajectory> wrapper is NOT stored in the graph
        # (executed action/observation pairs are, see _build_executed_trajectory),
        # and asking the model to emit it caused action-vs-reasoning parsing
        # collisions on reasoning models. Just point at the task's own output
        # format and let the solver emit the action directly.
        return "Follow the output format specified by the task above."

    def _build_agent_prompt(
        self,
        task,
        base_prompt: str,
        few_shots: list[str],
        successful_trajectories: list[dict],
        insights: list[str],
        role: str,
    ) -> str:
        memory_few_shots = self._build_memory_fewshots(successful_trajectories)
        task_prompt = format_task_prompt_with_insights(
            few_shots=few_shots,
            memory_few_shots=memory_few_shots,
            insights=insights,
            task_description=base_prompt,
        )
        output_contract = self._build_output_contract(bool(getattr(task, "action_mode", False)))

        role_system = (
            PROMPTS.solver_system_prompt
            if role == "solver"
            else PROMPTS.ground_truth_system_prompt
        )
        task_system_suffix = self._task_system_suffix(task)
        system_block = role_system
        if task_system_suffix:
            system_block = f"{role_system}\n\n{task_system_suffix}"
        return (
            f"[SYSTEM]\n{system_block}\n\n"
            f"[USER]\n{task_prompt}\n\n{output_contract}\n"
        )

    def _generate_with_retries(self, prompt: str, model_name: str, tries: int = 3) -> str:
        output = ""
        for _ in range(max(1, int(tries))):
            output = self._llm_call(
                model_name=model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            if str(output or "").strip():
                break
        return str(output or "")

    def _solver_stuck(self, task_name: str, current_output: str) -> bool:
        recent = self._task_recent_outputs.setdefault(task_name, [])
        cur = str(current_output or "").strip()
        if not cur:
            return True
        stuck = len(recent) >= 2 and cur == recent[-1] and cur == recent[-2]
        recent.append(cur)
        if len(recent) > 6:
            del recent[:-6]
        return stuck

    def _build_executed_trajectory(self, record: dict, model_output: str) -> str:
        eval_trace = (
            record.get("eval_debug", {}).get("eval_trace", [])
            if isinstance(record, dict)
            else []
        )
        lines: list[str] = []
        if isinstance(eval_trace, list) and eval_trace:
            for item in eval_trace:
                action = str(item.get("executed_action", "")).strip()
                observation = str(item.get("observation", "")).strip()
                if action:
                    lines.append(f"> {action}")
                if observation:
                    lines.append(observation)
            if lines:
                return "\n".join(lines).strip()

        actions = []
        if isinstance(record, dict):
            actions = list(record.get("executed_actions", []) or record.get("parsed_actions", []) or [])
        if actions:
            lines = [f"> {str(a).strip()}" for a in actions if str(a).strip()]
            last_observation = str(record.get("last_observation", "")).strip()
            if last_observation:
                lines.append(last_observation)
            if lines:
                return "\n".join(lines).strip()

        trajectory = self._extract_tag_content(model_output, "trajectory")
        return trajectory if trajectory else str(model_output)

    def _solver_stuck_action(self, task_name: str, current_action: str) -> bool:
        hist = self._task_action_history.setdefault(task_name, [])
        action = str(current_action or "").strip().lower()
        if not action:
            return True
        stuck = len(hist) >= 2 and action == hist[-1] and action == hist[-2]
        hist.append(action)
        if len(hist) > 12:
            del hist[:-12]
        return stuck

    def _extract_single_action(self, task, output: str) -> str:
        if hasattr(task, "_extract_actions"):
            try:
                actions = list(task._extract_actions(output))
            except Exception:
                actions = []
            if actions:
                return str(actions[0]).strip().lower()
        text = str(output or "")
        trajectory = self._extract_tag_content(text, "trajectory")
        if trajectory:
            # Reasoning lives inside <trajectory>, the action outside. Strip
            # the reasoning block before scanning for the action line; fall
            # back to the trajectory content only when nothing follows.
            stripped = re.sub(r"<trajectory>.*?</trajectory>", "", text, flags=re.DOTALL).strip()
            source = stripped if stripped else trajectory
        else:
            source = text
        for line in source.splitlines():
            candidate = str(line).strip()
            if not candidate:
                continue
            candidate = re.sub(r"^>\s*", "", candidate).strip()
            if not candidate:
                continue
            low = candidate.lower()
            if low.startswith("think"):
                continue
            return low
        return ""

    def _build_step_prompt_text(self, base_prompt: str, current_observation: str, recent_steps: list[dict],
                                initial_observation: str = "") -> str:
        lines = [str(base_prompt).strip()]
        init_obs = str(initial_observation or "").strip()
        if init_obs:
            lines.append("Initial observation:")
            lines.append(init_obs)
        obs = str(current_observation or "").strip()
        if obs and obs != init_obs:
            lines.append("Current observation:")
            lines.append(obs)
        if recent_steps:
            lines.append("Recent interaction history:")
            for row in recent_steps:
                a = str(row.get("action", "")).strip()
                o = str(row.get("observation", "")).strip()
                if a:
                    lines.append(f"- Action: {a}")
                if o:
                    lines.append(f"  Observation: {o}")
        lines.append("Output exactly one next executable action.")
        return "\n".join(lines).strip()

    def _build_trial_model_output(self, step_rows: list[dict], success: bool) -> str:
        lines = []
        for row in step_rows:
            action = str(row.get("action", "")).strip()
            observation = str(row.get("observation", "")).strip()
            if action:
                lines.append(f"Action: {action}")
            if observation:
                lines.append(observation)
        return "\n".join(lines).strip()

    def retrieve_memory(self, task, **kwargs):
        task_name = task.name
        engine = self._engine_for_task(task_name)

        query = str(kwargs.get("query", "")).strip()
        if not query:
            query = str(kwargs.get("question", "")).strip()

        successful, failed, insights = engine.retrieve_memory(
            query_task=query,
            successful_topk=self.successful_topk,
            failed_topk=self.failed_topk,
            insight_topk=self.insights_topk,
            threshold=self.threshold,
        )
        memory = {
            "query": query,
            "successful_trajectories": successful,
            "failed_trajectories": failed,
            "insights": insights,
        }
        self._task_pending_context[task_name] = {
            "query": query,
            "memory": memory,
            "entry": kwargs.get("entry"),
        }
        return memory

    def run_trial(self, task, entry, inputs, timer, update_memory=True):
        if self.fidelity != "high":
            return None
        if not all(hasattr(task, name) for name in ("start_episode", "step_episode", "finish_episode")):
            return None

        task_name = task.name
        engine = self._engine_for_task(task_name)
        prompt = task.build_prompt(entry)
        query = task.get_query(entry, inputs)

        with timer.track(f"{self.name}/{task_name}/retrieve"):
            memory = self.retrieve_memory(task, query=query, entry=entry, **inputs)

        successful_trajectories = list(memory.get("successful_trajectories", []))
        raw_insights = list(memory.get("insights", []))
        few_shots = list(inputs.get("fewshots", []) or [])

        solver_insights = list(raw_insights)
        ground_truth_insights = list(raw_insights)
        if self.use_projector and raw_insights:
            solver_insights = engine.project_insights(raw_insights, role="solver")
            ground_truth_insights = engine.project_insights(raw_insights, role="ground truth agent")

        self._task_action_history.setdefault(task_name, [])
        current_observation = task.start_episode(entry)
        initial_observation = current_observation
        step_rows = []
        agent_messages = []
        used_agent = "solver"

        max_steps = max(1, int(getattr(task, "max_steps", 30)))
        for step_idx in range(max_steps):
            step_prompt_text = self._build_step_prompt_text(
                prompt, current_observation, step_rows,
                initial_observation=initial_observation,
            )
            solver_prompt = self._build_agent_prompt(
                task=task,
                base_prompt=step_prompt_text,
                few_shots=few_shots,
                successful_trajectories=successful_trajectories,
                insights=solver_insights,
                role="solver",
            )
            with timer.track(f"{self.name}/{task_name}/generate"):
                solver_output = self._generate_with_retries(solver_prompt, self.generation_model_name)
            action = self._extract_single_action(task, solver_output)
            raw_output = solver_output
            agent_name = "solver"
            system_instruction = PROMPTS.solver_system_prompt

            if self._solver_stuck_action(task_name, action):
                ground_prompt = self._build_agent_prompt(
                    task=task,
                    base_prompt=step_prompt_text,
                    few_shots=few_shots,
                    successful_trajectories=successful_trajectories,
                    insights=ground_truth_insights,
                    role="ground_truth",
                )
                with timer.track(f"{self.name}/{task_name}/generate"):
                    ground_output = self._generate_with_retries(ground_prompt, self.generation_model_name)
                ground_action = self._extract_single_action(task, ground_output)
                if ground_action:
                    action = ground_action
                    raw_output = ground_output
                    agent_name = "ground_truth"
                    system_instruction = PROMPTS.ground_truth_system_prompt
                    used_agent = "ground_truth"

            action = str(action or "").strip()
            if not action:
                break

            agent_message = {
                "step": step_idx + 1,
                "agent_name": agent_name,
                "system_instruction": system_instruction,
                "user_instruction": step_prompt_text,
                "message": action,
                "raw_output": raw_output,
            }
            agent_messages.append(agent_message)
            step_result = task.step_episode(
                action,
                raw_action=action,
                meta={"agent_name": agent_name},
            )
            row = {
                "step": step_idx + 1,
                "action": str(step_result.get("executed_action", action)),
                "observation": str(step_result.get("raw_observation", step_result.get("observation", ""))),
                "reward": str(step_result.get("reward", "")),
                "done": bool(step_result.get("done", False)),
                "won": bool(step_result.get("won", False)),
                "agent_name": agent_name,
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

        state_chain = []
        for row, msg in zip(step_rows, agent_messages):
            state_chain.append(
                {
                    "step": int(row.get("step", 0)),
                    "action": row.get("action", ""),
                    "observation": row.get("observation", ""),
                    "reward": row.get("reward", ""),
                    "done": bool(row.get("done", False)),
                    "won": bool(row.get("won", False)),
                    "agent_nodes": [msg],
                }
            )
        record["agent_messages"] = agent_messages
        record["state_chain"] = state_chain
        record["trial_feedback"] = {
            "success": bool(score >= 1.0),
            "steps_executed": int(final.get("steps_executed", 0)),
            "last_observation": str(final.get("last_observation", "")),
        }

        ctx = self._task_pending_context.setdefault(task_name, {})
        ctx.update(
            {
                "query": query,
                "prompt": str(prompt),
                "few_shots": few_shots,
                "used_agent": used_agent,
                "retrieved_success": len(successful_trajectories),
                "retrieved_fail": len(memory.get("failed_trajectories", [])),
                "retrieved_insights": len(raw_insights),
                "projected_solver_insights": len(solver_insights),
                "projected_ground_truth_insights": len(ground_truth_insights),
            }
        )
        if update_memory:
            with timer.track(f"{self.name}/{task_name}/update"):
                self.update_memory(task, model_output, record=record, entry=entry, **inputs)
        return score

    def generate(self, task, prompt, **kwargs):
        task_name = task.name
        engine = self._engine_for_task(task_name)

        memory = kwargs.get("memory") or {}
        successful_trajectories = list(memory.get("successful_trajectories", []))
        raw_insights = list(memory.get("insights", []))
        few_shots = list(kwargs.get("fewshots", []) or [])

        solver_insights = list(raw_insights)
        ground_truth_insights = list(raw_insights)
        if self.use_projector and raw_insights:
            solver_insights = engine.project_insights(raw_insights, role="solver")
            ground_truth_insights = engine.project_insights(
                raw_insights,
                role="ground truth agent",
            )

        solver_prompt = self._build_agent_prompt(
            task=task,
            base_prompt=str(prompt),
            few_shots=few_shots,
            successful_trajectories=successful_trajectories,
            insights=solver_insights,
            role="solver",
        )
        solver_output = self._generate_with_retries(solver_prompt, self.generation_model_name)

        use_ground_truth = self._solver_stuck(task_name, solver_output)
        final_output = solver_output
        used_agent = "solver"
        if use_ground_truth:
            ground_prompt = self._build_agent_prompt(
                task=task,
                base_prompt=str(prompt),
                few_shots=few_shots,
                successful_trajectories=successful_trajectories,
                insights=ground_truth_insights,
                role="ground_truth",
            )
            ground_output = self._generate_with_retries(
                ground_prompt,
                self.generation_model_name,
            )
            if str(ground_output or "").strip():
                final_output = ground_output
                used_agent = "ground_truth"

        ctx = self._task_pending_context.setdefault(task_name, {})
        ctx.update(
            {
                "prompt": str(prompt),
                "few_shots": few_shots,
                "used_agent": used_agent,
                "retrieved_success": len(successful_trajectories),
                "retrieved_fail": len(memory.get("failed_trajectories", [])),
                "retrieved_insights": len(raw_insights),
                "projected_solver_insights": len(solver_insights),
                "projected_ground_truth_insights": len(ground_truth_insights),
            }
        )
        return final_output

    def update_memory(self, task, output, **kwargs):
        task_name = task.name
        engine = self._engine_for_task(task_name)

        record = kwargs.get("record") or {}
        ctx = self._task_pending_context.get(task_name, {})
        query = str(ctx.get("query") or kwargs.get("query") or record.get("question") or "").strip()
        if not query:
            return

        model_output = str(record.get("model_output", output))
        feedback = str(record.get("feedback", "")).lower()
        score = float(record.get("score", 0.0))
        label = feedback == "success" or score >= 1.0

        trajectory = self._build_executed_trajectory(record, model_output)

        memory_record = {
            "task_name": task_name,
            "task_main": query,
            "task_description": str(ctx.get("prompt", "")),
            "task_trajectory": str(trajectory),
            "label": bool(label),
            "extra_fields": {
                "raw_output": model_output,
                "used_agent": ctx.get("used_agent", "solver"),
                "retrieved_success": int(ctx.get("retrieved_success", 0)),
                "retrieved_fail": int(ctx.get("retrieved_fail", 0)),
                "retrieved_insights": int(ctx.get("retrieved_insights", 0)),
                "state_chain": record.get("state_chain", []),
                "agent_messages": record.get("agent_messages", []),
                "trial_feedback": record.get("trial_feedback", {}),
            },
        }

        stored = engine.add_memory(memory_record, max_num_rules=self.max_num_rules)
        engine.backward(bool(label))

        self._task_snapshot_counter[task_name] = self._task_snapshot_counter.get(task_name, 0) + 1
        sample_idx = int(self._task_snapshot_counter[task_name])
        snapshot_path = ""
        snapshot_triggered = sample_idx % self.snapshot_every == 0
        if snapshot_triggered:
            try:
                snapshot_path = self._snapshot_state(task_name, sample_idx=sample_idx, final=False)
            except Exception:
                snapshot_path = ""

        current_insights = [
            str(row.get("rule", "")).strip()
            for row in engine.insights_layer.insights_memory
            if str(row.get("rule", "")).strip()
        ]
        enriched = {
            "task": task_name,
            **record,
            "memory": {
                "insights": current_insights[: self.insights_topk] if self.insights_topk > 0 else [],
                "stored_task_main": stored.get("task_main", ""),
                "stored_label": bool(stored.get("label", False)),
                "stored_key_steps": stored.get("extra_fields", {}).get("key_steps", ""),
                "stored_fail_reason": stored.get("extra_fields", {}).get("fail_reason", ""),
            },
            "memory_meta": {
                "retrieved_success": int(ctx.get("retrieved_success", 0)),
                "retrieved_fail": int(ctx.get("retrieved_fail", 0)),
                "retrieved_insights": int(ctx.get("retrieved_insights", 0)),
                "used_agent": ctx.get("used_agent", "solver"),
                "use_projector": bool(self.use_projector),
                "hop": int(self.hop),
                "threshold": float(self.threshold),
                "memory_size": int(engine.memory_size),
                "insights_size": len(engine.insights_layer.insights_memory),
                "persist_dir": str(self._task_persist_dir(task_name)),
                "snapshot_every": int(self.snapshot_every),
                "snapshot_triggered": bool(snapshot_triggered),
                "snapshot_path": snapshot_path,
            },
        }
        self.memory.append(enriched)

    def reset_task_state(self, task):
        task_name = task.name
        self._engine_for_task(task_name)
        self._task_pending_context[task_name] = {}
        self._task_recent_outputs[task_name] = []
        self._task_action_history[task_name] = []
        self._task_snapshot_counter.setdefault(task_name, 0)
        return None

    def finalize_task_state(self, task):
        task_name = task.name
        self._engine_for_task(task_name)
        sample_idx = int(self._task_snapshot_counter.get(task_name, 0))
        try:
            self._snapshot_state(task_name, sample_idx=sample_idx, final=True)
        except Exception:
            pass
        return None
