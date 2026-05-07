"""Baseline: pure LLM inference with the task's own build_prompt, no memory.

Used as the t=0 data point in holdout trajectories. For non-interactive tasks
(MATH, APIBench, ...), all three memory hooks are no-ops and generation is
one shot on `task.build_prompt(entry)`.

For interactive step-wise tasks (ALFWorld) Baseline overrides `run_trial` so
the model drives the env step-by-step. The per-step prompt mirrors what the
training step-wise methods build — shared ALFWorld system instruction +
static few-shots from `tasks/alfworld_fewshots.py` (these define the action
grammar, not memory) + task goal + recent trajectory — but omits every
memory-derived block (workflows / exemplars / rules / cheatsheets /
insights) since this is the no-memory baseline.
"""
import re
from pathlib import Path

from methods.base_method import BaseMethod
from utils.llm_client import generate as llm_generate


ROOT = Path(__file__).resolve().parents[1]
_ALFWORLD_SYSTEM_PROMPT_PATH = ROOT / "prompts" / "alfworld" / "system_instruction.txt"


class Baseline(BaseMethod):
    name = "Baseline"
    # Stateless single-shot generation → safe to dispatch samples concurrently
    # via runner's --max-parallel-samples thread pool. The runner additionally
    # gates on `not hasattr(task, "step_episode")`, so ALFWorld still runs
    # sequentially through `run_trial`.
    supports_parallel_samples = True

    def __init__(
        self,
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        temperature=0.0,
        max_new_tokens=2048,
        history_char_limit=8000,
        **_ignored_kwargs,
    ):
        super().__init__()
        self.generation_model_name = generation_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.history_char_limit = int(history_char_limit)
        self._alfworld_system_prompt = self._load_alfworld_system_prompt()
        self._last_alfworld_trace = []

    @staticmethod
    def _load_alfworld_system_prompt():
        try:
            return _ALFWORLD_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except Exception:
            return (
                "Interact with a household to solve a task. Imagine you are an "
                "intelligent agent in a household environment and your target is "
                "to perform actions to complete the task goal.\n"
                "Your response should use one of the following formats:\n\n"
                "Thought: <your thoughts>\nAction: <your next action>"
            )

    def retrieve_memory(self, task, **kwargs):
        return []

    def generate(self, task, prompt, **kwargs):
        return llm_generate(
            model_name=self.generation_model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

    def update_memory(self, task, output, **kwargs):
        return

    def _supports_interactive_task(self, task):
        return all(
            hasattr(task, name) for name in ("start_episode", "step_episode", "finish_episode")
        )

    @staticmethod
    def _is_alfworld_task(task_name):
        return str(task_name or "").lower().startswith("alfworld")

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
        for line in text.splitlines():
            line = str(line).strip()
            if not line or line.startswith("```") or line.startswith("#"):
                continue
            if line.startswith(">"):
                line = re.sub(r"^>\s*", "", line).strip()
            low = line.lower()
            if low.startswith("think:") or low.startswith("thought:") or low.startswith("observation:"):
                continue
            if low.startswith("action:"):
                return line.split(":", 1)[1].strip()
            return line
        return ""

    def _truncate(self, text):
        limit = max(1, int(self.history_char_limit * 3))
        s = str(text or "")
        if len(s) <= limit:
            return s
        head = limit // 2
        tail = limit - head - 20
        return s[:head] + "\n...[truncated]...\n" + s[-tail:]

    def _build_alfworld_step_prompt(
        self, task_prompt, current_observation, step_rows, step_idx, max_steps,
        manual_fewshots=None, initial_observation=None,
    ):
        history_lines = []
        init_obs = str(initial_observation or "").strip()
        if init_obs:
            history_lines.append(init_obs)
        for row in step_rows:
            history_lines.append(f"Action: {row.get('action', '')}")
            obs = str(row.get("observation", "")).strip()
            if obs:
                history_lines.append(obs)
        history_block = "\n".join(history_lines).strip() if history_lines else "(none)"
        fewshot_text = "\n\n".join(
            str(x).strip() for x in (manual_fewshots or []) if str(x).strip()
        )
        lines = [self._alfworld_system_prompt]
        if fewshot_text:
            lines.extend(["", "## Few-shot Examples", fewshot_text])
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
        return "\n".join(lines)

    def run_trial(self, task, entry, inputs, timer, update_memory=True):
        if not self._supports_interactive_task(task):
            return None
        task_name = task.name
        if not self._is_alfworld_task(task_name):
            return None

        prompt = task.build_prompt(entry)
        manual_fewshots = list((inputs or {}).get("fewshots", []) or [])
        current_obs = task.start_episode(entry)
        initial_obs = current_obs
        max_steps = max(1, int(getattr(task, "max_steps", 30)))
        step_rows = []
        self._last_alfworld_trace = []
        broke_on_empty = False

        for step_idx in range(max_steps):
            step_prompt = self._build_alfworld_step_prompt(
                task_prompt=prompt,
                current_observation=current_obs,
                step_rows=step_rows,
                step_idx=step_idx + 1,
                max_steps=max_steps,
                manual_fewshots=manual_fewshots,
                initial_observation=initial_obs,
            )
            with timer.track(f"{self.name}/{task_name}/generate"):
                raw_output = llm_generate(
                    model_name=self.generation_model_name,
                    prompt=step_prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                )
            action = self._extract_single_action(task, raw_output)
            if not action:
                self._last_alfworld_trace.append({
                    "step": step_idx + 1,
                    "prompt": step_prompt,
                    "raw_output": str(raw_output or ""),
                    "action": "",
                    "observation": str(current_obs or ""),
                    "reward": None,
                    "done": False,
                    "broke_on_empty_action": True,
                })
                broke_on_empty = True
                break
            step_out = task.step_episode(action, raw_action=raw_output)
            new_obs = step_out.get("observation", "")
            self._last_alfworld_trace.append({
                "step": step_idx + 1,
                "prompt": step_prompt,
                "raw_output": str(raw_output or ""),
                "action": action,
                "executed_action": step_out.get("executed_action", action),
                "observation": str(new_obs or ""),
                "reward": str(step_out.get("reward", "")),
                "done": bool(step_out.get("done", False)),
                "won": bool(step_out.get("won", False)),
            })
            current_obs = new_obs
            step_rows.append({"action": action, "observation": str(new_obs or "")})
            if step_out.get("done", False):
                break

        final = task.finish_episode()
        try:
            score = float(final.get("score", 0.0)) if isinstance(final, dict) else 0.0
        except (TypeError, ValueError):
            score = 0.0
        self._last_alfworld_meta = {
            "task_goal": str(prompt).strip(),
            "gamefile": str(entry.get("gamefile", "")),
            "initial_observation": str(initial_obs or ""),
            "fewshots_count": len(manual_fewshots),
            "steps_executed": int(final.get("steps_executed", 0)) if isinstance(final, dict) else 0,
            "broke_on_empty_action": broke_on_empty,
            "score": score,
            "success": bool(final.get("success", False)) if isinstance(final, dict) else False,
        }
        return score

    def export_eval_context(self, task, entry, memory):
        if not self._is_alfworld_task(getattr(task, "name", "")):
            return {}
        meta = getattr(self, "_last_alfworld_meta", None) or {}
        trace = list(getattr(self, "_last_alfworld_trace", []) or [])
        if not trace and not meta:
            return {}
        return {"alfworld_meta": meta, "alfworld_trace": trace}
