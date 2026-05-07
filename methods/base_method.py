from abc import ABC, abstractmethod

from utils.io import write_json, write_jsonl


class BaseMethod(ABC):
    name = "BaseMethod"

    def __init__(self):
        self.memory = []

    def _snapshot_record(self, record):
        # Avoid recursive expansion when exporting per-step memory snapshots.
        return {k: v for k, v in record.items() if k not in {"memory"}}

    def build_task_memory_snapshot(self, task_name):
        rows = [row for row in self.memory if row.get("task") == task_name]
        return [self._snapshot_record(row) for row in rows]

    def reset_task_state(self, task):
        # Optional hook for methods that keep per-task states.
        return None

    def restore_state_from_memory(self, task):
        """Rebuild internal state from preloaded self.memory records on resume.

        Called after self.memory has been populated from the existing JSONL.
        Override in methods that maintain derived state beyond self.memory
        (e.g. DC-RS cheatsheets, embeddings caches).
        """
        return None

    def get_timing_adjustments(self, task):
        # Optional hook for stage re-attribution.
        return {}

    def run_trial(self, task, entry, inputs, timer, update_memory=True):
        # Optional high-fidelity trial hook. Return float score when handled.
        return None

    def finalize_task_state(self, task):
        # Optional hook for end-of-task finalization.
        return None

    def export_eval_context(self, task, entry, memory):
        """Return a dict of extra per-sample context the runner should add to
        eval_log.jsonl. Default is empty. Methods whose prompt depends on
        state outside the retrieve_memory return (e.g. DC-RS cheatsheet,
        ExpeL rule library) should override this so holdout eval logs
        capture what actually went into the prompt."""
        return {}

    def export_task_memory(self, task_name, output_path):
        rows = [row for row in self.memory if row.get("task") == task_name]
        write_jsonl(output_path, rows)
        readable_path = output_path.with_name(f"{output_path.stem}_readable.json")
        write_json(readable_path, rows)

    @staticmethod
    def _format_memory_block(retrieved):
        if not retrieved:
            return "No retrieved memory."
        lines = []
        for idx, item in enumerate(retrieved, start=1):
            lines.append(
                f"[Memory {idx}]\n"
                f"Question: {item.get('question', '')}\n"
                f"Model Output: {item.get('model_output', '')}\n"
                f"Feedback: {item.get('feedback', '')}"
            )
        return "\n\n".join(lines)

    def build_memory_augmented_prompt(self, task_prompt, retrieved):
        memory_block = self._format_memory_block(retrieved)
        return (
            f"Retrieved memory:\n{memory_block}\n\n"
            f"Current problem:\n{task_prompt}\n"
        )

    @abstractmethod
    def retrieve_memory(self, task, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate(self, task, prompt, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_memory(self, task, output, **kwargs):
        raise NotImplementedError
