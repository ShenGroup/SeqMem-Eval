import json
from pathlib import Path

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.math_answer_utils import normalized_gold_answer, normalized_last_boxed


class OmniMATHTask(BaseTask):
    name = "Omni-MATH"
    prompt_key = "omni_math"

    def __init__(self, prompt_file=None, data_file=None):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        self.data_file = (
            Path(data_file) if data_file else root / "data" / "Omni-MATH" / "test.jsonl"
        )
        self._entries = None

    def _load(self):
        if self._entries is not None:
            return
        if not self.data_file.exists():
            raise RuntimeError(f"Omni-MATH file not found: {self.data_file}")

        entries = []
        with self.data_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                row["_qid"] = str(row.get("id", idx))
                entries.append(row)
        self._entries = entries

    def _question(self, item):
        return item.get("problem", "") or item.get("question", "")

    def _solution(self, item):
        return item.get("solution", "") or item.get("Solution", "")

    def _gold_final_answer(self, item):
        return normalized_gold_answer(
            answer_text=item.get("answer", ""),
            fallback_solution_text=self._solution(item),
        )

    def iter_entries(self):
        self._load()
        for item in self._entries:
            yield item

    def total_entries(self):
        self._load()
        return len(self._entries)

    def build_prompt(self, entry):
        return get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._question(entry),
        )

    def build_inputs(self, entry):
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question(entry),
            "answer": str(entry.get("answer", "")),
            "solution": self._solution(entry),
            "gold_final_answer": self._gold_final_answer(entry),
        }

    def evaluate_entry(self, output, entry):
        pred = normalized_last_boxed(output)
        gold = self._gold_final_answer(entry)
        return 1.0 if pred and gold and pred == gold else 0.0

    def build_memory_record(self, entry, output, feedback, score):
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question(entry),
            "gold_final_answer": self._gold_final_answer(entry),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
        }

    def get_prompt(self):
        raise NotImplementedError("Omni-MATH uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("Omni-MATH uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("Omni-MATH uses evaluate_entry(output, entry).")
