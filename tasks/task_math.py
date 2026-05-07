from pathlib import Path

import pandas as pd

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.math_answer_utils import normalized_last_boxed


class MATHTask(BaseTask):
    name = "MATH"
    prompt_key = "math"

    def __init__(self, prompt_file=None, data_dir=None):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(data_dir) if data_dir else root / "data" / "MATH"
        self._entries = None

    def _load(self):
        if self._entries is not None:
            return
        if not self.data_dir.exists():
            raise RuntimeError(f"MATH data directory not found: {self.data_dir}")

        entries = []
        test_files = sorted(self.data_dir.glob("*/test-*.parquet"))
        if not test_files:
            raise RuntimeError(
                f"No MATH test parquet files found under: {self.data_dir}"
            )

        for test_file in test_files:
            subject = test_file.parent.name
            frame = pd.read_parquet(test_file)
            for idx, item in enumerate(frame.to_dict(orient="records")):
                row = dict(item)
                row["_subject"] = subject
                row["_qid"] = f"{subject}-{idx}"
                entries.append(row)

        self._entries = entries

    def _question(self, item):
        return item.get("problem", "") or item.get("question", "")

    def _solution(self, item):
        return item.get("solution", "") or item.get("Solution", "")

    def _gold_final_answer(self, item):
        return normalized_last_boxed(self._solution(item))

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
            "subject": entry.get("_subject", ""),
            "question": self._question(entry),
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
            "subject": entry.get("_subject", ""),
            "question": self._question(entry),
            "gold_final_answer": self._gold_final_answer(entry),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
        }

    def get_prompt(self):
        raise NotImplementedError("MATH uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("MATH uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("MATH uses evaluate_entry(output, entry).")
