import json
from pathlib import Path

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.math_answer_utils import normalized_last_boxed


class MATH500Task(BaseTask):
    name = "MATH500"
    prompt_key = "math"

    def __init__(self, prompt_file=None, data_dir=None, data_file=None):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        default = root / "data" / "MATH500" / "test-2.jsonl"
        if data_file is not None:
            path = Path(data_file)
        elif data_dir is not None:
            path = Path(data_dir)
            if path.is_dir():
                path = path / "test-2.jsonl"
        else:
            path = default
        self.jsonl_path = path
        self._entries = None

    def _load(self):
        if self._entries is not None:
            return
        if not self.jsonl_path.exists():
            raise RuntimeError(f"MATH500 jsonl not found: {self.jsonl_path}")
        entries = []
        with self.jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                row = dict(item)
                row["_subject"] = item.get("subject", "")
                qid = item.get("unique_id") or f"{row['_subject']}-{len(entries)}"
                row["_qid"] = qid
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
        raise NotImplementedError("MATH500 uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("MATH500 uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("MATH500 uses evaluate_entry(output, entry).")
