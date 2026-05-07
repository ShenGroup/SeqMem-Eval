import json
import re
from pathlib import Path

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.math_answer_utils import normalized_last_boxed


_INT_RE = re.compile(r"\b(\d{1,3})\b")


def _extract_aime_answer(output: str) -> str:
    """Boxed-first integer extraction with last-int fallback (0..999)."""
    if not output:
        return ""
    boxed = normalized_last_boxed(output)
    if boxed:
        m = re.search(r"\d+", boxed)
        if m:
            try:
                v = int(m.group(0))
                if 0 <= v <= 999:
                    return str(v)
            except ValueError:
                pass
    # Fallback: last 1–3 digit integer in the output that lies in [0, 999].
    candidates = _INT_RE.findall(output)
    for tok in reversed(candidates):
        try:
            v = int(tok)
        except ValueError:
            continue
        if 0 <= v <= 999:
            return str(v)
    return ""


class AIME2024Task(BaseTask):
    name = "AIME2024"
    prompt_key = "aime2024"
    _default_year = 2024

    def __init__(
        self,
        prompt_file=None,
        hf_path="Maxwell-Jia/AIME_2024",
        split="train",
        data_file=None,
    ):
        self.prompt_file = prompt_file
        self.hf_path = hf_path
        self.split = split
        self.data_file = data_file
        self._dataset = None

    def _load_local(self, path: Path):
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        self._dataset = items

    def _load(self):
        if self._dataset is not None:
            return
        if self.data_file:
            self._load_local(Path(self.data_file))
            return
        # Fall back to HF if no local file given.
        from datasets import load_dataset
        self._dataset = load_dataset(self.hf_path, split=self.split)

    def _question(self, item):
        return (
            item.get("Problem", "")
            or item.get("problem", "")
            or item.get("question", "")
            or item.get("Question", "")
        )

    def _answer(self, item):
        for key in ("Answer", "answer", "Solution", "solution"):
            v = item.get(key, "")
            if v != "" and v is not None:
                return str(v).strip()
        return ""

    def _solution(self, item):
        return item.get("Solution", "") or item.get("solution", "")

    def _qid(self, item, idx):
        for key in ("qid", "ID", "id", "problem_id"):
            if key in item and item[key] not in ("", None):
                return str(item[key])
        return str(idx)

    def iter_entries(self):
        self._load()
        for idx, item in enumerate(self._dataset):
            item_copy = dict(item)
            item_copy["_qid"] = self._qid(item_copy, idx)
            yield item_copy

    def total_entries(self):
        self._load()
        return len(self._dataset)

    def build_prompt(self, entry):
        return get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._question(entry),
        )

    def build_inputs(self, entry):
        return {
            "question": self._question(entry),
            "answer": self._answer(entry),
            "solution": self._solution(entry),
            "qid": entry.get("_qid", ""),
        }

    def evaluate_entry(self, output, entry):
        gold = self._answer(entry)
        if not gold:
            return 0.0
        pred = _extract_aime_answer(str(output))
        if not pred:
            return 0.0
        # Normalize both to int for comparison (gold may be "33" or "033").
        try:
            return 1.0 if int(pred) == int(gold) else 0.0
        except ValueError:
            return 1.0 if pred == gold else 0.0

    def build_memory_record(self, entry, output, feedback, score):
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question(entry),
            "gold_answer": self._answer(entry),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
        }

    def get_prompt(self):
        raise NotImplementedError("AIME2024 uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("AIME2024 uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("AIME2024 uses evaluate_entry(output, entry).")
