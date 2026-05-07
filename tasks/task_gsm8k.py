import re
from decimal import Decimal, InvalidOperation

from datasets import load_dataset

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask


_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _extract_last_number(text):
    if text is None:
        return ""
    matches = _NUMBER_PATTERN.findall(str(text))
    return matches[-1] if matches else ""


def _canonicalize_number(token):
    token = str(token).strip()
    if not token:
        return ""
    token = token.replace(",", "")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return token
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


class GSM8KTask(BaseTask):
    name = "GSM8K"
    prompt_key = "gsm8k"

    def __init__(
        self,
        prompt_file=None,
        hf_path="openai/gsm8k",
        subset="main",
        split="test",
    ):
        self.prompt_file = prompt_file
        self.hf_path = hf_path
        self.subset = subset
        self.split = split
        self._dataset = None

    def _load(self):
        if self._dataset is not None:
            return
        self._dataset = load_dataset(self.hf_path, self.subset, split=self.split)

    def _question(self, item):
        return item.get("question", "") or item.get("Question", "")

    def _answer_text(self, item):
        return item.get("answer", "") or item.get("Answer", "")

    def _gold_final_number(self, item):
        answer_text = self._answer_text(item)
        # GSM8K answers are typically like "... #### 42".
        if "####" in answer_text:
            answer_text = answer_text.split("####")[-1]
        return _canonicalize_number(_extract_last_number(answer_text))

    def _qid(self, item, idx):
        for key in ("id", "ID", "question_id"):
            if key in item:
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
            "answer": self._answer_text(entry),
            "gold_final_number": self._gold_final_number(entry),
            "qid": entry.get("_qid", ""),
        }

    def evaluate_entry(self, output, entry):
        pred = _canonicalize_number(_extract_last_number(str(output)))
        gold = self._gold_final_number(entry)
        return 1.0 if pred and gold and pred == gold else 0.0

    def build_memory_record(self, entry, output, feedback, score):
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question(entry),
            "gold_final_number": self._gold_final_number(entry),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
        }

    def get_prompt(self):
        raise NotImplementedError("GSM8K uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("GSM8K uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("GSM8K uses evaluate_entry(output, entry).")
