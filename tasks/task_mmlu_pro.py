import re
from pathlib import Path

import pandas as pd

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.math_answer_utils import extract_last_boxed


LETTERS = "ABCDEFGHIJ"


def _format_options(options):
    lines = []
    for idx, opt in enumerate(options):
        if idx >= len(LETTERS):
            break
        lines.append(f"{LETTERS[idx]}. {opt}")
    return "\n".join(lines)


_ANSWER_RE_LIST = [
    re.compile(r"answer\s*(?:is|:|=)\s*\*?\*?\(?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"\boption\s*\(?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"^\s*\(?\s*([A-J])\s*\)?\s*$", re.IGNORECASE | re.MULTILINE),
]


def _extract_letter(text):
    """Extract a single option letter (A-J) from model output.

    Priority: \\boxed{...} -> explicit "answer is X" phrasings -> last bare letter.
    """
    if not text:
        return ""
    raw = str(text)

    boxed = extract_last_boxed(raw)
    if boxed:
        m = re.search(r"([A-J])", boxed)
        if m:
            return m.group(1).upper()
        m = re.search(r"([A-J])", boxed, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    for pattern in _ANSWER_RE_LIST:
        matches = pattern.findall(raw)
        if matches:
            return matches[-1].upper()

    return ""


class MMLUProTask(BaseTask):
    """Base class for all MMLU-Pro category subsets.

    Subclasses set ``name``, ``prompt_key``, and ``_category``.
    """

    _category = None  # override in subclass: "math", "physics", "engineering", ...

    def __init__(self, prompt_file=None, data_file=None):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        self.data_file = (
            Path(data_file) if data_file else root / "data" / "MMLU-Pro" / "test.parquet"
        )
        self._entries = None

    def _load(self):
        if self._entries is not None:
            return
        if not self.data_file.exists():
            raise RuntimeError(f"MMLU-Pro data not found: {self.data_file}")
        suffix = self.data_file.suffix.lower()
        if suffix in (".jsonl", ".json"):
            import json as _json
            rows = []
            with self.data_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(_json.loads(line))
            frame = pd.DataFrame(rows)
        else:
            frame = pd.read_parquet(self.data_file)
        if self._category and "category" in frame.columns:
            frame = frame[frame["category"] == self._category].reset_index(drop=True)

        entries = []
        for item in frame.to_dict(orient="records"):
            row = dict(item)
            raw_options = row.get("options")
            if raw_options is None:
                options = []
            else:
                options = [str(o) for o in list(raw_options)]
            row["options"] = options
            row["_qid"] = str(row.get("question_id", len(entries)))
            row["_subject"] = str(row.get("src", self._category or ""))
            row["_gold_letter"] = str(row.get("answer", "")).strip().upper()
            entries.append(row)
        self._entries = entries

    def _question(self, item):
        return item.get("question", "") or ""

    def _options_block(self, item):
        return _format_options(item.get("options") or [])

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
            options=self._options_block(entry),
        )

    def build_inputs(self, entry):
        return {
            "qid": entry.get("_qid", ""),
            "subject": entry.get("_subject", ""),
            "question": self._question(entry),
            "options": list(entry.get("options") or []),
            "gold_letter": entry.get("_gold_letter", ""),
        }

    def get_query(self, entry, inputs):
        opts = inputs.get("options") or []
        options_text = _format_options(opts)
        return f"{inputs.get('question', '')}\n{options_text}"

    def evaluate_entry(self, output, entry):
        pred = _extract_letter(output)
        gold = entry.get("_gold_letter", "")
        return 1.0 if pred and gold and pred == gold else 0.0

    def build_memory_record(self, entry, output, feedback, score):
        return {
            "qid": entry.get("_qid", ""),
            "subject": entry.get("_subject", ""),
            "question": self._question(entry),
            "options": list(entry.get("options") or []),
            "gold_letter": entry.get("_gold_letter", ""),
            "model_output": str(output),
            "predicted_letter": _extract_letter(output),
            "feedback": feedback,
            "score": score,
        }

    def get_prompt(self):
        raise NotImplementedError(
            f"{self.name} uses build_prompt(entry) in stream mode."
        )

    def get_inputs(self):
        raise NotImplementedError(
            f"{self.name} uses build_inputs(entry) in stream mode."
        )

    def evaluate(self, output):
        raise NotImplementedError(
            f"{self.name} uses evaluate_entry(output, entry)."
        )
