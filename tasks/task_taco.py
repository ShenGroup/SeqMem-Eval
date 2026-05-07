"""TACO (Topics in Algorithmic COde generation) task adapter.

Data: `data/TACO/test.parquet` (download via `python data/TACO/download.py`).
Official source: https://huggingface.co/datasets/BAAI/TACO, FlagOpen/TACO.

Evaluation: execution-based pass@1 — the model's program is run against the
test cases in `input_output`. `score == 1.0` iff **all** selected test cases pass
(both stdin/stdout mode and fn_name call-based mode are supported).
Per-problem cost is controlled by `taco_max_tests` and `taco_per_timeout`.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.taco_code_exec import evaluate as taco_evaluate
from tasks.taco_code_exec import extract_python_code


_DIFFICULTIES = {"EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"}


def _call_based_instruction(fn_name: str) -> str:
    name = fn_name.strip() or "solve"
    return (
        "Use Call-Based format: define a `class Solution` whose method "
        f"`{name}(...)` returns the answer for each test case. Do not read stdin "
        "or print anything; only the method's return value is evaluated. Infer "
        "the method's argument list from the problem statement."
    )


def _stdin_instruction() -> str:
    return (
        "Use Standard Input format: read the input from stdin and print the "
        "answer to stdout. The program is run once per test case."
    )


def _starter_block(starter_code: str) -> str:
    code = (starter_code or "").strip()
    if not code:
        return ""
    return (
        "Starter code (you must keep the given signatures/names):\n"
        "```python\n" + code + "\n```\n\n"
    )


class TACOTask(BaseTask):
    """BAAI/TACO test split: algorithmic competitive-programming problems."""

    name = "TACO"
    prompt_key = "taco"

    # --- knobs that main.py can override after construction ---
    taco_max_tests: int = 15       # cap per-problem test cases executed (0 → no cap)
    taco_per_timeout: int = 6      # seconds per individual test case
    taco_difficulties: Optional[List[str]] = None  # e.g. ["EASY","MEDIUM"]

    def __init__(self, prompt_file: Optional[str] = None, data_file: Optional[str] = None):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        self.data_file = (
            Path(data_file) if data_file else root / "data" / "TACO" / "test.parquet"
        )
        self._entries: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if self._entries is not None:
            return
        if not self.data_file.exists():
            raise RuntimeError(
                f"TACO data file not found: {self.data_file}\n"
                f"Run: python {self.data_file.parent}/download.py"
            )
        frame = pd.read_parquet(self.data_file)
        allow = self._difficulty_filter()
        entries: List[Dict[str, Any]] = []
        for idx, row in enumerate(frame.to_dict(orient="records")):
            difficulty = str(row.get("difficulty", "")).strip().upper()
            if allow is not None and difficulty not in allow:
                continue
            item = dict(row)
            item["_qid"] = str(idx)
            item["_difficulty"] = difficulty
            entries.append(item)
        self._entries = entries

    def _difficulty_filter(self) -> Optional[set]:
        diffs = self.taco_difficulties
        if not diffs:
            return None
        norm = {str(d).strip().upper() for d in diffs if str(d).strip()}
        return norm & _DIFFICULTIES if norm else None

    # ------------------------------------------------------------------ #
    # Field accessors
    # ------------------------------------------------------------------ #

    @staticmethod
    def _question(item: Dict[str, Any]) -> str:
        return item.get("question", "") or ""

    @staticmethod
    def _starter(item: Dict[str, Any]) -> str:
        return item.get("starter_code", "") or ""

    @staticmethod
    def _input_output_json(item: Dict[str, Any]) -> str:
        raw = item.get("input_output", "") or ""
        if isinstance(raw, (dict, list)):
            return json.dumps(raw)
        return str(raw)

    @staticmethod
    def _fn_name(item: Dict[str, Any]) -> str:
        try:
            io_spec = json.loads(TACOTask._input_output_json(item) or "{}")
        except Exception:
            return ""
        return str(io_spec.get("fn_name") or "")

    # ------------------------------------------------------------------ #
    # BaseTask API
    # ------------------------------------------------------------------ #

    def iter_entries(self) -> Iterable[Dict[str, Any]]:
        self._load()
        for item in self._entries or []:
            yield item

    def total_entries(self) -> int:
        self._load()
        return len(self._entries or [])

    def build_prompt(self, entry: Dict[str, Any]) -> str:
        fn_name = self._fn_name(entry)
        format_spec = (
            _call_based_instruction(fn_name) if fn_name else _stdin_instruction()
        )
        starter_block = _starter_block(self._starter(entry))
        return get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._question(entry),
            starter_block=starter_block,
            format_spec=format_spec,
        )

    def build_inputs(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question(entry),
            "difficulty": entry.get("_difficulty", ""),
            "source": str(entry.get("source", "")),
            "starter_code": self._starter(entry),
            "fn_name": self._fn_name(entry),
            "input_output": self._input_output_json(entry),
            "tags": str(entry.get("tags", "")),
            "skill_types": str(entry.get("skill_types", "")),
        }

    def get_query(self, entry: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        # Methods (HistoryRAG etc.) embed this text for similarity retrieval.
        return str(inputs.get("question") or self._question(entry))

    def evaluate_entry(self, output: str, entry: Dict[str, Any]) -> float:
        code = extract_python_code(output)
        result = taco_evaluate(
            user_code=code,
            input_output_json=self._input_output_json(entry),
            per_timeout=int(self.taco_per_timeout),
            max_tests=(None if not self.taco_max_tests else int(self.taco_max_tests)),
        )
        # Stash rich per-sample info so build_memory_record can embed it.
        self._last_eval_info = {
            "extracted_code": code,
            "passed_cases": result["passed"],
            "total_cases": result["total"],
            "pass_rate": result["pass_rate"],
            "first_error": result["first_error"],
            "mode": result["mode"],
        }
        return 1.0 if result["all_pass"] else 0.0

    def build_memory_record(
        self,
        entry: Dict[str, Any],
        output: str,
        feedback: str,
        score: float,
    ) -> Dict[str, Any]:
        info = getattr(self, "_last_eval_info", None) or {}
        passed = info.get("passed_cases", 0)
        total = info.get("total_cases", 0)
        mode = info.get("mode", "")
        first_err = info.get("first_error", "")
        human_feedback = (
            "success: all test cases passed"
            if score >= 1.0
            else f"failure: {passed}/{total} test cases passed"
            + (f" | first_error={first_err}" if first_err else "")
        )
        return {
            "qid": entry.get("_qid", ""),
            "difficulty": entry.get("_difficulty", ""),
            "source": str(entry.get("source", "")),
            "tags": str(entry.get("tags", "")),
            "skill_types": str(entry.get("skill_types", "")),
            "fn_name": self._fn_name(entry),
            "mode": mode,
            "question": self._question(entry),
            "starter_code": self._starter(entry),
            "model_output": str(output),
            "extracted_code": info.get("extracted_code", ""),
            "passed_cases": passed,
            "total_cases": total,
            "pass_rate": info.get("pass_rate", 0.0),
            "first_error": first_err,
            "feedback": human_feedback,
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # Abstract stubs (BaseTask requires them; TACO uses the streaming API)
    # ------------------------------------------------------------------ #

    def get_prompt(self):
        raise NotImplementedError("TACO uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("TACO uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("TACO uses evaluate_entry(output, entry).")
