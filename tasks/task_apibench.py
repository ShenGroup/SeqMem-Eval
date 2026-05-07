"""Gorilla APIBench (HuggingFace / TensorFlow Hub / TorchHub) task adapter.

Data source: https://huggingface.co/datasets/gorilla-llm/APIBench
Evaluates only the `*_eval.json` test splits (911 HF / 688 TF / 186 TH),
using the dataset's paired `*_api.jsonl` files as the candidate API pool.

Official prompt (Gorilla `encode_question`) and evaluation logic (subset-
specific AST subtree matching + domain equality) are preserved. The AST
evaluator is a stdlib port; see `tasks/apibench_ast_eval.py`.

A single class hierarchy shares data-loading + evaluation logic across the
three subsets, with each subset mapping to its own task name so retrieval-
based methods (HistoryRAG etc.) keep memory scoped per subset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from prompts.prompt_loader import get_prompt
from tasks import apibench_ast_eval
from tasks.base_task import BaseTask


_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "APIBench"
_SYSTEM_INSTRUCTION_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "apibench" / "system_instruction.txt"
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_system_instruction() -> str:
    try:
        return _SYSTEM_INSTRUCTION_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "You are a helpful API writer who can write APIs based on requirements."


class APIBenchBase(BaseTask):
    """Shared base for HF / TF / TH subsets.

    Subclasses must set: `name`, `_subset`, `_eval_filename`, `_pool_filename`,
    `prompt_key`.
    """

    # subclasses override
    name: str = "APIBench"
    prompt_key: str = "apibench_hf"
    _subset: str = "hf"                  # "hf" | "tf" | "th"
    _eval_filename: str = ""
    _pool_filename: str = ""

    def __init__(self, prompt_file: Optional[str] = None, data_dir: Optional[str] = None):
        self.prompt_file = prompt_file
        self._data_dir = Path(data_dir) if data_dir else _DATA_ROOT
        self._entries: Optional[List[Dict[str, Any]]] = None
        self._api_pool: Optional[List[Dict[str, Any]]] = None
        self._system_instruction = _read_system_instruction()

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if self._entries is not None and self._api_pool is not None:
            return
        eval_path = self._data_dir / self._eval_filename
        pool_path = self._data_dir / self._pool_filename
        if not eval_path.exists():
            raise RuntimeError(
                f"APIBench eval file not found: {eval_path}. "
                f"Expected under data/APIBench/."
            )
        if not pool_path.exists():
            raise RuntimeError(
                f"APIBench api pool file not found: {pool_path}. "
                f"Expected under data/APIBench/."
            )
        entries = _read_jsonl(eval_path)
        for idx, item in enumerate(entries):
            item["_qid"] = str(idx + 1)  # 1-based, matches Gorilla question_id
        self._entries = entries
        self._api_pool = _read_jsonl(pool_path)

    # ------------------------------------------------------------------ #
    # Field accessors
    # ------------------------------------------------------------------ #

    @staticmethod
    def _question_src(item: Dict[str, Any]) -> str:
        # `code` field in Gorilla is the prompt text like "###Instruction: ...\n###Output: ..."
        return str(item.get("code") or "")

    @staticmethod
    def _gold_api_call(item: Dict[str, Any]) -> str:
        return str(item.get("api_call") or "")

    @staticmethod
    def _gold_api_data(item: Dict[str, Any]) -> Dict[str, Any]:
        data = item.get("api_data")
        return dict(data) if isinstance(data, dict) else {}

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
        user_block = get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._question_src(entry),
        )
        system = self._system_instruction.strip()
        if not system:
            return user_block
        return f"{system}\n\n{user_block}"

    def build_inputs(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question_src(entry),
            "gold_api_call": self._gold_api_call(entry),
            "gold_domain": self._gold_api_data(entry).get("domain", ""),
            "provider": str(entry.get("provider") or ""),
        }

    def get_query(self, entry: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        return str(inputs.get("question") or self._question_src(entry))

    def evaluate_entry(self, output: str, entry: Dict[str, Any]) -> float:
        self._load()
        info = apibench_ast_eval.evaluate_prediction(
            output_text=str(output or ""),
            gold_api_data=self._gold_api_data(entry),
            api_pool=self._api_pool or [],
            subset=self._subset,
        )
        # Stash for build_memory_record + runner hallucination aggregation.
        self._last_eval_info = {
            "correct": bool(info.get("correct", False)),
            "hallucinated": int(bool(info.get("hallucinated", False))),
            "matched_idx": info.get("matched_idx"),
            "predicted_call_text": info.get("predicted_call_text", ""),
            "reason": info.get("reason", ""),
        }
        return 1.0 if info.get("correct") else 0.0

    def build_memory_record(
        self,
        entry: Dict[str, Any],
        output: str,
        feedback: str,
        score: float,
    ) -> Dict[str, Any]:
        info = getattr(self, "_last_eval_info", None) or {}
        return {
            "qid": entry.get("_qid", ""),
            "subset": self._subset,
            "question": self._question_src(entry),
            "model_output": str(output),
            "predicted_api_call": info.get("predicted_call_text", ""),
            "gold_api_call": self._gold_api_call(entry),
            "gold_domain": self._gold_api_data(entry).get("domain", ""),
            "matched_idx": info.get("matched_idx"),
            "correct": bool(info.get("correct", False)),
            "hallucinated": bool(info.get("hallucinated", 0)),
            "reason": info.get("reason", ""),
            "feedback": feedback,
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # Abstract stubs (BaseTask requires them; APIBench uses streaming path)
    # ------------------------------------------------------------------ #

    def get_prompt(self):
        raise NotImplementedError("APIBench uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("APIBench uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("APIBench uses evaluate_entry(output, entry).")


class APIBenchHF(APIBenchBase):
    name = "APIBench-HF"
    prompt_key = "apibench_hf"
    _subset = "hf"
    _eval_filename = "huggingface_eval.json"
    _pool_filename = "huggingface_api.jsonl"


class APIBenchTF(APIBenchBase):
    name = "APIBench-TF"
    prompt_key = "apibench_tf"
    _subset = "tf"
    _eval_filename = "tensorflow_eval.json"
    _pool_filename = "tensorflowhub_api.jsonl"


class APIBenchTH(APIBenchBase):
    name = "APIBench-TH"
    prompt_key = "apibench_th"
    _subset = "th"
    _eval_filename = "torchhub_eval.json"
    _pool_filename = "torchhub_api.jsonl"
