import json
from pathlib import Path

from tasks.task_math import MATHTask


class MATH500Task(MATHTask):
    name = "MATH500"

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
