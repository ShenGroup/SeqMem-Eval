"""Sandboxed Python execution for HumanEval problems.

Each HumanEval example ships a `test` string that defines
`def check(candidate): ...`. Evaluation runs the model's completed code,
then calls `check({entry_point})`. The sample passes iff the subprocess
exits cleanly (no exception, no AssertionError, no timeout).

Public API: `evaluate(user_code, test_src, entry_point, timeout=10)`.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py|python3)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_python_code(text: str) -> str:
    """Return the *last* fenced python block, or best-effort text tail."""
    if not text:
        return ""
    raw = str(text)
    matches = _CODE_BLOCK_RE.findall(raw)
    if matches:
        return matches[-1].strip()
    # Fallback: "ANSWER:" prefix, else raw text.
    tail = raw.split("ANSWER:", 1)[1] if "ANSWER:" in raw else raw
    return tail.strip()


def _build_program(user_code: str, test_src: str, entry_point: str) -> str:
    return (
        user_code.rstrip()
        + "\n\n"
        + test_src.rstrip()
        + "\n\n"
        + f"check({entry_point})\n"
    )


def evaluate(
    user_code: str,
    test_src: str,
    entry_point: str,
    timeout: int = 10,
) -> Dict[str, Any]:
    """Run `check(entry_point)` against `user_code + test_src`.

    Returns a dict with the same shape used by the TACO evaluator so
    downstream code paths stay uniform:
        passed (int, 0 or 1), total (int, 1),
        pass_rate (float), all_pass (bool),
        first_error (str), stderr_tail (str), mode ("call_based").
    """
    total = 1
    if not user_code or not user_code.strip():
        return {
            "passed": 0,
            "total": total,
            "pass_rate": 0.0,
            "all_pass": False,
            "first_error": "empty_code",
            "stderr_tail": "",
            "mode": "call_based",
        }
    if not entry_point:
        return {
            "passed": 0,
            "total": total,
            "pass_rate": 0.0,
            "all_pass": False,
            "first_error": "missing_entry_point",
            "stderr_tail": "",
            "mode": "call_based",
        }

    program = _build_program(user_code, test_src, entry_point)
    with tempfile.TemporaryDirectory(prefix="humaneval_") as tmpdir:
        prog_path = Path(tmpdir) / "prog.py"
        prog_path.write_text(program, encoding="utf-8")
        try:
            cp = subprocess.run(
                [sys.executable, str(prog_path)],
                capture_output=True,
                text=True,
                timeout=max(2, int(timeout)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "total": total,
                "pass_rate": 0.0,
                "all_pass": False,
                "first_error": "timeout",
                "stderr_tail": "",
                "mode": "call_based",
            }

    if cp.returncode == 0:
        return {
            "passed": 1,
            "total": total,
            "pass_rate": 1.0,
            "all_pass": True,
            "first_error": "",
            "stderr_tail": "",
            "mode": "call_based",
        }

    stderr = (cp.stderr or "").strip()
    # Last non-empty line usually names the exception type + message.
    tail_lines = [ln for ln in stderr.splitlines() if ln.strip()]
    first_error = tail_lines[-1][:300] if tail_lines else f"exit_{cp.returncode}"
    return {
        "passed": 0,
        "total": total,
        "pass_rate": 0.0,
        "all_pass": False,
        "first_error": first_error,
        "stderr_tail": "\n".join(tail_lines[-20:])[:2000],
        "mode": "call_based",
    }
