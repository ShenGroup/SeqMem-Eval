"""Sandboxed Python execution for TACO problems.

Supports two TACO modes:
  * `call_based`: `input_output["fn_name"]` is set. Inputs are arg lists, outputs
    are the expected return values. We spawn a driver that instantiates
    `Solution()` (or uses module-level `fn_name`) and invokes the target.
  * `standard_input`: no `fn_name`. Inputs are stdin strings, outputs are
    expected stdout strings. We patch `sys.stdin` / `sys.stdout` per test and
    exec the user program.

Each problem runs in a single subprocess to amortize interpreter startup.
Per-test timeouts use `signal.alarm` (Unix only). The parent additionally wraps
the subprocess with `subprocess.run(timeout=...)` as a hard upper bound.

The public entry point is `evaluate(user_code, input_output_json_str, ...)`,
returning a dict consumed by `tasks.task_taco`.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Code extraction
# --------------------------------------------------------------------------- #

_CODE_BLOCK_RE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Return the *last* fenced python block, or the full text if no fence exists."""
    if not text:
        return ""
    raw = str(text)
    matches = _CODE_BLOCK_RE.findall(raw)
    if matches:
        return matches[-1].strip()
    # Fallback: many models emit bare code after "ANSWER:".
    if "ANSWER:" in raw:
        tail = raw.split("ANSWER:", 1)[1]
    else:
        tail = raw
    return tail.strip()


# --------------------------------------------------------------------------- #
# Drivers executed inside the sandbox subprocess.
# `{user_code}` is injected via simple string concatenation (not .format) so
# that braces inside the user's code do not need escaping.
# --------------------------------------------------------------------------- #

_CALL_BASED_DRIVER_TAIL = r"""

# --- TACO call-based driver -------------------------------------------------
import json as _taco_json
import signal as _taco_signal
import sys as _taco_sys
import math as _taco_math


class _TacoTimeout(Exception):
    pass


def _taco_alarm_handler(signum, frame):
    raise _TacoTimeout()


_taco_signal.signal(_taco_signal.SIGALRM, _taco_alarm_handler)


def _taco_normalize(value):
    if isinstance(value, tuple):
        return [_taco_normalize(v) for v in value]
    if isinstance(value, list):
        return [_taco_normalize(v) for v in value]
    return value


def _taco_compare(actual, expected):
    a = _taco_normalize(actual)
    e = _taco_normalize(expected)
    if a == e:
        return True
    # Expected values from TACO are sometimes wrapped in a single-element list.
    if isinstance(e, list) and len(e) == 1 and a == _taco_normalize(e[0]):
        return True
    try:
        def _walk(x, y):
            if isinstance(x, list) and isinstance(y, list):
                if len(x) != len(y):
                    return False
                return all(_walk(xa, ya) for xa, ya in zip(x, y))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if _taco_math.isnan(float(x)) or _taco_math.isnan(float(y)):
                    return False
                return _taco_math.isclose(float(x), float(y), rel_tol=1e-4, abs_tol=1e-6)
            return x == y
        if _walk(a, e):
            return True
        if isinstance(e, list) and len(e) == 1 and _walk(a, e[0]):
            return True
    except Exception:
        pass
    return False


def _taco_main():
    spec_path = _taco_sys.argv[1]
    with open(spec_path, "r", encoding="utf-8") as _f:
        _spec = _taco_json.load(_f)
    _fn_name = _spec["fn_name"]
    _inputs = _spec["inputs"]
    _outputs = _spec["outputs"]
    _per_timeout = int(_spec.get("per_timeout", 6))

    _sol = None
    try:
        if "Solution" in globals():
            _sol = globals()["Solution"]()
    except Exception as exc:
        print(_taco_json.dumps({"fatal": "solution_ctor:" + repr(exc)}))
        return

    _fn = None
    if _sol is not None and hasattr(_sol, _fn_name):
        _fn = getattr(_sol, _fn_name)
    elif _fn_name in globals():
        _fn = globals()[_fn_name]

    if _fn is None:
        print(_taco_json.dumps({"fatal": "fn_not_found:" + str(_fn_name)}))
        return

    _results = []
    for _args, _exp in zip(_inputs, _outputs):
        _taco_signal.alarm(_per_timeout)
        try:
            if isinstance(_args, list):
                _actual = _fn(*_args)
            else:
                _actual = _fn(_args)
            _taco_signal.alarm(0)
            _ok = _taco_compare(_actual, _exp)
            _results.append({"ok": bool(_ok)})
        except _TacoTimeout:
            _taco_signal.alarm(0)
            _results.append({"ok": False, "err": "timeout"})
        except Exception as _exc:
            _taco_signal.alarm(0)
            _results.append({"ok": False, "err": type(_exc).__name__ + ":" + str(_exc)[:200]})

    print(_taco_json.dumps({"results": _results}))


_taco_main()
"""


_STDIN_DRIVER = r"""
# --- TACO stdin driver ------------------------------------------------------
import json, signal, sys, io, os


class _TacoTimeout(Exception):
    pass


def _h(signum, frame):
    raise _TacoTimeout()


signal.signal(signal.SIGALRM, _h)


user_code_path = sys.argv[1]
spec_path = sys.argv[2]
per_timeout = int(sys.argv[3])

with open(user_code_path, "r", encoding="utf-8") as _f:
    _src = _f.read()
_compiled = compile(_src, user_code_path, "exec")

with open(spec_path, "r", encoding="utf-8") as _f:
    _spec = json.load(_f)


def _to_text(x):
    if isinstance(x, list):
        return "\n".join(str(v) for v in x)
    return str(x) if x is not None else ""


def _compare(out, exp):
    if out == exp:
        return True
    a = out.replace("\r\n", "\n").strip()
    b = exp.replace("\r\n", "\n").strip()
    if a == b:
        return True
    la = [ln.rstrip() for ln in a.split("\n")]
    lb = [ln.rstrip() for ln in b.split("\n")]
    # Drop trailing blank lines on both sides.
    while la and not la[-1]:
        la.pop()
    while lb and not lb[-1]:
        lb.pop()
    if la == lb:
        return True
    try:
        ta = a.split()
        tb = b.split()
        if len(ta) == len(tb) and len(ta) > 0:
            for x, y in zip(ta, tb):
                if x == y:
                    continue
                fx, fy = float(x), float(y)
                if abs(fx - fy) > max(1e-4 * abs(fy), 1e-6):
                    return False
            return True
    except Exception:
        return False
    return False


_results = []
_saved_in = sys.stdin
_saved_out = sys.stdout
for _inp, _exp in zip(_spec["inputs"], _spec["outputs"]):
    _stdin_text = _to_text(_inp)
    _exp_text = _to_text(_exp)
    _buf = io.StringIO()
    sys.stdin = io.StringIO(_stdin_text)
    sys.stdout = _buf
    signal.alarm(per_timeout)
    try:
        exec(_compiled, {"__name__": "__main__", "__file__": user_code_path})
        signal.alarm(0)
        _out = _buf.getvalue()
        _results.append({"ok": _compare(_out, _exp_text)})
    except _TacoTimeout:
        signal.alarm(0)
        _results.append({"ok": False, "err": "timeout"})
    except SystemExit:
        signal.alarm(0)
        _out = _buf.getvalue()
        _results.append({"ok": _compare(_out, _exp_text)})
    except Exception as _exc:
        signal.alarm(0)
        _results.append({"ok": False, "err": type(_exc).__name__ + ":" + str(_exc)[:200]})
    finally:
        sys.stdin = _saved_in
        sys.stdout = _saved_out

print(json.dumps({"results": _results}))
"""


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #


def _select_tests(
    io_spec: Dict[str, Any], max_tests: Optional[int]
) -> Tuple[List[Any], List[Any]]:
    inputs = list(io_spec.get("inputs") or [])
    outputs = list(io_spec.get("outputs") or [])
    n = min(len(inputs), len(outputs))
    inputs, outputs = inputs[:n], outputs[:n]
    if max_tests and max_tests > 0 and n > max_tests:
        inputs, outputs = inputs[:max_tests], outputs[:max_tests]
    return inputs, outputs


def _aggregate(results: List[Dict[str, Any]], total: int) -> Dict[str, Any]:
    passed = sum(1 for r in results if r.get("ok"))
    first_err = ""
    for r in results:
        if not r.get("ok"):
            first_err = r.get("err", "wrong_answer")
            break
    return {
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) if total else 0.0,
        "all_pass": passed == total and total > 0,
        "first_error": first_err,
        "per_test": results,
    }


def _run_call_based(
    user_code: str,
    io_spec: Dict[str, Any],
    per_timeout: int,
    max_tests: Optional[int],
    hard_timeout: int,
) -> Dict[str, Any]:
    inputs, outputs = _select_tests(io_spec, max_tests)
    total = len(inputs)
    if total == 0:
        return _aggregate([], 0)

    fn_name = io_spec.get("fn_name") or ""
    with tempfile.TemporaryDirectory(prefix="taco_cb_") as tmpdir:
        spec_path = Path(tmpdir) / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "fn_name": fn_name,
                    "inputs": inputs,
                    "outputs": outputs,
                    "per_timeout": per_timeout,
                }
            ),
            encoding="utf-8",
        )
        prog = user_code + _CALL_BASED_DRIVER_TAIL
        prog_path = Path(tmpdir) / "prog.py"
        prog_path.write_text(prog, encoding="utf-8")

        try:
            cp = subprocess.run(
                [sys.executable, str(prog_path), str(spec_path)],
                capture_output=True,
                text=True,
                timeout=hard_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _aggregate(
                [{"ok": False, "err": "hard_timeout"}] * total, total
            )
        out = cp.stdout or ""
        last_line = out.strip().splitlines()[-1] if out.strip() else ""
        try:
            payload = json.loads(last_line)
        except Exception:
            err = (cp.stderr or "").strip().splitlines()
            msg = err[-1] if err else "driver_parse_error"
            return _aggregate(
                [{"ok": False, "err": f"driver:{msg[:200]}"}] * total, total
            )
        if "fatal" in payload:
            return _aggregate(
                [{"ok": False, "err": payload["fatal"][:200]}] * total, total
            )
        return _aggregate(payload.get("results") or [], total)


def _run_stdin(
    user_code: str,
    io_spec: Dict[str, Any],
    per_timeout: int,
    max_tests: Optional[int],
    hard_timeout: int,
) -> Dict[str, Any]:
    inputs, outputs = _select_tests(io_spec, max_tests)
    total = len(inputs)
    if total == 0:
        return _aggregate([], 0)

    with tempfile.TemporaryDirectory(prefix="taco_si_") as tmpdir:
        code_path = Path(tmpdir) / "user_code.py"
        code_path.write_text(user_code, encoding="utf-8")
        spec_path = Path(tmpdir) / "spec.json"
        spec_path.write_text(
            json.dumps({"inputs": inputs, "outputs": outputs}),
            encoding="utf-8",
        )
        driver_path = Path(tmpdir) / "driver.py"
        driver_path.write_text(_STDIN_DRIVER, encoding="utf-8")

        try:
            cp = subprocess.run(
                [
                    sys.executable,
                    str(driver_path),
                    str(code_path),
                    str(spec_path),
                    str(per_timeout),
                ],
                capture_output=True,
                text=True,
                timeout=hard_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _aggregate(
                [{"ok": False, "err": "hard_timeout"}] * total, total
            )
        out = cp.stdout or ""
        last_line = out.strip().splitlines()[-1] if out.strip() else ""
        try:
            payload = json.loads(last_line)
        except Exception:
            err = (cp.stderr or "").strip().splitlines()
            msg = err[-1] if err else "driver_parse_error"
            return _aggregate(
                [{"ok": False, "err": f"driver:{msg[:200]}"}] * total, total
            )
        return _aggregate(payload.get("results") or [], total)


def evaluate(
    user_code: str,
    input_output_json: str,
    per_timeout: int = 6,
    max_tests: Optional[int] = 15,
) -> Dict[str, Any]:
    """Execute `user_code` against the test cases encoded in `input_output_json`.

    Returns a dict with `passed`, `total`, `pass_rate`, `all_pass`, `first_error`,
    `per_test`, `mode`. If the user's code is empty, all tests fail.
    """
    if not user_code or not user_code.strip():
        return {
            "passed": 0,
            "total": 0,
            "pass_rate": 0.0,
            "all_pass": False,
            "first_error": "empty_code",
            "per_test": [],
            "mode": "unknown",
        }

    try:
        io_spec = json.loads(input_output_json) if isinstance(input_output_json, str) else dict(input_output_json)
    except Exception as exc:
        return {
            "passed": 0,
            "total": 0,
            "pass_rate": 0.0,
            "all_pass": False,
            "first_error": f"bad_input_output_json:{exc}",
            "per_test": [],
            "mode": "unknown",
        }

    fn_name = io_spec.get("fn_name")
    mode = "call_based" if fn_name else "standard_input"
    inputs, _ = _select_tests(io_spec, max_tests)
    n_selected = len(inputs)
    # Hard per-problem wall clock ceiling: per-test × selected + startup slack.
    hard_timeout = max(30, per_timeout * max(1, n_selected) + 15)

    if mode == "call_based":
        result = _run_call_based(user_code, io_spec, per_timeout, max_tests, hard_timeout)
    else:
        result = _run_stdin(user_code, io_spec, per_timeout, max_tests, hard_timeout)
    result["mode"] = mode
    return result
