"""BFCL multi_turn_base task.

Adapts the Berkeley Function Calling Leaderboard multi-turn-base split
(https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
into the harness. Each sample is a multi-turn tool-use dialogue
over a fixed set of BFCL's simulated Python backends (GorillaFileSystem,
MessageAPI, TradingBot, TravelAPI, TicketAPI, TwitterAPI, VehicleControlAPI,
MathAPI).

Data resolution (first match wins): `data/bfcl/<dataset>.json` if the user
dropped a copy locally, otherwise the dataset bundled with the installed
`bfcl-eval` package (`<site-packages>/bfcl_eval/data/BFCL_v{4,3}_multi_turn_base.json`).
Ground-truth trajectories come from `possible_answer/<dataset>.json` and
function schemas from `multi_turn_func_doc/<class>.json`; both are merged
into each entry at iter time. Backend API classes are loaded dynamically
from `bfcl_eval.eval_checker.multi_turn_eval.func_source_code.*`
(install with `pip install bfcl-eval`).

Scoring is a simplified two-check pass: backend-state equality vs a fresh
ground-truth replay + ground-truth call list is a subsequence of the model's
executed calls. Full BFCL fidelity (read-only arg matching, inference-time
tool feedback within a turn) is left to method-level `run_trial` overrides.
"""

import ast
import copy
import importlib
import json
import re
from pathlib import Path

from tasks.base_task import BaseTask


# BFCL involved-class name -> upstream module name under
# bfcl_eval.eval_checker.multi_turn_eval.func_source_code
_CLASS_MODULE_MAP = {
    "GorillaFileSystem": "gorilla_file_system",
    "MessageAPI": "message_api",
    "MathAPI": "math_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "TicketAPI": "ticket_api",
    "TwitterAPI": "posting_api",
    "VehicleControlAPI": "vehicle_control",
}

# For each involved class, the corresponding multi_turn_func_doc JSONL file.
_CLASS_FUNC_DOC_MAP = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MessageAPI": "message_api.json",
    "MathAPI": "math_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "TicketAPI": "ticket_api.json",
    "TwitterAPI": "posting_api.json",
    "VehicleControlAPI": "vehicle_control.json",
}

_FUNC_SOURCE_ROOT = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"

# Try v4 first (current package convention), then v3 (older releases).
_DATASET_BASENAMES = (
    "BFCL_v4_multi_turn_base.json",
    "BFCL_v3_multi_turn_base.json",
)


def _bfcl_package_data_root():
    """Return the bundled dataset root inside an installed bfcl-eval, or None."""
    try:
        import bfcl_eval  # noqa: F401
    except ImportError:
        return None
    try:
        root = Path(importlib.import_module("bfcl_eval").__file__).parent / "data"
    except Exception:
        return None
    return root if root.is_dir() else None


def _extract_tag_content(text, tag_name):
    if not text:
        return ""
    start = f"<{tag_name}>"
    end = f"</{tag_name}>"
    if start not in text or end not in text:
        return ""
    try:
        return text.split(start, 1)[1].split(end, 1)[0].strip()
    except Exception:
        return ""


def _strip_code_fences(text):
    # Remove ```python ... ``` fences but keep the inner code.
    return re.sub(r"```[a-zA-Z0-9_+\-]*\s*\n?|\n?```", "\n", str(text or ""))


def parse_function_calls(text):
    """Extract top-level Python call expressions from a model response.

    Accepts the official BFCL output format (a single list literal
    `[f(...), g(...)]`) as well as bare calls, newline-separated calls, and
    calls wrapped in `<trajectory>...</trajectory>` or ``` code fences.
    Non-call statements are dropped silently. Order-preserving.
    """
    if not text:
        return []
    body = _extract_tag_content(text, "trajectory") or text
    body = _strip_code_fences(body).strip("`\n ")
    calls = []

    # Mirror BFCL's decode_execute_prompting: strip outer backticks/whitespace,
    # wrap with [...] if not already a list, then AST-parse as a single expr.
    probes = [body]
    if body and not body.startswith("["):
        probes.append("[" + body.rstrip(",;") + "]")
    for probe in probes:
        try:
            expr = ast.parse(probe, mode="eval")
        except SyntaxError:
            continue
        nodes = []
        if isinstance(expr.body, (ast.List, ast.Tuple)):
            nodes = list(expr.body.elts)
        elif isinstance(expr.body, ast.Call):
            nodes = [expr.body]
        for node in nodes:
            if isinstance(node, ast.Call):
                try:
                    calls.append(ast.unparse(node))
                except Exception:
                    continue
        if calls:
            return calls

    # Multi-statement fallback (Expr wrappers, newline-separated statements).
    try:
        tree = ast.parse(body)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if isinstance(node, ast.Expr):
                val = node.value
                if isinstance(val, ast.Call):
                    try:
                        calls.append(ast.unparse(val))
                    except Exception:
                        continue
                elif isinstance(val, (ast.List, ast.Tuple)):
                    for elt in val.elts:
                        if isinstance(elt, ast.Call):
                            try:
                                calls.append(ast.unparse(elt))
                            except Exception:
                                continue
        if calls:
            return calls

    # Line-by-line fallback.
    for raw in body.splitlines():
        line = raw.strip().rstrip(",;")
        if not line or line.startswith("#"):
            continue
        try:
            e = ast.parse(line, mode="eval")
        except SyntaxError:
            continue
        if isinstance(e.body, ast.Call):
            try:
                calls.append(ast.unparse(e.body))
            except Exception:
                continue
    return calls


def _flatten_ground_truth(entry):
    """Return a flat list of ground-truth call strings across all turns.

    BFCL stores the reference trajectory under `ground_truth` (list of
    per-turn call lists). Older/sibling formats may use `path`; support both.
    """
    gt = entry.get("ground_truth")
    if gt is None:
        gt = entry.get("path")
    out = []
    if isinstance(gt, list):
        for turn in gt:
            if isinstance(turn, list):
                out.extend(str(c) for c in turn)
            elif isinstance(turn, str):
                out.append(turn)
    return out


def _normalize_call_string(call_str):
    """Canonicalize a call string so trivial formatting differences don't break
    subsequence matching. Keeps func name + sorted kw args."""
    try:
        expr = ast.parse(str(call_str), mode="eval")
    except SyntaxError:
        return str(call_str).strip()
    if not isinstance(expr.body, ast.Call):
        return str(call_str).strip()
    node = expr.body
    if isinstance(node.func, ast.Attribute):
        fn_name = node.func.attr
    elif isinstance(node.func, ast.Name):
        fn_name = node.func.id
    else:
        try:
            fn_name = ast.unparse(node.func)
        except Exception:
            fn_name = "?"
    pos = []
    for a in node.args:
        try:
            pos.append(ast.unparse(a))
        except Exception:
            pos.append("?")
    kw = []
    for k in node.keywords:
        try:
            kw.append((k.arg or "", ast.unparse(k.value)))
        except Exception:
            continue
    kw.sort(key=lambda x: x[0])
    parts = pos + [f"{k}={v}" for k, v in kw]
    return f"{fn_name}(" + ", ".join(parts) + ")"


def _is_subsequence(needles, haystack):
    """Both arg lists are already normalized. needles must appear in order."""
    idx = 0
    for item in haystack:
        if idx < len(needles) and item == needles[idx]:
            idx += 1
    return idx == len(needles)


def _load_backend_class(class_name):
    module_name = _CLASS_MODULE_MAP.get(class_name)
    if module_name is None:
        raise RuntimeError(
            f"Unknown BFCL backend class '{class_name}'. "
            "Update _CLASS_MODULE_MAP in tasks/task_bfcl.py if upstream added a new one."
        )
    try:
        module = importlib.import_module(f"{_FUNC_SOURCE_ROOT}.{module_name}")
    except ImportError as exc:
        raise RuntimeError(
            "BFCL task requires `pip install bfcl-eval`. "
            f"Failed to import {_FUNC_SOURCE_ROOT}.{module_name}: {exc}"
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        # TwitterAPI etc. may live alongside differently-named symbols.
        for attr in dir(module):
            if attr.lower() == class_name.lower():
                cls = getattr(module, attr)
                break
    if cls is None:
        raise RuntimeError(
            f"Class '{class_name}' not found in {_FUNC_SOURCE_ROOT}.{module_name}"
        )
    return cls


def _instantiate_backends(involved_classes, initial_config):
    """Instantiate each involved class and load its per-entry scenario."""
    backends = {}
    for cls_name in involved_classes:
        cls = _load_backend_class(cls_name)
        inst = cls()
        scenario = (initial_config or {}).get(cls_name, {})
        if hasattr(inst, "_load_scenario"):
            try:
                inst._load_scenario(copy.deepcopy(scenario))
            except Exception as exc:
                raise RuntimeError(
                    f"{cls_name}._load_scenario failed: {exc}"
                ) from exc
        else:
            # Fall back to direct attribute assignment (best-effort).
            for key, value in scenario.items():
                try:
                    setattr(inst, key, copy.deepcopy(value))
                except Exception:
                    continue
        backends[cls_name] = inst
    return backends


def _build_exec_globals(backends):
    """Map bare method names in a call string to bound methods on a backend.

    If two backends share the same method name, later one wins; this matches
    BFCL's assumption that the dispatch target is unambiguous per sample
    (the function schema given to the model is the union of involved classes).
    """
    g = {"__builtins__": {"True": True, "False": False, "None": None}}
    for inst in backends.values():
        for attr in dir(inst):
            if attr.startswith("_"):
                continue
            fn = getattr(inst, attr)
            if callable(fn):
                g[attr] = fn
        # Also expose the instance itself under its class name, so the model
        # can still use attribute-style calls like `GorillaFileSystem.ls(...)`.
        g[inst.__class__.__name__] = inst
    return g


def _exec_one_call(call_str, backends):
    """Execute a single call string. Returns (ok, result_or_error_text)."""
    try:
        expr = ast.parse(str(call_str), mode="eval")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    if not isinstance(expr.body, ast.Call):
        return False, "not a function call"
    code = compile(expr, "<bfcl_call>", "eval")
    try:
        result = eval(code, _build_exec_globals(backends))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, result


def _serialize_state(inst):
    """Return a JSON-safe snapshot of the public attributes of a backend.

    We stringify non-JSON-native values (Directory / File / other backend
    objects) up-front so downstream `json.dumps` without a `default=` hook
    (e.g. in write_jsonl) doesn't explode.
    """
    out = {}
    for k, v in vars(inst).items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            out[k] = repr(v)
        else:
            out[k] = v
    return out


class BFCLMultiTurnBaseTask(BaseTask):
    name = "BFCL-MultiTurnBase"
    prompt_key = "bfcl_multi_turn"
    action_mode = True

    def __init__(
        self,
        prompt_file=None,
        data_file=None,
        max_steps_per_turn=20,
        max_steps=80,
    ):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        self.local_data_root = root / "data" / "bfcl"
        self.data_file = Path(data_file) if data_file else None
        self._dataset_root = None  # directory that holds the chosen dataset file
        self._dataset_basename = None  # e.g. "BFCL_v4_multi_turn_base.json"
        self.max_steps_per_turn = int(max_steps_per_turn)
        self.max_steps = int(max_steps)
        self.debug_eval = False

        self._entries = None
        self._func_schemas_by_class = {}
        self._episode_state = None
        self._last_eval_info = {}

    # ----- dataset -----

    def _resolve_data_file(self):
        """Pick the dataset file to load. Order: --bfcl-data-file arg >
        local data/bfcl/ override > bundled bfcl_eval data."""
        if self.data_file is not None and self.data_file.is_file():
            return self.data_file
        if self.local_data_root.is_dir():
            for name in _DATASET_BASENAMES:
                candidate = self.local_data_root / name
                if candidate.is_file():
                    return candidate
        pkg_root = _bfcl_package_data_root()
        if pkg_root is not None:
            for name in _DATASET_BASENAMES:
                candidate = pkg_root / name
                if candidate.is_file():
                    return candidate
        raise RuntimeError(
            "BFCL multi_turn_base dataset not found. Either install "
            "`bfcl-eval` (which ships the data under its package dir) or "
            "drop BFCL_v4_multi_turn_base.json into data/bfcl/."
        )

    def _read_json_rows(self, path):
        rows = []
        with path.open("r", encoding="utf-8") as f:
            head = f.read(1)
            f.seek(0)
            if head == "[":
                rows = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        return rows

    def _load_ground_truth_map(self, dataset_root, basename):
        """Return {id: [[call_str, ...], ...]} from possible_answer/<basename>."""
        pa_path = dataset_root / "possible_answer" / basename
        if not pa_path.is_file():
            return {}
        out = {}
        for row in self._read_json_rows(pa_path):
            rid = row.get("id")
            if rid is None:
                continue
            gt = row.get("ground_truth")
            if gt is not None:
                out[rid] = gt
        return out

    def _load_func_schemas(self, dataset_root, class_names):
        """Return {class_name: [schema_dict, ...]} merged from
        multi_turn_func_doc/<file>.json (JSONL)."""
        base = dataset_root / "multi_turn_func_doc"
        out = {}
        for cls in class_names:
            if cls in self._func_schemas_by_class:
                out[cls] = self._func_schemas_by_class[cls]
                continue
            fname = _CLASS_FUNC_DOC_MAP.get(cls)
            if fname is None:
                continue
            path = base / fname
            if not path.is_file():
                continue
            try:
                schemas = self._read_json_rows(path)
            except Exception:
                schemas = []
            self._func_schemas_by_class[cls] = schemas
            out[cls] = schemas
        return out

    def _load_entries(self):
        if self._entries is not None:
            return
        data_path = self._resolve_data_file()
        self._dataset_root = data_path.parent
        self._dataset_basename = data_path.name
        rows = self._read_json_rows(data_path)
        gt_map = self._load_ground_truth_map(self._dataset_root, self._dataset_basename)
        for row in rows:
            rid = row.get("id")
            if rid in gt_map and not row.get("ground_truth"):
                row["ground_truth"] = gt_map[rid]
        self._entries = rows

    def iter_entries(self):
        self._load_entries()
        for idx, item in enumerate(self._entries):
            row = dict(item)
            row["_qid"] = str(row.get("id", idx))
            yield row

    def total_entries(self):
        self._load_entries()
        return len(self._entries)

    # ----- prompt / inputs -----

    def _collect_user_messages(self, entry):
        """question is list[list[msg]]; concatenate user-role messages per turn."""
        turns = entry.get("question") or []
        user_turns = []
        for turn in turns:
            parts = []
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        parts.append(str(msg.get("content", "")).strip())
            elif isinstance(turn, str):
                parts.append(turn.strip())
            user_turns.append("\n".join(p for p in parts if p))
        return user_turns

    def _render_turns(self, user_turns):
        lines = []
        for i, t in enumerate(user_turns):
            lines.append(f"[Turn {i + 1}] {t}")
        return "\n".join(lines)

    def _entry_function_schemas(self, entry):
        """Return the list of function schemas the model sees for this entry.

        If `entry.function` is non-empty, use it as-is. Otherwise merge the
        per-class docs from multi_turn_func_doc/ filtered by involved_classes,
        and drop any names listed in `excluded_function[ClassName]`.
        """
        baked = entry.get("function") or []
        if baked:
            return list(baked)
        if self._dataset_root is None:
            self._load_entries()
        involved = list(entry.get("involved_classes") or [])
        raw_excluded = entry.get("excluded_function")
        if isinstance(raw_excluded, dict):
            # Legacy format: per-class skip list.
            skip_by_class = {k: set(v or []) for k, v in raw_excluded.items()}
            flat_skip = set()
        else:
            # v4 format: flat list of function names to drop regardless of class.
            skip_by_class = {}
            flat_skip = set(raw_excluded or [])
        by_class = self._load_func_schemas(self._dataset_root, involved)
        schemas = []
        for cls in involved:
            skip = skip_by_class.get(cls, set()) | flat_skip
            for s in by_class.get(cls, []):
                if s.get("name") in skip:
                    continue
                schemas.append(s)
        return schemas

    def _render_functions(self, entry):
        """Stringify the function schema block shown to the model."""
        funcs = self._entry_function_schemas(entry)
        try:
            return json.dumps(funcs, ensure_ascii=False, indent=2)
        except Exception:
            return str(funcs)

    def build_prompt(self, entry):
        """Official BFCL classic system prompt — verbatim from
        `bfcl_eval/constants/default_prompts.py::_DEFAULT_SYSTEM_PROMPT`
        (default format: ret_fmt=python, tool_call_tag=False,
        func_doc_fmt=json, prompt_fmt=plaintext, style=classic).

        Turn-advance protocol (also official): model emits function calls
        until it has no more to call for the current turn, then emits an
        empty or non-decodable response — the system then advances to the
        next user turn. No sentinel function is required.
        """
        functions_block = self._render_functions(entry)
        return (
            "You are an expert in composing functions. You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
            "If none of the functions can be used, point it out. "
            "If the given question lacks the parameters required by the function, also point it out.\n"
            "You should only return the function calls in your response.\n\n"
            "If you decide to invoke any of the function(s), you MUST put it in the format of "
            "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n"
            "You SHOULD NOT include any other text in the response.\n\n"
            "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
            "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
            "Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\n"
            "Here is a list of functions in JSON format that you can invoke.\n"
            f"{functions_block}\n"
        )

    def build_inputs(self, entry):
        user_turns = self._collect_user_messages(entry)
        return {
            "question": "\n".join(user_turns),
            "qid": entry.get("_qid", ""),
            "involved_classes": list(entry.get("involved_classes", [])),
            "turns": user_turns,
            "function_schemas": self._entry_function_schemas(entry),
        }

    def get_query(self, entry, inputs):
        turns = inputs.get("turns") or []
        # Use the first turn as the retrieval key: stable and topical.
        if turns:
            return str(turns[0])
        return str(inputs.get("question", ""))

    # ----- interactive episode (turn-by-turn rollout) -----

    # Kept for back-compat: older prompts still told the model to emit
    # `done_turn()` as an explicit turn-end sentinel. The official prompt
    # does not require it, but if a model emits it we still treat it as a
    # turn-advance signal rather than an unknown-function error.
    _TURN_ADVANCE_NAMES = {"done_turn", "finish_turn", "end_turn", "next_turn", "advance_turn"}

    # Synthetic sentinel returned from _extract_actions when the model
    # emits a non-decodable response (official BFCL: advance the turn).
    _TURN_ADVANCE_SENTINEL = "__bfcl_advance_turn__"

    def _extract_actions(self, output):
        """Return a single-element list holding the batch of calls from
        this model response, matching official BFCL semantics where one
        model response equals one batch of calls executed together.

        Returns:
            [] if the model produced no text at all (crash / rate-limit);
                runner treats this as an empty action and ends the episode.
            [SENTINEL] if the response has text but no decodable calls;
                step_episode treats this as a turn-advance (official BFCL
                "once you have no more functions to call, the system will
                proceed to the next turn").
            ["call1()\\ncall2()..."] otherwise — a single joined batch
                string that step_episode will parse and execute in order.
        """
        text = str(output or "").strip()
        if not text:
            return []
        calls = parse_function_calls(text)
        if not calls:
            return [self._TURN_ADVANCE_SENTINEL]
        return ["\n".join(calls)]

    def _call_name(self, call_str):
        try:
            expr = ast.parse(str(call_str), mode="eval")
        except SyntaxError:
            return ""
        if not isinstance(expr.body, ast.Call):
            return ""
        fn = expr.body.func
        if isinstance(fn, ast.Attribute):
            return fn.attr
        if isinstance(fn, ast.Name):
            return fn.id
        return ""

    def _format_call_result(self, result):
        try:
            return json.dumps(result, ensure_ascii=False, default=str)[:1500]
        except Exception:
            return str(result)[:1500]

    def _format_batch_observation(self, results):
        """Render a list of {call, result, ok} dicts as a tool-results message
        the model will see on its next query. Mirrors BFCL's own format:
        one entry per executed call, tagged with the call string."""
        if not results:
            return "[no calls executed]"
        lines = []
        for item in results:
            tag = "tool" if item.get("ok") else "error"
            lines.append(f"[{tag}] {item['call']} -> {item['result']}")
        return "\n".join(lines)

    def _format_turn_prompt(self, turn_idx, is_start=False):
        state = self._episode_state
        turns = state["user_turns"]
        total = len(turns)
        if turn_idx >= total:
            return f"[All {total} turns completed.]"
        prefix = "" if is_start else "[Previous turn complete.]\n"
        return f"{prefix}[Turn {turn_idx + 1}/{total}] {turns[turn_idx]}"

    def start_episode(self, entry):
        involved = list(entry.get("involved_classes") or [])
        initial_config = entry.get("initial_config", {}) or {}
        backends = _instantiate_backends(involved, initial_config)
        user_turns = self._collect_user_messages(entry)
        self._episode_state = {
            "entry": dict(entry),
            "backends": backends,
            "involved": involved,
            "user_turns": user_turns,
            "turn_idx": 0,
            "steps_in_turn": 0,
            "total_steps": 0,
            "executed_actions": [],
            "parsed_actions": [],
            "model_errors": [],
            "trace": [],
            "done": False,
        }
        return self._format_turn_prompt(0, is_start=True)

    def _advance_turn(self, state, raw, kind):
        """Advance to the next user turn. Officially, this fires when the
        model emits an empty or non-decodable response; we also fire it on
        an explicit done_turn()-style sentinel or step-limit exhaustion."""
        turns = state["user_turns"]
        state["turn_idx"] += 1
        state["steps_in_turn"] = 0
        state["total_steps"] += 1
        if state["turn_idx"] >= len(turns):
            state["done"] = True
            obs = "[All turns completed.]"
        else:
            obs = self._format_turn_prompt(state["turn_idx"], is_start=False)
        state["trace"].append({
            "step": state["total_steps"],
            "turn_idx": state["turn_idx"] - 1,
            "action": raw,
            "kind": kind,
            "observation": obs,
        })
        if state["total_steps"] >= self.max_steps and not state["done"]:
            state["done"] = True
        return {
            "observation": obs,
            "done": bool(state["done"]),
            "won": False,
            "reward": "(0,)",
            "executed_action": "",
        }

    def step_episode(self, action, raw_action=None, meta=None):
        """Execute one model response against the simulated backends.

        `action` is either a synthesized turn-advance sentinel or a batch
        of one-or-more Python call expressions separated by newlines (as
        packed by `_extract_actions`). All calls in the batch are
        executed in order and their results are concatenated into a
        single observation, mirroring official BFCL behavior.
        """
        state = self._episode_state
        if not isinstance(state, dict):
            raise RuntimeError("BFCL episode not started. Call start_episode(entry) first.")
        if state.get("done", False):
            return {
                "observation": "[episode ended]",
                "done": True,
                "won": bool(self._last_eval_info.get("state_match", False)
                           and self._last_eval_info.get("subset_match", False)),
                "reward": "(0,)",
                "executed_action": "",
            }

        raw_text = str(raw_action if raw_action is not None else action or "")
        state["parsed_actions"].append(raw_text)

        # Empty action ⇒ hard episode end (model crashed / rate-limit).
        if not str(action or "").strip():
            state["done"] = True
            state["total_steps"] += 1
            trace_item = {
                "step": state["total_steps"],
                "turn_idx": state["turn_idx"],
                "action": "",
                "kind": "empty",
                "observation": "[no action emitted - ending episode]",
            }
            state["trace"].append(trace_item)
            return {
                "observation": trace_item["observation"],
                "done": True,
                "won": False,
                "reward": "(0,)",
                "executed_action": "",
            }

        # Synthesized advance sentinel from _extract_actions (non-decodable
        # response in the official BFCL protocol).
        if str(action).strip() == self._TURN_ADVANCE_SENTINEL:
            return self._advance_turn(state, raw=raw_text, kind="turn_advance_undecodable")

        # Parse the batch. If nothing parseable (shouldn't happen because
        # _extract_actions guards, but methods may call step_episode
        # directly), treat as turn-advance per official protocol.
        calls = parse_function_calls(str(action))
        if not calls:
            return self._advance_turn(state, raw=raw_text, kind="turn_advance_undecodable")

        # Explicit legacy done_turn() among the parsed calls: treat as
        # turn-advance (back-compat with earlier prompt versions).
        if any(self._call_name(c) in self._TURN_ADVANCE_NAMES for c in calls):
            return self._advance_turn(state, raw=raw_text, kind="turn_end_explicit")

        # Step-limit-per-turn exhausted before this batch even ran — advance.
        if state["steps_in_turn"] >= self.max_steps_per_turn:
            return self._advance_turn(state, raw=raw_text, kind="turn_forced")

        # Execute each call in order. Stop at per-turn or hard episode cap.
        results = []
        for call_str in calls:
            if state["total_steps"] >= self.max_steps:
                state["done"] = True
                break
            if state["steps_in_turn"] >= self.max_steps_per_turn:
                break
            ok, out = _exec_one_call(call_str, state["backends"])
            state["total_steps"] += 1
            state["steps_in_turn"] += 1
            if ok:
                state["executed_actions"].append(call_str)
                obs_text = self._format_call_result(out)
                # Surface backend-returned `{error: ...}` dicts (many BFCL
                # backends return error strings instead of raising).
                if isinstance(out, dict) and out.get("error"):
                    state["model_errors"].append({"call": call_str, "error": str(out.get("error"))})
            else:
                state["model_errors"].append({"call": call_str, "error": str(out)})
                obs_text = f"[exec_error] {out}"
            results.append({"call": call_str, "result": obs_text, "ok": ok})
            state["trace"].append({
                "step": state["total_steps"],
                "turn_idx": state["turn_idx"],
                "action": call_str,
                "kind": "call_ok" if ok else "call_err",
                "observation": obs_text,
            })

        obs = self._format_batch_observation(results)
        if state["total_steps"] >= self.max_steps:
            state["done"] = True
        return {
            "observation": obs,
            "done": bool(state["done"]),
            "won": False,
            "reward": "(0,)",
            "executed_action": "\n".join(r["call"] for r in results if r["ok"]),
        }

    def finish_episode(self):
        state = self._episode_state
        if not isinstance(state, dict):
            self._last_eval_info = {}
            return {
                "score": 0.0,
                "success": False,
                "steps_executed": 0,
                "executed_actions": [],
                "parsed_actions": [],
                "last_observation": "",
                "eval_trace": [],
            }
        entry = state["entry"]
        gt_calls = _flatten_ground_truth(entry)

        model_states = {
            cls: _serialize_state(inst) for cls, inst in state["backends"].items()
        }
        gt_states, gt_executed, gt_errors = self._replay(entry, gt_calls)
        state_ok = model_states == gt_states

        norm_model = [_normalize_call_string(c) for c in state["executed_actions"]]
        norm_gt = [_normalize_call_string(c) for c in gt_calls]
        subset_ok = _is_subsequence(norm_gt, norm_model)
        score = 1.0 if (state_ok and subset_ok) else 0.0

        last_obs = state["trace"][-1].get("observation", "") if state["trace"] else ""
        self._last_eval_info = {
            "steps_executed": len(state["executed_actions"]),
            "parsed_actions": list(state["parsed_actions"]),
            "executed_actions": list(state["executed_actions"]),
            "model_errors": list(state["model_errors"]),
            "gt_executed": list(gt_executed),
            "gt_errors": gt_errors,
            "state_match": bool(state_ok),
            "subset_match": bool(subset_ok),
            "turns_completed": int(state["turn_idx"]),
            "total_turns": len(state["user_turns"]),
            "last_observation": last_obs,
        }
        if self.debug_eval:
            self._last_eval_info["model_states"] = model_states
            self._last_eval_info["gt_states"] = gt_states
            self._last_eval_info["eval_trace"] = list(state["trace"])

        result = {
            "score": score,
            "success": bool(state_ok and subset_ok),
            "steps_executed": len(state["executed_actions"]),
            "executed_actions": list(state["executed_actions"]),
            "parsed_actions": list(state["parsed_actions"]),
            "state_match": bool(state_ok),
            "subset_match": bool(subset_ok),
            "last_observation": last_obs,
            "eval_trace": list(state["trace"]),
        }
        self._episode_state = None
        return result

    # ----- evaluation (single-shot fallback for non-interactive methods) -----

    def _replay(self, entry, call_strings):
        """Run call_strings against a fresh backend snapshot. Returns
        (final_states_per_class, executed_ok_list, errors_list)."""
        involved = list(entry.get("involved_classes", []))
        initial_config = entry.get("initial_config", {}) or {}
        backends = _instantiate_backends(involved, initial_config)
        executed = []
        errors = []
        for call in call_strings:
            ok, out = _exec_one_call(call, backends)
            if ok:
                executed.append(call)
            else:
                errors.append({"call": call, "error": str(out)})
        states = {cls: _serialize_state(inst) for cls, inst in backends.items()}
        return states, executed, errors

    def evaluate_entry(self, output, entry):
        model_calls = parse_function_calls(output)
        gt_calls = _flatten_ground_truth(entry)

        model_states, model_executed, model_errors = self._replay(entry, model_calls)
        gt_states, gt_executed, gt_errors = self._replay(entry, gt_calls)

        state_ok = model_states == gt_states

        norm_model = [_normalize_call_string(c) for c in model_executed]
        norm_gt = [_normalize_call_string(c) for c in gt_calls]
        subset_ok = _is_subsequence(norm_gt, norm_model)

        score = 1.0 if (state_ok and subset_ok) else 0.0

        self._last_eval_info = {
            "steps_executed": len(model_executed),
            "parsed_actions": list(model_calls),
            "executed_actions": list(model_executed),
            "model_errors": model_errors,
            "gt_executed": list(gt_executed),
            "gt_errors": gt_errors,
            "state_match": bool(state_ok),
            "subset_match": bool(subset_ok),
        }
        if self.debug_eval:
            self._last_eval_info["model_states"] = model_states
            self._last_eval_info["gt_states"] = gt_states
        return score

    def build_memory_record(self, entry, output, feedback, score):
        info = dict(self._last_eval_info or {})
        record = {
            "qid": entry.get("_qid", ""),
            "question": "\n".join(self._collect_user_messages(entry)),
            "involved_classes": list(entry.get("involved_classes", [])),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
            "parsed_actions": info.get("parsed_actions", []),
            "executed_actions": info.get("executed_actions", []),
            "steps_executed": info.get("steps_executed", 0),
            "state_match": info.get("state_match", False),
            "subset_match": info.get("subset_match", False),
            "turns_completed": info.get("turns_completed", 0),
            "total_turns": info.get("total_turns", 0),
        }
        if self.debug_eval:
            record["eval_debug"] = {
                "model_errors": info.get("model_errors", []),
                "gt_errors": info.get("gt_errors", []),
                "model_states": info.get("model_states", {}),
                "gt_states": info.get("gt_states", {}),
                "eval_trace": info.get("eval_trace", []),
            }
        return record

    def get_prompt(self):
        raise NotImplementedError("BFCL uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("BFCL uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("BFCL uses evaluate_entry(output, entry).")
