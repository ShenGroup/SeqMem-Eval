"""APIBench AST evaluator (stdlib port of Gorilla's tree-sitter logic).

Port of `gorilla/eval/eval-scripts/ast_eval_{hf,tf,th}.py` to Python's
built-in `ast` module. Drops the tree-sitter + compiled-grammar dependency.

Matching rule preserved from upstream:
  - Extract the model's predicted `api_call` text from its free-form output.
  - Parse with `ast.parse`; walk every `ast.Call`.
  - For each pool entry: parse its `api_call`, take the dotted function name
    and a subset-specific set of "required arg" text snippets.
  - A candidate matches a pool entry iff some candidate Call has the same
    function name AND every required-arg substring appears in the candidate's
    unparsed text (single-quote stripped, one layer).
  - Functionality-correct iff the matched pool entry's `domain` equals the
    gold sample's `api_data.domain` (strict string equality, matching upstream).

Public API:
    evaluate_prediction(output_text, gold_api_data, api_pool, subset) -> dict
        Returns {
            "correct": bool,
            "hallucinated": bool,
            "matched_idx": Optional[int],
            "predicted_call_text": str,
            "reason": str,   # "correct" | "wrong_domain" | "no_pool_match"
                             # | "no_call" | "parse_error"
        }
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple


Subset = str  # "hf" | "tf" | "th"


# --------------------------------------------------------------------------- #
# Output extraction (mirrors Gorilla's split("api_call") logic)
# --------------------------------------------------------------------------- #


def extract_candidate_api_call(output_text: str) -> str:
    """Extract the predicted api_call substring from a model output.

    Upstream logic:
      - Split by literal "api_call". If only 1 chunk: treat whole output as
        candidate (HF behavior; we extend this to TF/TH for forgiveness).
      - Else take chunks[1], split by "api_provider", take [0].
      - Slice from first ":" to last ")" inclusive, i.e. the value part.
    """
    if output_text is None:
        return ""
    text = str(output_text)
    parts = text.split("api_call")
    if len(parts) == 1:
        return text
    tail = parts[1].split("api_provider")[0]
    start = tail.index(":") if ":" in tail else 0
    end = tail.rindex(")") if ")" in tail else -2
    # Upstream uses [start + 2 : end + 1]; preserve that exactly.
    return tail[start + 2 : end + 1]


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #


def _dotted_func_name(call: ast.Call) -> str:
    """Return "a.b.c" style dotted name for a Call's func, or "" on failure."""
    node: ast.AST = call.func
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        # rare: chained call like f()(); fall back to unparsed func
        try:
            return ast.unparse(call.func)
        except Exception:
            return ""
    else:
        try:
            return ast.unparse(call.func)
        except Exception:
            return ""
    parts.reverse()
    return ".".join(parts)


def _all_calls(tree: ast.AST) -> List[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _required_args_for_reference(ref_call: ast.Call, subset: Subset) -> List[str]:
    """Extract subset-specific required-arg text snippets from a reference call.

    Mirrors upstream Gorilla `get_args` per subset:

    - hf: every positional arg as full text; for every kwarg, the RHS only.
    - tf: every positional arg as full text; for kwargs named "model" (or
      "model " in source form), the RHS only; for every other kwarg, the
      full "key=value" text. (Upstream iterates the arg-list tree and
      appends each child's full text unless the child is a `model=` kwarg.)
    - th: only the RHS of kwargs named "repo_or_dir" or "model"; positional
      args and other kwargs are ignored entirely.
    """
    out: List[str] = []
    subset = subset.lower()
    if subset == "hf":
        for a in ref_call.args:
            try:
                out.append(ast.unparse(a))
            except Exception:
                continue
        for kw in ref_call.keywords:
            if kw.arg is None:
                continue
            try:
                out.append(ast.unparse(kw.value))
            except Exception:
                continue
    elif subset == "tf":
        for a in ref_call.args:
            try:
                out.append(ast.unparse(a))
            except Exception:
                continue
        for kw in ref_call.keywords:
            if kw.arg is None:
                continue
            try:
                if kw.arg == "model":
                    out.append(ast.unparse(kw.value))
                else:
                    out.append(f"{kw.arg}={ast.unparse(kw.value)}")
            except Exception:
                continue
    elif subset == "th":
        for kw in ref_call.keywords:
            if kw.arg in ("repo_or_dir", "model"):
                try:
                    out.append(ast.unparse(kw.value))
                except Exception:
                    continue
    else:
        raise ValueError(f"Unknown APIBench subset: {subset!r}")
    return out


def _strip_one_quote_layer(s: str) -> str:
    """Match upstream `arg.decode().lstrip("'").rstrip("'")`.

    Note: Python's str.lstrip / rstrip strip *all* consecutive matching
    characters, not one layer. Preserve upstream behavior exactly.
    """
    return s.lstrip("'").rstrip("'")


def _parse_single_call(call_text: str) -> Optional[ast.Call]:
    """Parse a reference api_call string; return its top-level ast.Call or None.

    The reference api_calls in the pool are typically single expressions like
    `torch.hub.load(...)` or `AutoModel.from_pretrained(...)`. We accept any
    module with at least one Call node and return the first one found.
    """
    try:
        module = ast.parse(call_text.strip())
    except SyntaxError:
        return None
    calls = _all_calls(module)
    return calls[0] if calls else None


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def _match_candidate_against_pool(
    candidate_calls: List[ast.Call],
    pool: List[Dict[str, Any]],
    subset: Subset,
) -> Optional[int]:
    """Return the first pool index that matches, else None.

    For each pool entry we parse its `api_call`, extract the dotted function
    name and required-arg substrings, and search for a candidate Call with the
    same dotted name whose unparsed text contains every required substring.
    """
    # Precompute unparsed text per candidate (also their dotted names).
    cand_texts: List[Tuple[str, str]] = []
    for c in candidate_calls:
        try:
            cand_texts.append((_dotted_func_name(c), ast.unparse(c)))
        except Exception:
            continue

    for idx, entry in enumerate(pool):
        ref_src = entry.get("api_call")
        if not isinstance(ref_src, str) or not ref_src.strip():
            continue
        ref_call = _parse_single_call(ref_src)
        if ref_call is None:
            continue
        ref_name = _dotted_func_name(ref_call)
        if not ref_name:
            continue
        required = _required_args_for_reference(ref_call, subset)
        if not required:
            # Upstream skips entries with empty required-args.
            continue
        # Find a candidate Call with matching dotted function name.
        chosen_text: Optional[str] = None
        for cand_name, cand_text in cand_texts:
            if cand_name == ref_name:
                chosen_text = cand_text
                break
        if chosen_text is None:
            continue
        # All required-arg substrings must appear (after one quote-strip layer).
        ok = True
        for arg_text in required:
            needle = _strip_one_quote_layer(arg_text)
            if needle not in chosen_text:
                ok = False
                break
        if ok:
            return idx
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def evaluate_prediction(
    output_text: str,
    gold_api_data: Dict[str, Any],
    api_pool: List[Dict[str, Any]],
    subset: Subset,
) -> Dict[str, Any]:
    """Score a single prediction.

    Returns a dict with:
      correct:              bool  — AST-match AND domain-match
      hallucinated:         bool  — no pool entry matched (or parse-fail/no-call)
      matched_idx:          Optional[int]  — pool index if matched
      predicted_call_text:  str   — extracted candidate text
      reason:               str   — one of
                                    "correct", "wrong_domain",
                                    "no_pool_match", "no_call", "parse_error"
    """
    predicted_call_text = extract_candidate_api_call(output_text)
    info: Dict[str, Any] = {
        "correct": False,
        "hallucinated": False,
        "matched_idx": None,
        "predicted_call_text": predicted_call_text,
        "reason": "",
    }

    try:
        tree = ast.parse(predicted_call_text)
    except SyntaxError:
        info["hallucinated"] = True
        info["reason"] = "parse_error"
        return info

    calls = _all_calls(tree)
    if not calls:
        info["hallucinated"] = True
        info["reason"] = "no_call"
        return info

    matched_idx = _match_candidate_against_pool(calls, api_pool, subset)
    if matched_idx is None:
        info["hallucinated"] = True
        info["reason"] = "no_pool_match"
        return info

    info["matched_idx"] = int(matched_idx)
    ref_domain = api_pool[matched_idx].get("domain")
    gold_domain = (gold_api_data or {}).get("domain")
    if ref_domain == gold_domain:
        info["correct"] = True
        info["reason"] = "correct"
    else:
        info["correct"] = False
        info["reason"] = "wrong_domain"
    return info
