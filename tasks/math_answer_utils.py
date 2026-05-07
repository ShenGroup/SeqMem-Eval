import re


def extract_last_boxed(text):
    if text is None:
        return ""
    raw = str(text)
    token = r"\boxed{"
    start_positions = []
    offset = 0
    while True:
        idx = raw.find(token, offset)
        if idx < 0:
            break
        start_positions.append(idx)
        offset = idx + 1
    if not start_positions:
        return ""

    start = start_positions[-1] + len(token)
    depth = 1
    i = start
    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i].strip()
        i += 1
    return ""


def _strip_outer_delimiters(text):
    s = text.strip()
    changed = True
    while changed and len(s) >= 2:
        changed = False
        if (s.startswith("$") and s.endswith("$")) or (
            s.startswith("\\(") and s.endswith("\\)")
        ) or (s.startswith("\\[") and s.endswith("\\]")):
            if s.startswith("$"):
                s = s[1:-1].strip()
            else:
                s = s[2:-2].strip()
            changed = True
    return s


def _strip_balanced_outer_braces(text):
    s = text.strip()
    while s.startswith("{") and s.endswith("}"):
        depth = 0
        balanced = True
        for idx, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
                if depth == 0 and idx != len(s) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        s = s[1:-1].strip()
    return s


def normalize_math_expr(text):
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    s = _strip_outer_delimiters(s)
    s = _strip_balanced_outer_braces(s)
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    s = s.replace(r"\,", "")
    s = s.replace(r"\!", "")
    s = s.strip()
    s = re.sub(r"[.;:,]+$", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def normalized_last_boxed(text):
    return normalize_math_expr(extract_last_boxed(text))


def normalized_gold_answer(answer_text, fallback_solution_text=None):
    primary = str(answer_text or "").strip()
    if primary:
        if r"\boxed{" in primary:
            return normalized_last_boxed(primary)
        return normalize_math_expr(primary)
    if fallback_solution_text:
        return normalized_last_boxed(fallback_solution_text)
    return ""
