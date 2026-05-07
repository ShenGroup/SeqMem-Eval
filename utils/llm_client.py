from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Thread-local stash for the most recent generate() call's reasoning data.
# OpenRouter returns hidden thinking tokens via message.reasoning and a
# usage.completion_tokens_details.reasoning_tokens count; both are dropped by
# the chat-completions API surface (.message.content). We park them here so
# the runner can append to memory records without changing generate()'s
# return type.
_LAST_CALL: "threading.local" = threading.local()


def _reset_last_call() -> None:
    _LAST_CALL.reasoning_content = ""
    _LAST_CALL.reasoning_tokens = None
    _LAST_CALL.completion_tokens = None


def _accumulate(reasoning_tokens, completion_tokens) -> None:
    cur_r = getattr(_LAST_CALL, "reasoning_tokens_total", 0) or 0
    cur_c = getattr(_LAST_CALL, "completion_tokens_total", 0) or 0
    if reasoning_tokens:
        cur_r += int(reasoning_tokens)
    if completion_tokens:
        cur_c += int(completion_tokens)
    _LAST_CALL.reasoning_tokens_total = cur_r
    _LAST_CALL.completion_tokens_total = cur_c


def get_last_call_reasoning() -> Dict[str, Optional[object]]:
    """Return reasoning_content / reasoning_tokens / completion_tokens
    captured by the most recent generate() on this thread. Empty if the
    last call did not surface any reasoning data."""
    return {
        "reasoning_content": getattr(_LAST_CALL, "reasoning_content", "") or "",
        "reasoning_tokens":  getattr(_LAST_CALL, "reasoning_tokens",  None),
        "completion_tokens": getattr(_LAST_CALL, "completion_tokens", None),
    }


def reset_call_accumulators() -> None:
    """Zero per-sample reasoning/completion token totals (call before each sample)."""
    _LAST_CALL.reasoning_tokens_total = 0
    _LAST_CALL.completion_tokens_total = 0


def get_call_totals() -> Dict[str, int]:
    return {
        "reasoning_tokens_total":  getattr(_LAST_CALL, "reasoning_tokens_total", 0) or 0,
        "completion_tokens_total": getattr(_LAST_CALL, "completion_tokens_total", 0) or 0,
    }

# Keep transformers on PyTorch path in mixed environments.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


def _load_dotenv_once() -> None:
    """Populate os.environ from a repo-root .env file (no external dep).

    Existing env vars win over .env, so shell exports still override the file.
    """
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    except OSError:
        pass


_load_dotenv_once()

try:
    from openai import OpenAI

    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False


_CACHE: Dict[str, Tuple[object, object]] = {}
_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")


def _resolve_provider(model_name: str) -> str:
    model_key = model_name.strip().lower()
    if model_key.startswith("openrouter/") or model_key == "openrouter":
        return "openrouter"
    if model_key.startswith("openai/") or model_key in {"openai", "chatgpt"}:
        return "openai"
    return "local"


def _resolve_openrouter_model(model_name: str) -> str:
    model_key = model_name.strip()
    if model_key.lower().startswith("openrouter/"):
        remainder = model_key.split("/", 1)[1]
        if not remainder:
            raise RuntimeError(
                "OpenRouter model name must include a provider/model suffix, "
                "e.g. openrouter/openai/gpt-5.2"
            )
        return remainder
    if model_key.lower() == "openrouter":
        default = os.getenv("OPENROUTER_MODEL")
        if not default:
            raise RuntimeError(
                "OPENROUTER_MODEL is not set. Either pass "
                "--generation-model openrouter/<provider>/<model> "
                "or export OPENROUTER_MODEL=<provider>/<model>."
            )
        return default
    return model_key


def _generate_openrouter(
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    if not _HAS_OPENAI:
        raise RuntimeError("openai SDK not installed. Please install: pip install openai")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env or export it in your shell."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # max_retries=0 disables the openai SDK's internal retry loop so connection
    # errors propagate quickly to our outer max_retries loop instead of hanging
    # for hours in the SDK retry cycle. timeout caps individual request wall
    # time so a single hung request can't block a worker forever.
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        timeout=float(os.getenv("OPENROUTER_REQUEST_TIMEOUT", "180")),
    )
    or_model = _resolve_openrouter_model(model_name)

    extra_headers: Dict[str, str] = {}
    referer = os.getenv("OPENROUTER_SITE_URL")
    title = os.getenv("OPENROUTER_SITE_NAME")
    if referer:
        extra_headers["HTTP-Referer"] = referer
    if title:
        extra_headers["X-OpenRouter-Title"] = title

    extra_body: Dict[str, object] = {}
    provider_env = os.getenv("OPENROUTER_PROVIDER", "").strip()
    if provider_env:
        providers = [p.strip() for p in provider_env.split(",") if p.strip()]
        if providers:
            extra_body["provider"] = {
                "only": providers,
                "allow_fallbacks": False,
            }

    reasoning_env = os.getenv("OPENROUTER_REASONING", "").strip().lower()
    if reasoning_env in {"off", "disable", "disabled", "false", "0", "no"}:
        # Documented disable path per OpenRouter reasoning-tokens guide.
        extra_body["reasoning"] = {"effort": "none"}
    elif reasoning_env in {"on", "enable", "enabled", "true", "1", "yes"}:
        extra_body["reasoning"] = {"enabled": True}
    elif reasoning_env in {"low", "medium", "high"}:
        # Reasoning-only models (e.g. minimax-m2.7) ignore enabled=false;
        # pinning effort is the documented way to tune reasoning token budget.
        extra_body["reasoning"] = {"effort": reasoning_env}

    # Qwen3.5-specific: toggle non-thinking mode via chat_template_kwargs.
    # Per the HF Qwen3.5-9B model card, this is the authoritative switch
    # (/think or /nothink are not supported on Qwen3.5).
    enable_thinking_env = os.getenv("OPENROUTER_ENABLE_THINKING", "").strip().lower()
    if enable_thinking_env in {"off", "false", "0", "no", "disable", "disabled"}:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    elif enable_thinking_env in {"on", "true", "1", "yes", "enable", "enabled"}:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True

    # Sampling params. OpenAI Python SDK accepts top_p / presence_penalty /
    # frequency_penalty as kwargs; top_k / min_p / repetition_penalty are
    # OpenRouter-only so they go via extra_body (where OpenRouter forwards
    # them as top-level fields to the upstream provider).
    sampling_kwargs: Dict[str, float] = {}

    def _env_float(name: str):
        raw = os.getenv(name, "").strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    for name, key in (
        ("OPENROUTER_TOP_P", "top_p"),
        ("OPENROUTER_PRESENCE_PENALTY", "presence_penalty"),
        ("OPENROUTER_FREQUENCY_PENALTY", "frequency_penalty"),
    ):
        val = _env_float(name)
        if val is not None:
            sampling_kwargs[key] = val

    for name, key in (
        ("OPENROUTER_TOP_K", "top_k"),
        ("OPENROUTER_MIN_P", "min_p"),
        ("OPENROUTER_REPETITION_PENALTY", "repetition_penalty"),
    ):
        val = _env_float(name)
        if val is not None:
            extra_body[key] = val

    # NOTE 15 retries (was 5): connection-level errors from Cloudflare /
    # OpenRouter occasionally cluster for ~1-2 minutes during a sweep, and 5
    # retries (capped 31s of total backoff) was not enough — we lost three
    # long-running procs (qwen3-8b ALFWorld, minimax ALFWorld, qwen3-8b
    # MMLU-Pro-Engineering at 782/873). Bumping to 15 keeps a single sample
    # alive across a ~10-min outage.
    max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "15"))
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            prefill = os.getenv("OPENROUTER_ASSISTANT_PREFILL", "")
            if prefill:
                # Decode literal \n so a plain env value like
                # '<think>\n\n</think>\n\n' works without shell-quoting gymnastics.
                prefill = prefill.encode("utf-8").decode("unicode_escape")
                messages.append({"role": "assistant", "content": prefill})
            kwargs = {
                "model": or_model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": float(temperature),
            }
            kwargs.update(sampling_kwargs)
            if extra_headers:
                kwargs["extra_headers"] = extra_headers
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content if resp.choices else ""
            # Capture reasoning_content + reasoning_tokens (OpenRouter
            # returns these alongside content; we surface them via the
            # thread-local stash so the runner can persist them).
            try:
                msg = resp.choices[0].message if resp.choices else None
                reasoning = getattr(msg, "reasoning", None) if msg is not None else None
                if reasoning is None and msg is not None and hasattr(msg, "model_extra"):
                    reasoning = (msg.model_extra or {}).get("reasoning")
                _LAST_CALL.reasoning_content = (reasoning or "")
            except Exception:
                _LAST_CALL.reasoning_content = ""
            try:
                usage = getattr(resp, "usage", None)
                ctd = getattr(usage, "completion_tokens_details", None) if usage else None
                if ctd is None and usage is not None and hasattr(usage, "model_extra"):
                    ctd_d = (usage.model_extra or {}).get("completion_tokens_details") or {}
                    rt = ctd_d.get("reasoning_tokens") if isinstance(ctd_d, dict) else None
                else:
                    rt = getattr(ctd, "reasoning_tokens", None) if ctd is not None else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                _LAST_CALL.reasoning_tokens = rt
                _LAST_CALL.completion_tokens = ct
                _accumulate(rt, ct)
            except Exception:
                _LAST_CALL.reasoning_tokens = None
                _LAST_CALL.completion_tokens = None
            return (content or "").strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"OpenRouter error (try {attempt + 1}/{max_retries}): {exc}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(
                    f"OpenRouter request failed after retries: {exc}"
                ) from exc

    return ""


def _resolve_local_backend() -> str:
    backend = os.getenv("LLM_BACKEND", "transformers").strip().lower()
    if backend not in {"auto", "vllm", "transformers"}:
        raise RuntimeError(
            "Invalid LLM_BACKEND. Expected one of: auto, vllm, transformers"
        )
    return backend


def _resolve_openai_model(model_name: str) -> str:
    model_key = model_name.strip()
    if model_key.lower().startswith("openai/"):
        return model_key.split("/", 1)[1]
    if model_key.lower() in {"openai", "chatgpt"}:
        return os.getenv("OPENAI_MODEL", "gpt-5.2")
    return model_key


def _generate_openai(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    if not _HAS_OPENAI:
        raise RuntimeError("openai not installed. Please install: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    openai_model = _resolve_openai_model(model_name)

    max_retries = 5
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=openai_model,
                input=prompt,
                max_output_tokens=max_new_tokens,
                reasoning={"effort": "none"},
            )
            return (resp.output_text or "").strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"OpenAI error (try {attempt + 1}/{max_retries}): {exc}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(f"OpenAI request failed after retries: {exc}") from exc

    return ""


def _load_local_model_with_vllm(model_name: str):
    key = f"{model_name}|vllm"
    tok, llm = _CACHE.get(key, (None, None))
    if tok is not None and llm is not None:
        return tok, llm

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=_CACHE_DIR,
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9,
    }
    # By default we do not pass max_model_len, so vLLM can derive it from model config.
    user_max_model_len = os.getenv("VLLM_MAX_MODEL_LEN")
    if user_max_model_len:
        llm_kwargs["max_model_len"] = int(user_max_model_len)

    llm = LLM(**llm_kwargs)
    _CACHE[key] = (tok, llm)
    return tok, llm


def _generate_local_with_vllm(
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    tok, llm = _load_local_model_with_vllm(model_name)

    if hasattr(tok, "apply_chat_template"):
        chat_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        chat_text = prompt

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=float(temperature),
        n=1,
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
    )
    outputs = llm.generate([chat_text], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text.strip()


def _load_local_model_with_transformers(model_name: str):
    key = f"{model_name}|hf"
    tok, mdl = _CACHE.get(key, (None, None))
    if tok is not None and mdl is not None:
        return tok, mdl

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=_CACHE_DIR,
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=_CACHE_DIR,
    )
    mdl.eval()
    _CACHE[key] = (tok, mdl)
    return tok, mdl


def _generate_local_with_transformers(
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    tok, mdl = _load_local_model_with_transformers(model_name)

    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    if hasattr(tok, "apply_chat_template"):
        chat_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        chat_text = prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    inputs = tok(chat_text, return_tensors="pt").to(device)
    out_ids = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=float(temperature),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def generate(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
) -> str:
    _reset_last_call()
    provider = _resolve_provider(model_name)
    if provider == "openai":
        return _generate_openai(model_name, prompt, max_new_tokens)
    if provider == "openrouter":
        return _generate_openrouter(model_name, prompt, temperature, max_new_tokens)

    if not _HAS_TRANSFORMERS:
        raise RuntimeError(
            "Local generation requires transformers and torch. "
            "Please install: pip install transformers torch"
        )

    backend = _resolve_local_backend()

    if backend == "vllm":
        if not _HAS_VLLM:
            raise RuntimeError(
                "LLM_BACKEND=vllm is set, but vllm is not available. "
                "Please install: pip install vllm"
            )
        return _generate_local_with_vllm(model_name, prompt, temperature, max_new_tokens)

    if backend == "transformers":
        return _generate_local_with_transformers(model_name, prompt, temperature, max_new_tokens)

    # auto: prefer vLLM, fallback to transformers when vLLM fails at runtime.
    if _HAS_VLLM:
        try:
            return _generate_local_with_vllm(
                model_name,
                prompt,
                temperature,
                max_new_tokens,
            )
        except Exception as exc:
            print(f"vLLM failed, fallback to transformers: {exc}")

    return _generate_local_with_transformers(
        model_name,
        prompt,
        temperature,
        max_new_tokens,
    )
