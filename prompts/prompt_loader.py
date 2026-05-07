import json
from pathlib import Path

from config import PROMPT_FILE


def load_prompt_templates(prompt_file=None):
    path = Path(prompt_file) if prompt_file else PROMPT_FILE
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_prompt(template, **kwargs):
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def get_prompt(prompt_key, prompt_file=None, **kwargs):
    templates = load_prompt_templates(prompt_file)
    template = templates.get(prompt_key, prompt_key)
    return render_prompt(template, **kwargs)
