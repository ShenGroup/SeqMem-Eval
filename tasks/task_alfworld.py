import json
import re
from pathlib import Path

from tasks.alfworld_fewshots import ALFWORLD_FEWSHOTS
from tasks.base_task import BaseTask


def _extract_tag_content(text, tag_name):
    if not text:
        return ""
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    if start_tag not in text or end_tag not in text:
        return ""
    try:
        return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    except Exception:
        return ""


def _process_observation(obs):
    obs = str(obs or "")
    if obs.startswith("You arrive at loc "):
        dot = obs.find(". ")
        if dot >= 0:
            return obs[dot + 2 :]
    return obs


def _rewrite_local_data_path(data_root, value):
    text = str(value or "").strip()
    if not text:
        return value
    path = Path(text)
    if path.is_absolute():
        return str(path)
    if text.startswith("data/alfworld/"):
        relative = Path(*path.parts[2:]) if len(path.parts) >= 2 else Path()
        return str((data_root / relative).resolve())
    return str((data_root / path).resolve())


def _split_compact_action_line(line):
    # Common ALFWorld-style command starters.
    starters = {
        "go",
        "open",
        "close",
        "take",
        "put",
        "move",
        "drop",
        "examine",
        "look",
        "clean",
        "cool",
        "heat",
        "slice",
        "inventory",
    }
    tokens = line.split()
    if not tokens:
        return []
    segments = []
    current = []
    for idx, tok in enumerate(tokens):
        lower = tok.lower()
        is_new_start = idx > 0 and lower in starters
        if is_new_start and current:
            segments.append(" ".join(current).strip())
            current = [tok]
        else:
            current.append(tok)
    if current:
        segments.append(" ".join(current).strip())
    return [s for s in segments if s]


def _normalize_action_candidate(raw_line):
    line = str(raw_line or "").strip()
    if not line:
        return ""
    if line.startswith(">"):
        line = re.sub(r"^>\s*", "", line).strip()
    line = re.sub(r"^\d+\.\s*", "", line).strip()
    if line.lower().startswith("action:"):
        line = line.split(":", 1)[1].strip()
    return line


_ALFWORLD_ACTION_RE = re.compile(
    r"^(go to|open|close|take|put|move|use|heat|cool|look|clean|inventory|examine|drop|slice)\b",
    re.IGNORECASE,
)

_ALFWORLD_ENV_NAMES = [
    "pick_and_place",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "look_at_obj",
    "pick_two_obj",
]


def _normalize_task_main(goal_text):
    text = str(goal_text or "").strip()
    if not text:
        return ""
    m = re.search(r"Your task is to:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()
    text = re.sub(r"___\d+\s*$", "", text).strip()
    return text


class ALFWorldTask(BaseTask):
    name = "ALFWorld"
    prompt_key = "alfworld"
    action_mode = True

    def __init__(
        self,
        prompt_file=None,
        tasks_file=None,
        config_file=None,
        max_steps=30,
        data_file=None,
    ):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        self.data_root = root / "data" / "alfworld"
        # `data_file` is an alias of `tasks_file` so --task-data-override (which
        # dispatches as data_file=) can pin a holdout tasks JSON.
        explicit = tasks_file or data_file
        self._explicit_tasks_file = Path(explicit) if explicit else None
        self.tasks_file = self._explicit_tasks_file or (self.data_root / "alfworld_tasks_suffix.json")
        self.config_file = Path(config_file) if config_file else self.data_root / "alfworld.yaml"
        self.max_steps = int(max_steps)
        self.alfworld_fewshot_num = 1
        self.debug_eval = False
        # "ood" -> eval_out_of_distribution (valid_unseen, historical default).
        # "id"  -> eval_in_distribution   (valid_seen).
        self.split = "ood"

        self._entries = None
        self._alfworld_cfg = None
        self._main_env = None
        self._last_eval_info = {}
        self._fewshots_table = None
        self._episode_state = None

    def set_split(self, split):
        """Switch between 'ood' (valid_unseen) and 'id' (valid_seen).

        Updates both the tasks JSON file and the alfworld env `split` key so
        AlfredTWEnv picks the corresponding data_path. Only overrides the
        tasks file if the caller didn't pin one explicitly via `tasks_file`.
        """
        s = str(split or "").strip().lower()
        if s not in ("id", "ood"):
            raise ValueError(f"alfworld split must be 'id' or 'ood', got {split!r}")
        self.split = s
        if self._explicit_tasks_file is None:
            if s == "id":
                self.tasks_file = self.data_root / "alfworld_tasks_suffix_indist.json"
            else:
                self.tasks_file = self.data_root / "alfworld_tasks_suffix.json"
        # force-reload entries + cfg next access so split change takes effect
        self._entries = None
        self._alfworld_cfg = None
        self._main_env = None

    def _load_entries(self):
        if self._entries is not None:
            return
        if not self.tasks_file.exists():
            raise RuntimeError(
                f"ALFWorld task file not found: {self.tasks_file}. "
                "Please prepare local data under data/alfworld."
            )
        with self.tasks_file.open("r", encoding="utf-8") as f:
            self._entries = json.load(f)

    def _load_cfg(self):
        if self._alfworld_cfg is not None:
            return
        if not self.config_file.exists():
            raise RuntimeError(
                f"ALFWorld config file not found: {self.config_file}. "
                "Expected local config in data/alfworld/alfworld.yaml."
            )
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("ALFWorld task requires PyYAML. Install with: pip install pyyaml") from exc
        with self.config_file.open("r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        dataset_cfg = cfg_dict.get("dataset", {})
        for key in ["data_path", "eval_id_data_path", "eval_ood_data_path"]:
            if key in dataset_cfg and dataset_cfg[key]:
                dataset_cfg[key] = _rewrite_local_data_path(self.data_root, dataset_cfg[key])
        logic_cfg = cfg_dict.get("logic", {})
        for key in ["domain", "grammar"]:
            if key in logic_cfg and logic_cfg[key]:
                logic_cfg[key] = _rewrite_local_data_path(self.data_root, logic_cfg[key])
        mask_cfg = cfg_dict.get("mask_rcnn", {})
        if "pretrained_model_path" in mask_cfg and mask_cfg["pretrained_model_path"]:
            mask_cfg["pretrained_model_path"] = _rewrite_local_data_path(
                self.data_root, mask_cfg["pretrained_model_path"]
            )
        self._alfworld_cfg = cfg_dict

    def _resolve_gamefile(self, gamefile):
        path = Path(str(gamefile))
        if path.is_absolute():
            return path
        if str(path).startswith("data/"):
            # task file uses "data/alfworld/..." style paths; map to local project data root.
            relative = Path(*path.parts[2:]) if len(path.parts) >= 2 else Path()
            return (self.data_root / relative).resolve()
        return (self.data_root / path).resolve()

    def _qid(self, item, idx):
        goal = str(item.get("goal", ""))
        m = re.search(r"___(\d+)\s*$", goal)
        if m:
            return m.group(1)
        return str(idx)

    def _env_name_from_gamefile(self, gamefile):
        text = str(gamefile or "")
        for name in _ALFWORLD_ENV_NAMES:
            if name in text:
                return name
        return "pick_and_place"

    def _load_fewshots_table(self):
        if self._fewshots_table is not None:
            return self._fewshots_table
        try:
            self._fewshots_table = {
                str(k): [str(s) for s in list(v)]
                for k, v in dict(ALFWORLD_FEWSHOTS).items()
            }
        except Exception:
            self._fewshots_table = {}
        return self._fewshots_table

    def _fewshots_for_entry(self, entry):
        env_name = self._env_name_from_gamefile(entry.get("gamefile", ""))
        table = self._load_fewshots_table()
        shots = list(table.get(env_name, []))[: self.alfworld_fewshot_num]
        return shots

    def iter_entries(self):
        self._load_entries()
        for idx, item in enumerate(self._entries):
            row = dict(item)
            row["_qid"] = self._qid(row, idx)
            yield row

    def total_entries(self):
        self._load_entries()
        return len(self._entries)

    def build_prompt(self, entry):
        # Output format and action grammar are owned by the shared system prompt
        # at prompts/alfworld/system_instruction.txt (consumed by the step-wise
        # methods). Keep this minimal so it doesn't collide with that contract.
        goal = str(entry.get("goal", "")).strip()
        return f"{goal}\n"

    def build_inputs(self, entry):
        return {
            "question": str(entry.get("goal", "")).strip(),
            "goal": str(entry.get("goal", "")).strip(),
            "gamefile": str(entry.get("gamefile", "")).strip(),
            "qid": entry.get("_qid", ""),
            "fewshots": self._fewshots_for_entry(entry),
        }

    def get_query(self, entry, inputs):
        query = _normalize_task_main(inputs.get("goal", ""))
        if query:
            return query
        return str(inputs.get("goal", ""))

    def _extract_actions(self, output):
        text = str(output or "")
        trajectory = _extract_tag_content(text, "trajectory")
        if trajectory:
            # Contract: reasoning goes inside <trajectory>...</trajectory>,
            # the executable action is emitted outside the tag (e.g.
            # GMemory's output_contract in autogen_gmemory_online.py). Strip
            # the trajectory block and parse the remainder; fall back to the
            # trajectory content only if nothing is left outside.
            stripped = re.sub(r"<trajectory>.*?</trajectory>", "", text, flags=re.DOTALL).strip()
            source = stripped if stripped else trajectory
        else:
            source = text
        actions = []
        chunks = []
        for raw in source.splitlines():
            line = raw.strip()
            if not line:
                continue
            # Drop reasoning lines in any supported format:
            #   "> think: ..." / "think: ..."  (ReAct-style)
            #   "Thought: ..."                  (graph-of-skills style)
            low = line.lower()
            if low.startswith("> think") or low.startswith("think:") or low.startswith("thought:"):
                continue
            line = _normalize_action_candidate(line)
            if not line:
                continue
            # After stripping numeric list prefixes etc., re-check for a bare "Thought:" line.
            if line.lower().startswith("thought:"):
                continue
            # Keep only explicit actions. This prevents model-generated observations
            # ("You see ...", "The drawer is closed.") from entering the action list.
            if not _ALFWORLD_ACTION_RE.match(line):
                continue
            # Split overly compact trajectories into candidate single actions.
            parts = re.split(r"\s*(?:;|, then | then )\s*", line, flags=re.IGNORECASE)
            for part in parts:
                if part.strip():
                    chunks.append(part.strip())
        for line in chunks:
            line = _normalize_action_candidate(line)
            if line.startswith("- "):
                line = line[2:].strip()
            if not line:
                continue
            if line.lower().startswith("think"):
                continue
            if line.startswith("<") and line.endswith(">"):
                continue
            compact_parts = _split_compact_action_line(line)
            if len(compact_parts) > 1:
                for p in compact_parts:
                    if _ALFWORLD_ACTION_RE.match(p):
                        actions.append(p.lower().strip())
            else:
                if _ALFWORLD_ACTION_RE.match(line):
                    actions.append(line.lower().strip())
        return actions

    def _ensure_env(self, gamefile):
        self._load_cfg()
        if not Path(gamefile).exists():
            raise RuntimeError(
                f"ALFWorld game file not found: {gamefile}. "
                "Please populate local game data under data/alfworld/json_2.1.1."
            )
        try:
            import alfworld.agents.environment as alfworld_environment
        except ImportError as exc:
            raise RuntimeError(
                "ALFWorld task requires alfworld package. "
                "Install and prepare ALFWorld data first."
            ) from exc
        env_cfg = self._alfworld_cfg.get("env", {}) if isinstance(self._alfworld_cfg, dict) else {}
        env_type = env_cfg.get("type", "AlfredTWEnv")
        env_cls = alfworld_environment.get_environment(env_type)
        if getattr(self, "split", "ood") == "id":
            split_name = "eval_in_distribution"
        else:
            split_name = self._alfworld_cfg.get("split", "eval_out_of_distribution")
        main_env = env_cls(self._alfworld_cfg, train_eval=split_name)
        main_env.game_files = [str(gamefile)]
        return main_env.init_env(batch_size=1)

    def _canonicalize_action(self, action):
        action = str(action or "").strip().lower()
        if action.startswith("put"):
            m = re.match(r"put (\w+\s*\d*) (?:in|on) (\w+\s*\d+)", action)
            if m is not None:
                left = re.sub(r"\s+", " ", m.group(1)).strip()
                right = re.sub(r"\s+", " ", m.group(2)).strip()
                action = f"put {left} in/on {right}"
        return action

    def _extract_done_won(self, done, info):
        if isinstance(info, dict):
            won_raw = info.get("won", [False])
            won = bool(won_raw[0]) if isinstance(won_raw, (list, tuple)) and won_raw else bool(won_raw)
        else:
            won_raw = None
            won = False
        if isinstance(done, (list, tuple)):
            done_flag = bool(done[0]) if done else False
        else:
            done_flag = bool(done)
        return bool(done_flag), bool(won), won_raw

    def start_episode(self, entry):
        gamefile = self._resolve_gamefile(entry.get("gamefile", ""))
        env = self._ensure_env(gamefile)
        reset_out = env.reset()
        observation = ""
        if isinstance(reset_out, (tuple, list)) and reset_out:
            first = reset_out[0]
            if isinstance(first, (list, tuple)):
                observation = str(first[0]) if first else ""
            else:
                observation = str(first)
        self._episode_state = {
            "env": env,
            "entry": dict(entry),
            "gamefile": str(gamefile),
            "initial_observation": observation,
            "last_observation": _process_observation(observation),
            "executed_actions": [],
            "parsed_actions": [],
            "trace": [],
            "success": False,
            "done": False,
            "steps": 0,
        }
        return self._episode_state["last_observation"]

    def step_episode(self, action, raw_action=None, meta=None):
        if not isinstance(self._episode_state, dict) or self._episode_state.get("env") is None:
            raise RuntimeError("ALFWorld episode is not started. Call start_episode(entry) first.")
        state = self._episode_state
        env = state["env"]
        if state.get("done", False):
            return {
                "observation": state.get("last_observation", ""),
                "done": True,
                "won": bool(state.get("success", False)),
                "reward": "(0,)",
                "executed_action": "",
            }

        canonical_action = self._canonicalize_action(action)
        if not canonical_action:
            return {
                "observation": state.get("last_observation", ""),
                "done": bool(state.get("done", False)),
                "won": bool(state.get("success", False)),
                "reward": "(0,)",
                "executed_action": "",
            }

        observation, reward, done, info = env.step([canonical_action])
        done_flag, won, won_raw = self._extract_done_won(done, info)
        obs_text = str(observation[0] if observation else "")
        state["parsed_actions"].append(str(raw_action or canonical_action))
        state["executed_actions"].append(canonical_action)
        state["steps"] = int(state.get("steps", 0)) + 1
        state["last_observation"] = _process_observation(obs_text)
        state["done"] = bool(done_flag or won or state["steps"] >= self.max_steps)
        state["success"] = bool(won or state.get("success", False))

        trace_item = {
            "step": state["steps"],
            "raw_action": str(raw_action or canonical_action),
            "executed_action": canonical_action,
            "observation": obs_text,
            "reward": str(reward),
            "done": bool(done_flag),
            "won": bool(won),
            "won_raw": str(won_raw),
            "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
        }
        if meta:
            trace_item["meta"] = dict(meta)
        state["trace"].append(trace_item)

        return {
            "observation": state["last_observation"],
            "done": bool(state["done"]),
            "won": bool(state["success"]),
            "reward": str(reward),
            "executed_action": canonical_action,
            "raw_observation": obs_text,
        }

    def finish_episode(self):
        if not isinstance(self._episode_state, dict):
            return {
                "score": 0.0,
                "success": False,
                "steps_executed": 0,
                "executed_actions": [],
                "parsed_actions": [],
                "last_observation": "",
                "eval_trace": [],
            }
        state = self._episode_state
        self._last_eval_info = {
            "actions": list(state.get("executed_actions", [])),
            "parsed_actions": list(state.get("parsed_actions", [])),
            "steps_executed": int(state.get("steps", 0)),
            "last_observation": str(state.get("last_observation", "")),
            "success": bool(state.get("success", False)),
            "gamefile": str(state.get("gamefile", "")),
            "initial_observation": str(state.get("initial_observation", "")),
        }
        if self.debug_eval:
            self._last_eval_info["eval_trace"] = list(state.get("trace", []))
        result = {
            "score": 1.0 if state.get("success", False) else 0.0,
            "success": bool(state.get("success", False)),
            "steps_executed": int(state.get("steps", 0)),
            "executed_actions": list(state.get("executed_actions", [])),
            "parsed_actions": list(state.get("parsed_actions", [])),
            "last_observation": str(state.get("last_observation", "")),
            "eval_trace": list(state.get("trace", [])),
        }
        self._episode_state = None
        return result

    def evaluate_entry(self, output, entry):
        actions = self._extract_actions(output)
        self.start_episode(entry)
        for action in actions[: self.max_steps]:
            step_out = self.step_episode(action, raw_action=action)
            if bool(step_out.get("done", False)):
                break
        final = self.finish_episode()
        return float(final.get("score", 0.0))

    def build_memory_record(self, entry, output, feedback, score):
        info = dict(self._last_eval_info or {})
        normalized_question = _normalize_task_main(str(entry.get("goal", "")).strip())
        if not normalized_question:
            normalized_question = str(entry.get("goal", "")).strip()
        record = {
            "qid": entry.get("_qid", ""),
            "question": normalized_question,
            "goal": str(entry.get("goal", "")).strip(),
            "gamefile": str(entry.get("gamefile", "")).strip(),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
            "executed_actions": info.get("actions", []),
            "parsed_actions": info.get("parsed_actions", []),
            "steps_executed": info.get("steps_executed", 0),
            "last_observation": info.get("last_observation", ""),
        }
        if self.debug_eval:
            record["eval_debug"] = {
                "success": bool(info.get("success", False)),
                "eval_trace": info.get("eval_trace", []),
            }
        return record

    def get_prompt(self):
        raise NotImplementedError("ALFWorld uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("ALFWorld uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("ALFWorld uses evaluate_entry(output, entry).")
