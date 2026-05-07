# SeqMem-Eval

An evaluation harness for **online memory / experience-evolution methods** on
sequential reasoning, scientific QA, code, API selection, and embodied tasks.
A method is a triple `retrieve_memory → generate → update_memory` run
sample-by-sample, so memory evolves online within a task.

This repository contains code only. Datasets are not bundled — please obtain
them from their original public sources (see [Data](#data)).

## Install

```bash
conda env create -f environment.yml
conda activate seqmem-eval
pip install -r requirements.txt
```

Optional: ALFWorld requires the `alfworld` package and its asset bundle (see
its upstream repository for `alfworld-download`).

## Configure

Copy `.env.template` to `.env` and fill in the API keys you need:

```bash
cp .env.template .env
```

Variables read at runtime:

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | OpenAI generation backend (`--generation-model openai/<model>`) |
| `OPENROUTER_API_KEY` | OpenRouter generation backend |
| `OPENROUTER_PROVIDER` | Optional provider pin for OpenRouter |
| `LLM_BACKEND` | `transformers` (default), `vllm`, or `auto` for local HF models |
| `MODEL_CACHE_DIR` | HuggingFace download cache (default `./model_cache`) |

## Data

The repository ships **without** datasets. Download them yourself and place
them under `data/` using the layout below.

| Task name | Source |
|---|---|
| `MATH500` | HuggingFace: `HuggingFaceH4/MATH-500` |
| `AIME2024`, `AIME2025` | HuggingFace: `Maxwell-Jia/AIME_2024`, `opencompass/AIME2025` |
| `MMLU-Pro-Math`, `MMLU-Pro-Physics`, `MMLU-Pro-Engineering` | HuggingFace: `TIGER-Lab/MMLU-Pro` (filtered by category) |
| `HumanEval` | HuggingFace: `openai_humaneval` |
| `APIBench-HF`, `APIBench-TF`, `APIBench-TH` | Gorilla APIBench official release |
| `ALFWorld` | `alfworld` package + bundled PDDL/JSON assets |

Each task file (`tasks/task_*.py`) documents the directory it expects. As a
rough guide, place each dataset under `data/<task-name>/` with the JSONL or
JSON files the loader looks up.

## Run

```bash
# Quick smoke test (5 samples)
python main.py --method ExpRAG --tasks AIME2024 --run-first-k 5 --timing

# Multi-task sweep
python main.py --method ExpRAG --tasks MATH500,HumanEval

# OpenAI generation backend
python main.py --method DC-RS --tasks AIME2024 --generation-model openai/gpt-4o-mini

# Step-wise embodied task (high-fidelity GMemory)
python main.py --method AutoGen-GMemory --tasks ALFWorld --gmemory-fidelity high
```

Useful flags:

- `--run-first-k N` — process only the first N samples (smoke test).
- `--output-dir DIR` — where to write per-task results (default `outputs/`).
- `--timing` — record per-stage timings into the stats file.
- `--llm-backend {auto,vllm,transformers}` — local-HF backend selector.
- `--generation-model NAME` — HuggingFace repo id, `openai/<model>`, or
  `chatgpt`.

## Outputs

Per `(task, method)` pair the runner writes into `--output-dir`:

- `{task}_{method}_memory.jsonl` — full per-sample memory records.
- `{task}_{method}_memory_readable.json` — pretty-printed view.
- `{task}_{method}_stats.json` — accuracy plus optional timing.

Outputs are persisted after every sample, so a partial run still leaves usable
data on disk.

## Methods

| Name (`--method`) | Description |
|---|---|
| `Baseline` (`NoMemory`) | No memory — direct generation. |
| `ExpRecent` | Append the most-recent N `(query, output)` pairs. |
| `ExpRAG` | Retrieval-augmented memory of prior trajectories. |
| `AWM-Online` (`AWMOnline`) | Workflow induction every Nth success. |
| `DC-RS` (`DynamicCheatsheet_RetrievalSynthesis`) | Cheatsheet retrieval + synthesis. |
| `ExpeL-Online-MT` / `ExpeL-Online-ST` | Experience pool + insight-rule editing (multi-try / single-try). |
| `AutoGen-GMemory` (`AutoGenGMemory`, `g-memory-autogen`) | Graph-memory with high-fidelity step-wise rollout. |

## Tasks

`MATH500`, `AIME2024`, `AIME2025`, `MMLU-Pro-Math`, `MMLU-Pro-Physics`,
`MMLU-Pro-Engineering`, `HumanEval`, `APIBench-HF`, `APIBench-TF`,
`APIBench-TH`, `ALFWorld`.

## Notes

- For multi-model pipelines (AWM / DC-RS / ExpeL / GMemory) prefer
  `--llm-backend transformers`. Running multiple local models through vLLM in a
  single process is unstable in our setup.
- Method registries live in `methods/__init__.py`; task registries in
  `tasks/__init__.py`. To add a new method, also wire its CLI args in
  `main.py::main`.
