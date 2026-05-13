import argparse
import json
import os
from pathlib import Path

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_METHOD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TASKS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMING,
    DEFAULT_TOP_K,
    DEFAULT_UPDATE_MEMORY,
)
from registry import build_method, load_tasks
from runner import Runner
from timer import Timer
from utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evo benchmark runner")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names",
    )
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--generation-model", type=str, default=DEFAULT_GENERATION_MODEL)
    parser.add_argument(
        "--curator-model",
        type=str,
        default=None,
        help="Optional curator model for DynamicCheatsheet_RetrievalSynthesis; defaults to generation model.",
    )
    parser.add_argument(
        "--dc-rs-generator-prompt-path",
        type=str,
        default=None,
        help="Optional generator prompt template path for DC-RS.",
    )
    parser.add_argument(
        "--dc-rs-curator-prompt-path",
        type=str,
        default=None,
        help="Optional curator prompt template path for DC-RS.",
    )
    parser.add_argument(
        "--dc-rs-max-exec-rounds",
        type=int,
        default=3,
        help="Max rounds of Python code execution inside DC-RS generator loop. 0 disables execution.",
    )
    parser.add_argument(
        "--dc-rs-exec-timeout-seconds",
        type=int,
        default=15,
        help="Per-round subprocess timeout for DC-RS Python execution.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--induce-steps",
        type=int,
        default=1,
        help="Online workflow induction interval for AWM-Online.",
    )
    parser.add_argument(
        "--workflow-init",
        type=str,
        default="",
        help="Initial workflow text for AWM-Online.",
    )
    parser.add_argument(
        "--awminstruction-path",
        type=str,
        default=None,
        help="Optional induction instruction prompt path for AWM-Online.",
    )
    parser.add_argument(
        "--awm-oneshot-path",
        type=str,
        default=None,
        help="Optional induction one-shot prompt path for AWM-Online.",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=3,
        help="Maximum attempts per sample for ExpeL-Online-MT (multi-try only).",
    )
    parser.add_argument(
        "--batch-update-size",
        type=int,
        default=8,
        help="Batch size for ExpeL-Online-MT insight updates.",
    )
    parser.add_argument(
        "--insights-init",
        type=str,
        default="(empty)",
        help="Initial insight text for ExpeL-Online-MT.",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=None,
        help="Optional reflection model for ExpeL-Online-MT only; defaults to generation model.",
    )
    parser.add_argument(
        "--expel-max-steps",
        type=int,
        default=20,
        help=(
            "ExpeL-Online-MT/ST: per-sample ALFWorld step budget (default 20, "
            "matching the ExpeL paper's agent-level cap). Overrides "
            "--alfworld-max-steps for these methods only so multi-try retries "
            "stay within a comparable budget."
        ),
    )
    parser.add_argument(
        "--max-num-rules",
        type=int,
        default=20,
        help="Maximum number of maintained insight rules for ExpeL-Online-ST/MT.",
    )
    parser.add_argument(
        "--successful-topk",
        type=int,
        default=1,
        help="Number of successful trajectories retrieved by AutoGen-GMemory.",
    )
    parser.add_argument(
        "--failed-topk",
        type=int,
        default=1,
        help="Number of failed trajectories retrieved by AutoGen-GMemory.",
    )
    parser.add_argument(
        "--insights-topk",
        type=int,
        default=3,
        help="Number of insights retrieved by AutoGen-GMemory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Similarity threshold for AutoGen-GMemory trajectory retrieval.",
    )
    parser.add_argument(
        "--use-projector",
        action="store_true",
        default=False,
        help="Enable role-specific insight projection in AutoGen-GMemory.",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=1,
        help="Task graph hop distance for AutoGen-GMemory related-task retrieval.",
    )
    parser.add_argument(
        "--gmemory-persist-dir",
        type=str,
        default=None,
        help="Optional persistence directory for AutoGen-GMemory state.",
    )
    parser.add_argument(
        "--gmemory-reset",
        action="store_true",
        help=(
            "Wipe --gmemory-persist-dir at startup so the run begins with a "
            "clean graph/insights/records state (no carry-over from prior runs)."
        ),
    )
    parser.add_argument(
        "--gmemory-start-insights-threshold",
        type=int,
        default=5,
        help="AutoGen-GMemory minimum memory size before insight finetune starts.",
    )
    parser.add_argument(
        "--gmemory-rounds-per-insights",
        type=int,
        default=5,
        help="AutoGen-GMemory interval (in records) to trigger insight finetune.",
    )
    parser.add_argument(
        "--gmemory-insights-point-num",
        type=int,
        default=5,
        help="AutoGen-GMemory sampled points per insight finetune round.",
    )
    parser.add_argument(
        "--gmemory-max-num-rules",
        type=int,
        default=20,
        help="AutoGen-GMemory maximum maintained insight rules.",
    )
    parser.add_argument(
        "--gmemory-fidelity",
        type=str,
        choices=["high", "legacy"],
        default="high",
        help="AutoGen-GMemory execution fidelity mode (high: step-wise official-like; legacy: prior flow).",
    )
    parser.add_argument(
        "--gmemory-snapshot-every",
        type=int,
        default=100,
        help="AutoGen-GMemory snapshot interval in samples; final snapshot is always saved at task end.",
    )
    parser.add_argument(
        "--alfworld-fewshot-num",
        type=int,
        default=1,
        help="Number of ALFWorld few-shot examples per task type (default 1).",
    )
    parser.add_argument(
        "--alfworld-debug-eval",
        action="store_true",
        default=False,
        help="Dump per-step ALFWorld evaluator traces into memory outputs.",
    )
    parser.add_argument(
        "--alfworld-max-steps",
        type=int,
        default=30,
        help="Maximum number of evaluator-executed actions per ALFWorld sample.",
    )
    parser.add_argument(
        "--alfworld-split",
        type=str,
        default="ood",
        choices=["ood", "id"],
        help="ALFWorld eval split: 'ood' = eval_out_of_distribution (valid_unseen, default), "
             "'id' = eval_in_distribution (valid_seen).",
    )
    parser.add_argument(
        "--humaneval-timeout",
        type=int,
        default=10,
        help=(
            "HumanEval task: wall-clock timeout in seconds per problem "
            "(subprocess runs `check(entry_point)` once; default 10 s)."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["auto", "vllm", "transformers"],
        default=None,
        help=(
            "Optional local backend override for utils.llm_client "
            "(sets LLM_BACKEND for this run)."
        ),
    )
    parser.add_argument(
        "--openrouter-reasoning",
        type=str,
        choices=["on", "off", "low", "medium", "high"],
        default=None,
        help=(
            "For OpenRouter reasoning models: 'off' forces reasoning disabled "
            "via extra_body={'reasoning': {'effort': 'none'}}; 'on' forces it "
            "enabled; 'low'|'medium'|'high' pin the effort level via "
            "extra_body={'reasoning': {'effort': <value>}} (required for "
            "reasoning-only models like minimax-m2.7). Default: provider "
            "default. Sets OPENROUTER_REASONING for this run."
        ),
    )
    parser.add_argument(
        "--update-memory",
        action="store_true",
        default=DEFAULT_UPDATE_MEMORY,
    )
    parser.add_argument(
        "--no-update-memory",
        dest="update_memory",
        action="store_false",
    )
    parser.add_argument("--timing", action="store_true", default=DEFAULT_TIMING)
    parser.add_argument(
        "--max-parallel-samples",
        type=int,
        default=1,
        help=(
            "Intra-task per-sample concurrency for opt-in stateless methods "
            "(Baseline). 1 = sequential (default). When >1 and the method "
            "exposes supports_parallel_samples=True and the task is not "
            "interactive, samples are dispatched through a thread pool. "
            "Useful for OpenRouter-only runs where the bottleneck is API "
            "round-trip latency."
        ),
    )
    parser.add_argument(
        "--run-first-k",
        type=int,
        default=None,
        help="Run only the first K samples for each task.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing output files. Loads memory records from the "
            "existing JSONL, skips already-completed entries, and continues "
            "appending new results. Requires deterministic task ordering."
        ),
    )
    parser.add_argument(
        "--memory-seed-file",
        type=str,
        default=None,
        help=(
            "Seed method state from an existing memory JSONL (e.g., a training "
            "run truncated to K_i entries) for holdout evaluation. Unlike "
            "--resume, the seed file's qids are NOT added to the 'already done' "
            "set, so the task iterator runs all entries fresh. Intended to be "
            "combined with --no-update-memory for read-only holdout runs."
        ),
    )
    parser.add_argument(
        "--task-data-override",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "Override a task's data source. Format: TASK=PATH (repeatable). "
            "Dispatched to that task's constructor as data_dir= if PATH is a "
            "directory, else data_file=. Used by the holdout harness to point "
            "a task at its holdout split without affecting other tasks."
        ),
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    if args.llm_backend is not None:
        os.environ["LLM_BACKEND"] = args.llm_backend
    if args.openrouter_reasoning is not None:
        os.environ["OPENROUTER_REASONING"] = args.openrouter_reasoning
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]

    method_kwargs = {
        "top_k": args.top_k,
        "embedding_model_name": args.embedding_model,
        "generation_model_name": args.generation_model,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    if (
        args.method in {"DynamicCheatsheet_RetrievalSynthesis", "DC-RS"}
    ):
        if args.curator_model is not None:
            method_kwargs["curator_model_name"] = args.curator_model
        if args.dc_rs_generator_prompt_path is not None:
            method_kwargs["generator_prompt_path"] = args.dc_rs_generator_prompt_path
        if args.dc_rs_curator_prompt_path is not None:
            method_kwargs["curator_prompt_path"] = args.dc_rs_curator_prompt_path
        method_kwargs["max_exec_rounds"] = args.dc_rs_max_exec_rounds
        method_kwargs["exec_timeout_seconds"] = args.dc_rs_exec_timeout_seconds
    if args.method in {"AWM-Online", "AWMOnline"}:
        method_kwargs["induce_steps"] = args.induce_steps
        method_kwargs["workflow_init"] = args.workflow_init
        method_kwargs["awminstruction_path"] = args.awminstruction_path
        method_kwargs["awm_oneshot_path"] = args.awm_oneshot_path
        if args.curator_model is not None:
            method_kwargs["curator_model_name"] = args.curator_model
    if args.method in {"ExpeL-Online-MT", "ExpeL-Online", "ExpeLOnline", "ExpeL-Online-MT-tries1"}:
        method_kwargs["max_tries"] = args.max_tries
        method_kwargs["batch_update_size"] = args.batch_update_size
        method_kwargs["insights_init"] = args.insights_init
        method_kwargs["max_num_rules"] = args.max_num_rules
        if args.reflection_model is not None:
            method_kwargs["reflection_model_name"] = args.reflection_model
        if args.curator_model is not None:
            method_kwargs["curator_model_name"] = args.curator_model
    if args.method in {"ExpeL-Online-ST", "ExpeLOnlineST"}:
        method_kwargs["batch_update_size"] = args.batch_update_size
        method_kwargs["insights_init"] = args.insights_init
        method_kwargs["max_num_rules"] = args.max_num_rules
        if args.curator_model is not None:
            method_kwargs["curator_model_name"] = args.curator_model
    if args.method in {"AutoGen-GMemory", "AutoGenGMemory", "g-memory-autogen"}:
        method_kwargs["successful_topk"] = args.successful_topk
        method_kwargs["failed_topk"] = args.failed_topk
        method_kwargs["insights_topk"] = args.insights_topk
        method_kwargs["threshold"] = args.threshold
        method_kwargs["use_projector"] = args.use_projector
        method_kwargs["hop"] = args.hop
        method_kwargs["persist_dir"] = args.gmemory_persist_dir
        method_kwargs["reset_persist"] = args.gmemory_reset
        method_kwargs["start_insights_threshold"] = args.gmemory_start_insights_threshold
        method_kwargs["rounds_per_insights"] = args.gmemory_rounds_per_insights
        method_kwargs["insights_point_num"] = args.gmemory_insights_point_num
        method_kwargs["max_num_rules"] = args.gmemory_max_num_rules
        method_kwargs["fidelity"] = args.gmemory_fidelity
        method_kwargs["snapshot_every"] = args.gmemory_snapshot_every
        if args.curator_model is not None:
            method_kwargs["curator_model_name"] = args.curator_model

    method = build_method(args.method, **method_kwargs)

    per_task_kwargs = {}
    for spec in (args.task_data_override or []):
        if "=" not in spec:
            raise ValueError(
                f"--task-data-override expects TASK=PATH, got {spec!r}"
            )
        name, path_str = spec.split("=", 1)
        name = name.strip()
        path_str = path_str.strip()
        if not name or not path_str:
            raise ValueError(
                f"--task-data-override expects non-empty TASK and PATH, got {spec!r}"
            )
        p = Path(path_str)
        key = "data_dir" if p.is_dir() or (not p.exists() and p.suffix == "") else "data_file"
        per_task_kwargs.setdefault(name, {})[key] = str(p)

    tasks = load_tasks(
        task_names,
        per_task_kwargs=per_task_kwargs,
        prompt_file=args.prompt_file,
    )
    for task in tasks:
        if hasattr(task, "alfworld_fewshot_num"):
            task.alfworld_fewshot_num = max(0, int(args.alfworld_fewshot_num))
        if hasattr(task, "debug_eval"):
            task.debug_eval = bool(args.alfworld_debug_eval)
        if hasattr(task, "max_steps"):
            task.max_steps = max(1, int(args.alfworld_max_steps))
        if getattr(task, "name", "") == "ALFWorld" and hasattr(task, "set_split"):
            task.set_split(args.alfworld_split)
        # ExpeL official caps ALFWorld at 20 per attempt (configs/benchmark/alfworld.yaml:max_steps=20).
        # Override the shared --alfworld-max-steps for ExpeL methods only, so multi-try
        # runs stay within a comparable budget as the paper.
        if (
            getattr(task, "name", "") == "ALFWorld"
            and args.method in {"ExpeL-Online-MT", "ExpeL-Online", "ExpeLOnline", "ExpeL-Online-MT-tries1", "ExpeL-Online-ST"}
            and hasattr(task, "max_steps")
        ):
            task.max_steps = max(1, int(args.expel_max_steps))
        if hasattr(task, "humaneval_timeout"):
            task.humaneval_timeout = max(2, int(args.humaneval_timeout))
    timer = Timer(enabled=args.timing)

    runner = Runner(
        method, tasks, timer, output_dir=args.output_dir,
        max_parallel_samples=args.max_parallel_samples,
    )
    results = runner.run(
        update_memory=args.update_memory,
        run_first_k=args.run_first_k,
        resume=args.resume,
        memory_seed_file=args.memory_seed_file,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always write per-task stats. Timing fields are only included when enabled.
    stage_avg = timer.summary() if args.timing else {}
    stage_avg_by_task = {}
    per_sample_avg_seconds = {}
    sample_total_avg_seconds = {}
    for key, avg_sec in stage_avg.items():
        parts = key.split("/")
        if len(parts) != 3:
            continue
        _, task_name, stage_name = parts
        stage_avg_by_task.setdefault(task_name, {})[stage_name] = avg_sec
        if stage_name in {"retrieve", "generate", "update"}:
            per_sample_avg_seconds[task_name] = per_sample_avg_seconds.get(task_name, 0.0) + avg_sec
        if stage_name == "sample_total":
            sample_total_avg_seconds[task_name] = float(avg_sec)

    saved_stats_paths = []
    for task_result in results:
        task_name = task_result["task"]
        stats = {
            "task": task_name,
            "method": task_result["method"],
            "total": task_result["total"],
            "success": task_result["success"],
            "accuracy": task_result["accuracy"],
        }
        if task_result.get("avg_steps_executed") is not None:
            stats["avg_steps_executed"] = float(task_result.get("avg_steps_executed", 0.0))
            stats["total_steps_executed"] = int(task_result.get("total_steps_executed", 0))
            stats["counted_step_samples"] = int(task_result.get("counted_step_samples", 0))
        if task_result.get("hallucination_rate") is not None:
            stats["hallucination_rate"] = float(task_result.get("hallucination_rate", 0.0))
            stats["total_hallucinated"] = int(task_result.get("total_hallucinated", 0))
            stats["counted_hallucination_samples"] = int(
                task_result.get("counted_hallucination_samples", 0)
            )
        if args.timing:
            stats["stage_avg_seconds"] = stage_avg_by_task.get(task_name, {})
            stats["per_sample_avg_seconds"] = per_sample_avg_seconds.get(task_name, 0.0)
            stats["sample_total_avg_seconds"] = sample_total_avg_seconds.get(task_name, 0.0)

        stats_path = output_dir / f"{task_name}_{args.method}_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        saved_stats_paths.append(str(stats_path))

    print(json.dumps({"saved_stats_files": saved_stats_paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
