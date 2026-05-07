"""Single source of truth for holdout evaluation settings.

Keys are (model, task) or (method, task) tuples. When adding a new
(model, task, method) to the holdout harness, extend the dicts below and
nothing else — scripts/holdout/run.py reads from this module exclusively.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Checkpoint steps K_1..K_9 per (model, task). t=0 is NOT in this list; it is
# a separate no-memory baseline run handled by run_t0_baseline.py (TBD).
# Source: plan_holdout_testset.md, the per-model checkpoint tables.
# ---------------------------------------------------------------------------
CHECKPOINT_STEPS: dict[tuple[str, str], list[int]] = {
    ("qwen3-8b", "MATH"): [600, 1100, 1700, 2200, 2800, 3300, 3885, 4485, 5000],
    # MiniMax-M2.7 × APIBench-HF: eff_N=911, snap_every=100.
    # K_9 maps to final_000911_* (no step_000911 exists).
    ("minimax-m2.7", "APIBench-HF"): [100, 200, 300, 400, 500, 600, 700, 800, 911],
    # MiniMax-M2.7 × ALFWorld: eff_N=140 (training ran to sample 140).
    # Default is the evenly-spaced K_i for A/C methods:
    #   K_i = round(i × 140 / 9) = [16, 31, 47, 62, 78, 93, 109, 124, 140]
    # GMemory snapshots (snap_every=10) don't land on these exact steps, so
    # `CHECKPOINT_STEPS_METHOD_OVERRIDE` pins a GMemory-aligned list.
    ("minimax-m2.7", "ALFWorld"): [16, 31, 47, 62, 78, 93, 109, 124, 140],
    # MiniMax-M2.7 × HumanEval: holdout = HumanEval/132..163 (last 20%).
    # Training ran to sample 164 (overshooting holdout boundary), so to avoid
    # leakage we treat eff_N=130 and only use snapshots up to step_000130.
    # K_i = round(i × 130 / 9) → mapped to nearest snap_every=10 step:
    #   [14→10, 29→30, 43→40, 58→60, 72→70, 87→90, 101→100, 116→120, 130→130]
    ("minimax-m2.7", "HumanEval"): [10, 30, 40, 60, 70, 90, 100, 120, 130],
    # Qwen3-8B × HumanEval: same setup as MiniMax (holdout = HumanEval/132..163,
    # eff_N=130, snap_every=10). Snapshots step_000010 .. step_000160 exist.
    ("qwen3-8b", "HumanEval"): [10, 30, 40, 60, 70, 90, 100, 120, 130],
    # Qwen3-8B × ALFWorld: training ran to sample 140, holdout has 24 samples.
    # Default K_i = round(i * 140 / 9); GMemory uses an override aligned to
    # snap_every=10. snapshots: step_000010..130 + final_000140 (14 total).
    ("qwen3-8b", "ALFWorld"): [16, 31, 47, 62, 78, 93, 109, 124, 140],
    # Qwen3-8B × MATH500: training over all 500 MATH500 samples, snap_every=50.
    # Holdout reuses MATH train holdout (math_holdout.jsonl, 280 samples) which
    # is disjoint from MATH500 ⊆ MATH/test. Default K_i = round(i*500/9).
    ("qwen3-8b", "MATH500"): [56, 111, 167, 222, 278, 333, 389, 444, 500],
    # Qwen3-8B × MMLU-Pro-Engineering: training ran to sample 872 (eff_N=873
    # in plan, training off-by-1). snap_every=100. Default K_i = round(i*873/9).
    ("qwen3-8b", "MMLU-Pro-Engineering"): [97, 194, 291, 388, 485, 582, 679, 776, 873],
    # Qwen3-8B × APIBench-HF: eff_N=911, snap_every=100. K_i = round(i*911/9).
    ("qwen3-8b", "APIBench-HF"): [101, 202, 304, 405, 506, 607, 709, 810, 911],
    # Qwen3-8B × MMLU-Pro-Physics: N_ref=1170, training final at sample 1169.
    # snap_every=100. K_i = round(i*1170/9).
    ("qwen3-8b", "MMLU-Pro-Physics"): [130, 260, 390, 520, 650, 780, 910, 1040, 1170],
    # MiniMax × MATH500/MMLU: same task sizes as Qwen3 side.
    ("minimax-m2.7", "MATH500"): [56, 111, 167, 222, 278, 333, 389, 444, 500],
    ("minimax-m2.7", "MMLU-Pro-Physics"): [130, 260, 390, 520, 650, 780, 910, 1040, 1170],
    ("minimax-m2.7", "MMLU-Pro-Engineering"): [97, 194, 291, 388, 485, 582, 679, 776, 873],
    # ----- OOD holdout: MATH500 memory replayed on AIME / Physics -----------
    # All three eval sets reuse the MATH500 K_i so checkpoints align with the
    # MATH500 memory file's per-step truncation.
    ("qwen3-8b", "AIME2024"): [56, 111, 167, 222, 278, 333, 389, 444, 500],
    ("qwen3-8b", "AIME2025"): [56, 111, 167, 222, 278, 333, 389, 444, 500],
    ("qwen3-8b", "MMLU-Pro-Physics-OOD-MATH500"): [56, 111, 167, 222, 278, 333, 389, 444, 500],
    # Filled in as more (model, task) runs complete.
}


# Per-method override keyed by (model, task, method). Falls back to
# CHECKPOINT_STEPS[(model, task)] when missing.
CHECKPOINT_STEPS_METHOD_OVERRIDE: dict[tuple[str, str, str], list[int]] = {
    # MiniMax × ALFWorld × GMemory uses the actual snapshot step list.
    # step_000140 is saved under final_000140_* — run.py falls back from
    # step_{N:06d}_* to final_{N:06d}_* automatically.
    ("minimax-m2.7", "ALFWorld", "AutoGen-GMemory"): [20, 30, 50, 60, 80, 90, 110, 120, 140],
    # Qwen3-8B × ALFWorld × GMemory: same N=140, snap_every=10 layout as MiniMax.
    # K_i = round(i*140/9) → nearest snap_every=10 step → [20,30,50,60,80,90,110,120,140].
    # K_9=140 maps to final_000140_* (no step_000140 exists).
    ("qwen3-8b", "ALFWorld", "AutoGen-GMemory"): [20, 30, 50, 60, 80, 90, 110, 120, 140],
    # Qwen3-8B × MATH500 × GMemory: snap_every=50 → 11 snapshots (step_000050
    # ..500 + final_000500). K_i=round(i*500/9) mapped to nearest snap-aligned
    # step → [50,100,150,200,300,350,400,450,500]. K_5=278 picks 300 over 250
    # (|278-300|=22 < |278-250|=28).
    ("qwen3-8b", "MATH500", "AutoGen-GMemory"): [50, 100, 150, 200, 300, 350, 400, 450, 500],
    # Qwen3-8B × MMLU-Pro-Engineering × GMemory: 9 snapshots (step_100..800 +
    # final_000872). K_i = round(i*873/9), each mapped to nearest snap-aligned
    # step or final_000872 for K_9. K_9=873 → final_000872 (diff=1).
    # run.py falls back from step_{N:06d}_* to final_{N:06d}_* automatically,
    # so passing 872 here resolves to final_000872_*.
    ("qwen3-8b", "MMLU-Pro-Engineering", "AutoGen-GMemory"): [100, 200, 300, 400, 500, 600, 700, 800, 872],
    # Qwen3-8B × APIBench-HF × GMemory: snap_every=100, snapshots step_100..900
    # + final_000911 (10 snaps). K_i = round(i*911/9), each mapped to nearest
    # snap-aligned step → integers 100..800; K_9=911 → final_000911 (diff=0).
    ("qwen3-8b", "APIBench-HF", "AutoGen-GMemory"): [100, 200, 300, 400, 500, 600, 700, 800, 911],
    # Qwen3-8B × MMLU-Pro-Physics × GMemory: snap_every=100, snapshots
    # step_100..1100 + final_001169 (12 snaps). K_i = round(i*1170/9), each
    # mapped to nearest snap-aligned step (tie → prefer later step):
    #   130→100 (Δ30), 260→300 (Δ40), 390→400 (Δ10), 520→500 (Δ20),
    #   650→700 (Δ50, tie with 600), 780→800 (Δ20), 910→900 (Δ10),
    #   1040→1000 (Δ40), 1170→final_001169 (Δ1).
    ("qwen3-8b", "MMLU-Pro-Physics", "AutoGen-GMemory"): [100, 300, 400, 500, 700, 800, 900, 1000, 1169],
    # MiniMax-M2.7 × MATH500 × GMemory: snap_every=50, snapshots step_050..500 + final_500.
    ("minimax-m2.7", "MATH500", "AutoGen-GMemory"): [50, 100, 150, 200, 300, 350, 400, 450, 500],
    # MiniMax-M2.7 × MMLU-Pro-Physics × GMemory: snap_every=100, step_100..1100 + final_001170.
    # K_i = round(i*1170/9) → nearest: [100,300,400,500,700,800,900,1000,1170].
    ("minimax-m2.7", "MMLU-Pro-Physics", "AutoGen-GMemory"): [100, 300, 400, 500, 700, 800, 900, 1000, 1170],
    # MiniMax-M2.7 × MMLU-Pro-Engineering × GMemory: snap_every=100, step_100..800 + final_000873.
    # K_i = round(i*873/9) → nearest snap-aligned: [100,200,300,400,500,600,700,800,873].
    ("minimax-m2.7", "MMLU-Pro-Engineering", "AutoGen-GMemory"): [100, 200, 300, 400, 500, 600, 700, 800, 873],
    # OOD holdout (MATH500 GMemory snapshots replayed on AIME / Physics).
    # Snapshots step_000050..500 + final_000500. Same K_i scheme as
    # in-distribution MATH500 GMemory.
    ("qwen3-8b", "AIME2024", "AutoGen-GMemory"): [50, 100, 150, 200, 300, 350, 400, 450, 500],
    ("qwen3-8b", "AIME2025", "AutoGen-GMemory"): [50, 100, 150, 200, 300, 350, 400, 450, 500],
    ("qwen3-8b", "MMLU-Pro-Physics-OOD-MATH500", "AutoGen-GMemory"): [50, 100, 150, 200, 300, 350, 400, 450, 500],
}


# ---------------------------------------------------------------------------
# OOD holdout: per (model, eval_task), the source training task whose memory
# files / GMemory snapshots should be loaded. Used by scripts/holdout/run.py
# to compose the source memory filename and to look up TRAINING_OUTPUT_DIR /
# GMEMORY_SNAPSHOT_ROOT under the *training* task key when the eval task is
# different. The CLI flag --training-task overrides this.
# ---------------------------------------------------------------------------
TRAINING_TASK_NAME: dict[tuple[str, str], str] = {
    ("qwen3-8b", "AIME2024"): "MATH500",
    ("qwen3-8b", "AIME2025"): "MATH500",
    ("qwen3-8b", "MMLU-Pro-Physics-OOD-MATH500"): "MATH500",
}


# ---------------------------------------------------------------------------
# Training output directory — where {task}_{method}_memory.jsonl lives for
# Class A/C methods. For Qwen3-8B × MATH the archived copies live under
# memory/qwen3-8b/math/ (flat, one per method).
# ---------------------------------------------------------------------------
TRAINING_OUTPUT_DIR: dict[tuple[str, str], Path] = {
    ("qwen3-8b", "MATH"): ROOT / "memory" / "qwen3-8b" / "math",
    ("qwen3-8b", "HumanEval"): ROOT / "memory" / "qwen3-8b" / "humaneval",
    ("qwen3-8b", "ALFWorld"): ROOT / "memory" / "qwen3-8b" / "alfworld",
    ("qwen3-8b", "MATH500"): ROOT / "memory" / "qwen3-8b" / "math500",
    ("qwen3-8b", "MMLU-Pro-Engineering"): ROOT / "memory" / "qwen3-8b" / "MMLUpro-eng",
    ("qwen3-8b", "APIBench-HF"): ROOT / "memory" / "qwen3-8b" / "apibench-hf",
    ("qwen3-8b", "MMLU-Pro-Physics"): ROOT / "memory" / "qwen3-8b" / "MMLUpro-phys",
    ("minimax-m2.7", "APIBench-HF"): ROOT / "memory" / "minimax-m2.7" / "apibench-hf",
    ("minimax-m2.7", "ALFWorld"): ROOT / "memory" / "minimax-m2.7" / "alfworld",
    ("minimax-m2.7", "HumanEval"): ROOT / "memory" / "minimax-m2.7" / "humaneval",
    ("minimax-m2.7", "MATH500"): ROOT / "memory" / "minimax-m2.7" / "math500",
    ("minimax-m2.7", "MMLU-Pro-Physics"): ROOT / "memory" / "minimax-m2.7" / "MMLUpro-phys",
    ("minimax-m2.7", "MMLU-Pro-Engineering"): ROOT / "memory" / "minimax-m2.7" / "MMLUpro-eng",
}


# Per-method override of TRAINING_OUTPUT_DIR. Keyed by (model, task, method).
# Used when a method's training memory.jsonl lives in a non-default subdir.
# Falls back to TRAINING_OUTPUT_DIR[(model, task)] when missing.
#
# Concrete case: AllHistory is a proxy for "ExpRecent --top-k 10". Training
# was done in a separate run that wrote to `<task>/allhist_proxy/`. The file
# name ({task}_ExpRecent_memory.jsonl) collides with the regular ExpRecent
# (k=3) file at the parent dir, so we must point Class-A loading at the
# proxy subdir for AllHistory holdout.
TRAINING_OUTPUT_DIR_METHOD_OVERRIDE: dict[tuple[str, str, str], Path] = {
    ("qwen3-8b", "APIBench-HF", "AllHistory"):          ROOT / "memory" / "qwen3-8b" / "apibench-hf" / "allhist_proxy",
    ("qwen3-8b", "HumanEval", "AllHistory"):            ROOT / "memory" / "qwen3-8b" / "humaneval" / "allhist_proxy",
    ("qwen3-8b", "MATH500", "AllHistory"):              ROOT / "memory" / "qwen3-8b" / "math500" / "allhist_proxy",
    ("qwen3-8b", "MMLU-Pro-Physics", "AllHistory"):     ROOT / "memory" / "qwen3-8b" / "MMLUpro-phys" / "allhist_proxy",
    ("qwen3-8b", "MMLU-Pro-Engineering", "AllHistory"): ROOT / "memory" / "qwen3-8b" / "MMLUpro-eng" / "allhist_proxy",
    ("minimax-m2.7", "APIBench-HF", "AllHistory"):          ROOT / "memory" / "minimax-m2.7" / "apibench-hf" / "allhist_proxy",
    ("minimax-m2.7", "HumanEval", "AllHistory"):            ROOT / "memory" / "minimax-m2.7" / "humaneval" / "allhist_proxy",
    ("minimax-m2.7", "MATH500", "AllHistory"):              ROOT / "memory" / "minimax-m2.7" / "math500" / "allhist_proxy",
    ("minimax-m2.7", "MMLU-Pro-Physics", "AllHistory"):     ROOT / "memory" / "minimax-m2.7" / "MMLUpro-phys" / "allhist_proxy",
    ("minimax-m2.7", "MMLU-Pro-Engineering", "AllHistory"): ROOT / "memory" / "minimax-m2.7" / "MMLUpro-eng" / "allhist_proxy",
}


# ---------------------------------------------------------------------------
# GMemory snapshot root for Class B (AutoGen-GMemory). Points at the flat
# layout: {root}/step_{NNNNNN}_{ts}/ with chroma/, insights.json, task_graph.pkl
# (plus final_{N:06d}_* for some runs).
# ---------------------------------------------------------------------------
GMEMORY_SNAPSHOT_ROOT: dict[tuple[str, str], Path] = {
    ("qwen3-8b", "MATH"): ROOT / "memory" / "qwen3-8b" / "math" / ".gmemory",
    ("qwen3-8b", "HumanEval"): ROOT / "memory" / "qwen3-8b" / "humaneval" / ".gmemory",
    ("qwen3-8b", "ALFWorld"): ROOT / "memory" / "qwen3-8b" / "alfworld" / ".gmemory",
    ("qwen3-8b", "MATH500"): ROOT / "memory" / "qwen3-8b" / "math500" / ".gmemory",
    ("qwen3-8b", "MMLU-Pro-Engineering"): ROOT / "memory" / "qwen3-8b" / "MMLUpro-eng" / ".gmemory",
    ("qwen3-8b", "APIBench-HF"): ROOT / "memory" / "qwen3-8b" / "apibench-hf" / ".gmemory",
    ("qwen3-8b", "MMLU-Pro-Physics"): ROOT / "memory" / "qwen3-8b" / "MMLUpro-phys" / ".gmemory",
    ("minimax-m2.7", "APIBench-HF"): ROOT / "memory" / "minimax-m2.7" / "apibench-hf" / ".gmemory",
    ("minimax-m2.7", "ALFWorld"): ROOT / "memory" / "minimax-m2.7" / "alfworld" / ".gmemory",
    ("minimax-m2.7", "HumanEval"): ROOT / "memory" / "minimax-m2.7" / "humaneval" / ".gmemory",
    ("minimax-m2.7", "MATH500"): ROOT / "memory" / "minimax-m2.7" / "math500" / ".gmemory",
    ("minimax-m2.7", "MMLU-Pro-Physics"): ROOT / "memory" / "minimax-m2.7" / "MMLUpro-phys" / ".gmemory",
    ("minimax-m2.7", "MMLU-Pro-Engineering"): ROOT / "memory" / "minimax-m2.7" / "MMLUpro-eng" / ".gmemory",
}


# ---------------------------------------------------------------------------
# Task data override (passed through main.py's --task-data-override).
# Value is a path to the materialized holdout data dir/file that the task
# loader will read instead of the default training split.
# ---------------------------------------------------------------------------
HOLDOUT_DATA_OVERRIDE: dict[str, Path] = {
    "MATH": ROOT / "data" / "Holdout-data" / "MATH" / "materialized",
    # MATH500 reuses MATH train-holdout (math_holdout.jsonl, 280 samples).
    # MATH500 ⊆ MATH/test and math_holdout ⊆ MATH/train ⇒ disjoint, no leakage.
    "MATH500": ROOT / "data" / "Holdout-data" / "MATH" / "math_holdout.jsonl",
    "MMLU-Pro-Engineering": ROOT / "data" / "Holdout-data" / "MMLU-Pro" / "mmlu_pro_engineering_holdout.jsonl",
    "MMLU-Pro-Physics": ROOT / "data" / "Holdout-data" / "MMLU-Pro" / "mmlu_pro_physics_holdout.jsonl",
    "APIBench-HF": ROOT / "data" / "Holdout-data" / "APIBench" / "materialized",
    "ALFWorld": ROOT / "data" / "Holdout-data" / "ALFWorld" / "alfworld_holdout.json",
    "HumanEval": ROOT / "data" / "Holdout-data" / "HumanEval" / "humaneval_holdout.jsonl",
    # OOD holdout eval sets (memory comes from MATH500; see TRAINING_TASK_NAME).
    "AIME2024": ROOT / "data" / "AIME2024" / "aime2024.jsonl",
    "AIME2025": ROOT / "data" / "AIME2025" / "aime2025.jsonl",
    "MMLU-Pro-Physics-OOD-MATH500": ROOT / "data" / "Holdout-data" / "MMLU-Pro" / "mmlu_pro_physics_holdout.jsonl",
}


# ---------------------------------------------------------------------------
# Canonical training flags per (method, task). These are appended verbatim to
# the holdout main.py command so the retrieval / prompt-assembly code paths
# see the same configuration used at training time. Source: run_sweep.sh.
# Method name here must match the --method value used at both training and
# holdout; aliases (e.g., "AllHistory") map to the method actually used in
# the training sweep (ExpRecent@10).
# ---------------------------------------------------------------------------
TRAINING_FLAGS: dict[tuple[str, str], dict[str, str | int]] = {
    # HistoryRAG
    ("HistoryRAG", "MATH"): {"top-k": 3},
    # ExpRecent
    ("ExpRecent", "MATH"): {"top-k": 3},
    # AllHistory trained as ExpRecent --top-k 10 (see run_sweep.sh "AllHistory-proxy").
    # Loads <task>_ExpRecent_memory.jsonl (METHOD_TRAINING_FILENAME["AllHistory"]="ExpRecent")
    # with top-k=10 — equivalent to "ExpRecent k=10" in user-facing language.
    ("AllHistory", "MATH"): {"top-k": 10},
    ("AllHistory", "APIBench-HF"): {"top-k": 10},
    ("AllHistory", "HumanEval"): {"top-k": 10},
    ("AllHistory", "MATH500"): {"top-k": 10},
    ("AllHistory", "MMLU-Pro-Physics"): {"top-k": 10},
    ("AllHistory", "MMLU-Pro-Engineering"): {"top-k": 10},
    # AWM-Online
    ("AWM-Online", "MATH"): {"top-k": 3, "induce-steps": 1},
    ("AWM-Online", "HumanEval"): {"top-k": 3, "induce-steps": 1},
    ("AWM-Online", "APIBench-HF"): {"top-k": 3, "induce-steps": 1},
    ("AWM-Online", "MATH500"): {"top-k": 3, "induce-steps": 1},
    ("AWM-Online", "MMLU-Pro-Physics"): {"top-k": 3, "induce-steps": 1},
    ("AWM-Online", "MMLU-Pro-Engineering"): {"top-k": 3, "induce-steps": 1},
    ("AWM-Online", "ALFWorld"): {
        "top-k": 3, "induce-steps": 1,
        "alfworld-max-steps": 30, "alfworld-fewshot-num": 1,
    },
    # DC-RS
    ("DC-RS", "MATH"): {"top-k": 3},
    ("DC-RS", "HumanEval"): {"top-k": 3},
    ("DC-RS", "APIBench-HF"): {"top-k": 3},
    ("DC-RS", "MATH500"): {"top-k": 3},
    ("DC-RS", "MMLU-Pro-Physics"): {"top-k": 3},
    ("DC-RS", "MMLU-Pro-Engineering"): {"top-k": 3},
    # ExpeL-MT
    ("ExpeL-Online-MT", "MATH"): {
        "top-k": 3,
        "max-tries": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-MT", "HumanEval"): {
        "top-k": 3,
        "max-tries": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-MT", "MATH500"): {
        "top-k": 3,
        "max-tries": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-MT", "APIBench-HF"): {
        "top-k": 3,
        "max-tries": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-MT", "MMLU-Pro-Physics"): {
        "top-k": 3,
        "max-tries": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-MT", "MMLU-Pro-Engineering"): {
        "top-k": 3,
        "max-tries": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    # ExpeL-MT × ALFWorld — step-wise rollout (init-obs fix), needs alfworld step caps
    ("ExpeL-Online-MT", "ALFWorld"): {
        "top-k": 3,
        "max-tries": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
        "alfworld-max-steps": 30,
        "alfworld-fewshot-num": 1,
    },
    # ExpeL-ST
    ("ExpeL-Online-ST", "MATH"): {
        "top-k": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-ST", "HumanEval"): {
        "top-k": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-ST", "MATH500"): {
        "top-k": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-ST", "APIBench-HF"): {
        "top-k": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-ST", "MMLU-Pro-Physics"): {
        "top-k": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-ST", "MMLU-Pro-Engineering"): {
        "top-k": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
    },
    ("ExpeL-Online-ST", "ALFWorld"): {
        "top-k": 3,
        "batch-update-size": 8,
        "max-num-rules": 20,
        "alfworld-max-steps": 30,
        "alfworld-fewshot-num": 1,
    },
    # AutoGen-GMemory
    ("AutoGen-GMemory", "MATH"): {
        "successful-topk": 2,
        "failed-topk": 1,
        "insights-topk": 10,
        "gmemory-fidelity": "legacy",
        # --gmemory-snapshot-every is overridden by the launcher to 999999 so
        # no snapshot writes happen during read-only holdout eval.
    },
    # AutoGen-GMemory × APIBench-HF — match the user's canonical GMemory
    # holdout setting (succ=2, fail=1, ins=10) used across tasks. Note: prior
    # holdout (in `outputs/holdout/minimax-m2.7/APIBench-HF/AutoGen-GMemory/`)
    # was run with 1/1/3 from training memory_meta; needs re-run for fair
    # comparison.
    ("AutoGen-GMemory", "APIBench-HF"): {
        "successful-topk": 2,
        "failed-topk": 1,
        "insights-topk": 10,
        "gmemory-fidelity": "legacy",
    },
    # AutoGen-GMemory × HumanEval — match the user's prior inference-test
    # settings (succ=2, fail=1, ins=10), same family as MATH. HumanEval is
    # single-shot generation, so fidelity=legacy.
    ("AutoGen-GMemory", "HumanEval"): {
        "successful-topk": 2,
        "failed-topk": 1,
        "insights-topk": 10,
        "gmemory-fidelity": "legacy",
    },
    # AutoGen-GMemory × MATH500 — same canonical TopK family as MATH/HumanEval.
    # Single-shot math problems, fidelity=legacy.
    ("AutoGen-GMemory", "MATH500"): {
        "successful-topk": 2,
        "failed-topk": 1,
        "insights-topk": 10,
        "gmemory-fidelity": "legacy",
    },
    # AutoGen-GMemory × MMLU-Pro-Engineering — single-shot multiple choice,
    # fidelity=legacy, canonical TopK.
    ("AutoGen-GMemory", "MMLU-Pro-Engineering"): {
        "successful-topk": 2,
        "failed-topk": 1,
        "insights-topk": 10,
        "gmemory-fidelity": "legacy",
    },
    # AutoGen-GMemory × MMLU-Pro-Physics — same family as MMLU-Pro-Engineering.
    ("AutoGen-GMemory", "MMLU-Pro-Physics"): {
        "successful-topk": 2,
        "failed-topk": 1,
        "insights-topk": 10,
        "gmemory-fidelity": "legacy",
    },
    # AutoGen-GMemory × ALFWorld — training was step-wise (fidelity=high),
    # so holdout must also run fidelity=high. Shared ALFWorld flags match the
    # training sweep (run_alfworld_id_full_4methods.sh). TopKs match the
    # user's canonical GMemory holdout setting (succ=2, fail=1, ins=10).
    ("AutoGen-GMemory", "ALFWorld"): {
        "successful-topk": 2,
        "failed-topk": 1,
        "insights-topk": 10,
        "gmemory-fidelity": "high",
        "alfworld-max-steps": 30,
        "alfworld-fewshot-num": 1,
    },
    # Baseline × ALFWorld — no memory, step-wise rollout handled by
    # methods/baseline.py::run_trial. Only ALFWorld step cap is shared
    # (fewshot-num is irrelevant since Baseline doesn't use few-shots).
    ("Baseline", "ALFWorld"): {
        "alfworld-max-steps": 30,
    },
    # ---- OOD holdout (MATH500 → AIME / Physics) — clone MATH500 flags -----
    # Each method needs a (method, eval_task) entry because _build_main_cmd
    # reads TRAINING_FLAGS keyed by (method, args.task). Flags are copied
    # verbatim from the corresponding MATH500 entries above.
    ("HistoryRAG", "AIME2024"): {"top-k": 3},
    ("HistoryRAG", "AIME2025"): {"top-k": 3},
    ("HistoryRAG", "MMLU-Pro-Physics-OOD-MATH500"): {"top-k": 3},
    ("ExpRecent", "AIME2024"): {"top-k": 3},
    ("ExpRecent", "AIME2025"): {"top-k": 3},
    ("ExpRecent", "MMLU-Pro-Physics-OOD-MATH500"): {"top-k": 3},
    ("AllHistory", "AIME2024"): {"top-k": 10},
    ("AllHistory", "AIME2025"): {"top-k": 10},
    ("AllHistory", "MMLU-Pro-Physics-OOD-MATH500"): {"top-k": 10},
    ("AWM-Online", "AIME2024"): {"top-k": 3, "induce-steps": 1},
    ("AWM-Online", "AIME2025"): {"top-k": 3, "induce-steps": 1},
    ("AWM-Online", "MMLU-Pro-Physics-OOD-MATH500"): {"top-k": 3, "induce-steps": 1},
    ("DC-RS", "AIME2024"): {"top-k": 3},
    ("DC-RS", "AIME2025"): {"top-k": 3},
    ("DC-RS", "MMLU-Pro-Physics-OOD-MATH500"): {"top-k": 3},
    ("ExpeL-Online-MT", "AIME2024"): {"top-k": 3, "max-tries": 3, "batch-update-size": 8, "max-num-rules": 20},
    ("ExpeL-Online-MT", "AIME2025"): {"top-k": 3, "max-tries": 3, "batch-update-size": 8, "max-num-rules": 20},
    ("ExpeL-Online-MT", "MMLU-Pro-Physics-OOD-MATH500"): {"top-k": 3, "max-tries": 3, "batch-update-size": 8, "max-num-rules": 20},
    ("ExpeL-Online-ST", "AIME2024"): {"top-k": 3, "batch-update-size": 8, "max-num-rules": 20},
    ("ExpeL-Online-ST", "AIME2025"): {"top-k": 3, "batch-update-size": 8, "max-num-rules": 20},
    ("ExpeL-Online-ST", "MMLU-Pro-Physics-OOD-MATH500"): {"top-k": 3, "batch-update-size": 8, "max-num-rules": 20},
    ("AutoGen-GMemory", "AIME2024"): {
        "successful-topk": 2, "failed-topk": 1, "insights-topk": 10, "gmemory-fidelity": "legacy",
    },
    ("AutoGen-GMemory", "AIME2025"): {
        "successful-topk": 2, "failed-topk": 1, "insights-topk": 10, "gmemory-fidelity": "legacy",
    },
    ("AutoGen-GMemory", "MMLU-Pro-Physics-OOD-MATH500"): {
        "successful-topk": 2, "failed-topk": 1, "insights-topk": 10, "gmemory-fidelity": "legacy",
    },
    # ---- ExpeL-Online-MT-tries1: same flags as ExpeL-Online-MT, max-tries=1 -----
    # Holdout evaluates *memory quality* under single-shot generation, isolating
    # it from the multi-try retry mechanism. Loads the same training memory as
    # ExpeL-Online-MT (METHOD_TRAINING_FILENAME maps the alias back).
    ("ExpeL-Online-MT-tries1", "MATH"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
    ("ExpeL-Online-MT-tries1", "HumanEval"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
    ("ExpeL-Online-MT-tries1", "MATH500"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
    ("ExpeL-Online-MT-tries1", "APIBench-HF"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
    ("ExpeL-Online-MT-tries1", "MMLU-Pro-Physics"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
    ("ExpeL-Online-MT-tries1", "MMLU-Pro-Engineering"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
    ("ExpeL-Online-MT-tries1", "ALFWorld"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
        "alfworld-max-steps": 30, "alfworld-fewshot-num": 1,
    },
    ("ExpeL-Online-MT-tries1", "AIME2024"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
    ("ExpeL-Online-MT-tries1", "AIME2025"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
    ("ExpeL-Online-MT-tries1", "MMLU-Pro-Physics-OOD-MATH500"): {
        "top-k": 3, "max-tries": 1, "batch-update-size": 8, "max-num-rules": 20,
    },
}


# ---------------------------------------------------------------------------
# The method effectively used at training (for lookup into TRAINING_OUTPUT_DIR
# file names when --method is an alias). E.g., AllHistory really runs under
# the ExpRecent class/name during training, so its memory JSONL filename is
# MATH_ExpRecent_memory.jsonl.
# ---------------------------------------------------------------------------
METHOD_TRAINING_FILENAME: dict[str, str] = {
    "HistoryRAG": "HistoryRAG",
    "ExpRecent": "ExpRecent",
    "AllHistory": "ExpRecent",  # trained via ExpRecent@10 proxy
    "AWM-Online": "AWM-Online",
    "DC-RS": "DC-RS",
    "ExpeL-Online-MT": "ExpeL-Online-MT",
    # max-tries=1 alias loads the same training memory as canonical
    # ExpeL-Online-MT (filename {task}_ExpeL-Online-MT_memory.jsonl).
    "ExpeL-Online-MT-tries1": "ExpeL-Online-MT",
    "ExpeL-Online-ST": "ExpeL-Online-ST",
    "AutoGen-GMemory": "AutoGen-GMemory",
}


# ---------------------------------------------------------------------------
# Model → (OpenRouter id, embedding model, temperature, reasoning, max_new_tokens)
# ---------------------------------------------------------------------------
MODEL_PROFILES: dict[str, dict[str, str | float | int]] = {
    "qwen3-8b": {
        "generation_model": "openrouter/qwen/qwen3-8b",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "temperature": 0.7,
        "openrouter_reasoning": "off",
        "max_new_tokens": 2048,
    },
    "minimax-m2.7": {
        "generation_model": "openrouter/minimax/minimax-m2.7",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "temperature": 1.0,
        "openrouter_reasoning": "low",
        "max_new_tokens": 16483,
    },
}


# ---------------------------------------------------------------------------
# (model, task) → reasoning mode override. Applied on top of MODEL_PROFILES
# when the (model, task) pair is present. Qwen3-8B runs reasoning=off by
# default but needs reasoning=on for ALFWorld's multi-step planning, matching
# how the training sweep was run. MiniMax is already low everywhere so no
# per-task entry is needed.
# ---------------------------------------------------------------------------
MODEL_TASK_REASONING_OVERRIDE: dict[tuple[str, str], str] = {
    ("qwen3-8b", "ALFWorld"): "on",
}


# ---------------------------------------------------------------------------
# (model, task) → max_new_tokens override. Applied on top of MODEL_PROFILES
# when present. ALFWorld with reasoning=on needs a much larger budget than
# the single-shot default (Qwen3-8B reasoning traces + JSON action can run
# long; user requested 8192).
# ---------------------------------------------------------------------------
MODEL_TASK_MAX_NEW_TOKENS_OVERRIDE: dict[tuple[str, str], int] = {
    ("qwen3-8b", "ALFWorld"): 8192,
}


# ---------------------------------------------------------------------------
# Method → load class: "A" (seed memory.jsonl + restore), "B" (GMemory disk
# snapshot), "C" (needs patched restore_state_from_memory; same flow as A).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# BWT (Backward Transfer) experiment — Phase 1.
# Per-task t value sweep and training-source sample count N. The BWT formula
#   B(t) = Σ_τ [Acc(x_τ, s_{τ+t}) − Acc(x_τ, s_τ)]
# requires τ + t ≤ N. For HumanEval the holdout is HumanEval/132..163, so we
# cap N=132 to avoid leakage even though some training memory.jsonl files
# extend to 164. MATH500 trains over the full 500 samples.
# ---------------------------------------------------------------------------
BWT_T_VALUES: dict[str, list[int]] = {
    "HumanEval": [1, 2, 4, 8, 16, 32],
    "MATH500": [1, 2, 4, 8, 16, 32, 64, 128],
    "ALFWorld": [1, 2, 4, 8, 16, 32],
}

BWT_TRAIN_N: dict[str, int] = {
    "HumanEval": 132,
    "MATH500": 500,
    "ALFWorld": 140,
}

# Source dataset (parquet/jsonl/json) used during training, indexed in τ order.
# BWT uses these to materialize single-sample jsonls per τ.
BWT_TRAIN_SOURCE: dict[str, Path] = {
    "HumanEval": ROOT / "data" / "HumanEval" / "test.parquet",
    "MATH500": ROOT / "data" / "MATH500" / "test-2.jsonl",
    # ALFWorld training used --alfworld-split id (valid_seen, 140 entries).
    "ALFWorld": ROOT / "data" / "alfworld" / "alfworld_tasks_suffix_indist.json",
}


METHOD_LOAD_CLASS: dict[str, str] = {
    "HistoryRAG": "A",
    "ExpRecent": "A",
    "AllHistory": "A",
    "AWM-Online": "A",
    "DC-RS": "A",
    "ExpeL-Online-MT": "C",
    "ExpeL-Online-MT-tries1": "C",
    "ExpeL-Online-ST": "C",
    "AutoGen-GMemory": "B",
    # Class D — no memory load, no snapshot. Used for t=0 base-LLM baseline.
    # run.py forces a single step=0 and skips all work-dir prep.
    "Baseline": "D",
}
