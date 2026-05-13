TASK_REGISTRY = {}

try:
    from tasks.task_alfworld import ALFWorldTask

    TASK_REGISTRY[ALFWorldTask.name] = ALFWorldTask
except ImportError:
    ALFWorldTask = None

try:
    from tasks.task_aime2024 import AIME2024Task

    TASK_REGISTRY[AIME2024Task.name] = AIME2024Task
except ImportError:
    AIME2024Task = None

try:
    from tasks.task_aime2025 import AIME2025Task

    TASK_REGISTRY[AIME2025Task.name] = AIME2025Task
except ImportError:
    AIME2025Task = None

try:
    from tasks.task_math500 import MATH500Task

    TASK_REGISTRY[MATH500Task.name] = MATH500Task
except ImportError:
    MATH500Task = None

try:
    from tasks.task_mmlu_pro_math import MMLUProMathTask

    TASK_REGISTRY[MMLUProMathTask.name] = MMLUProMathTask
except ImportError:
    MMLUProMathTask = None

try:
    from tasks.task_mmlu_pro_physics import MMLUProPhysicsTask

    TASK_REGISTRY[MMLUProPhysicsTask.name] = MMLUProPhysicsTask
    # OOD alias used by the holdout harness when the eval is MMLU-Pro-Physics
    # but memory comes from a different (training) task such as MATH500. The
    # task body is identical; the alias only exists to give configs/holdout.py
    # a separate (model, task) key that doesn't clobber the in-distribution
    # MMLU-Pro-Physics holdout configuration.
    TASK_REGISTRY["MMLU-Pro-Physics-OOD-MATH500"] = MMLUProPhysicsTask
except ImportError:
    MMLUProPhysicsTask = None

try:
    from tasks.task_mmlu_pro_engineering import MMLUProEngineeringTask

    TASK_REGISTRY[MMLUProEngineeringTask.name] = MMLUProEngineeringTask
except ImportError:
    MMLUProEngineeringTask = None

try:
    from tasks.task_humaneval import HumanEvalTask

    TASK_REGISTRY[HumanEvalTask.name] = HumanEvalTask
except ImportError:
    HumanEvalTask = None

try:
    from tasks.task_apibench import APIBenchHF

    TASK_REGISTRY[APIBenchHF.name] = APIBenchHF
except ImportError:
    APIBenchHF = None

try:
    from tasks.task_apibench import APIBenchTF

    TASK_REGISTRY[APIBenchTF.name] = APIBenchTF
except ImportError:
    APIBenchTF = None

try:
    from tasks.task_apibench import APIBenchTH

    TASK_REGISTRY[APIBenchTH.name] = APIBenchTH
except ImportError:
    APIBenchTH = None

__all__ = [
    "ALFWorldTask",
    "AIME2024Task",
    "AIME2025Task",
    "MATH500Task",
    "MMLUProMathTask",
    "MMLUProPhysicsTask",
    "MMLUProEngineeringTask",
    "HumanEvalTask",
    "APIBenchHF",
    "APIBenchTF",
    "APIBenchTH",
    "TASK_REGISTRY",
]
