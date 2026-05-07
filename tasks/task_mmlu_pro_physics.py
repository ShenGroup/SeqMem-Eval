from tasks.task_mmlu_pro import MMLUProTask


class MMLUProPhysicsTask(MMLUProTask):
    name = "MMLU-Pro-Physics"
    prompt_key = "mmlu_pro_physics"
    _category = "physics"
