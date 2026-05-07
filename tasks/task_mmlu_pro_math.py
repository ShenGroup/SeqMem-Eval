from tasks.task_mmlu_pro import MMLUProTask


class MMLUProMathTask(MMLUProTask):
    name = "MMLU-Pro-Math"
    prompt_key = "mmlu_pro_math"
    _category = "math"
