from tasks.task_mmlu_pro import MMLUProTask


class MMLUProEngineeringTask(MMLUProTask):
    name = "MMLU-Pro-Engineering"
    prompt_key = "mmlu_pro_engineering"
    _category = "engineering"
