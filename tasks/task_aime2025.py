from tasks.task_aime2024 import AIME2024Task


class AIME2025Task(AIME2024Task):
    name = "AIME2025"
    prompt_key = "aime2025"
    _default_year = 2025

    def __init__(
        self,
        prompt_file=None,
        hf_path="MathArena/aime_2025",
        split="train",
        data_file=None,
    ):
        super().__init__(
            prompt_file=prompt_file,
            hf_path=hf_path,
            split=split,
            data_file=data_file,
        )
