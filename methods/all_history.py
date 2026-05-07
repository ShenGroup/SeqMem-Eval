from methods.base_method import BaseMethod
from utils.llm_client import generate as llm_generate


class AllHistory(BaseMethod):
    name = "AllHistory"

    def __init__(
        self,
        top_k=None,
        generation_model_name="Qwen/Qwen3-4B-Instruct-2507",
        temperature=0.0,
        max_new_tokens=1024,
        **kwargs,
    ):
        super().__init__()
        self.top_k = top_k
        self.generation_model_name = generation_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def retrieve_memory(self, task, **kwargs):
        task_name = task.name
        rows = [row for row in self.memory if row.get("task") == task_name]
        if self.top_k is not None and self.top_k > 0:
            return rows[-self.top_k :]
        return rows

    def generate(self, task, prompt, **kwargs):
        retrieved = kwargs.get("memory", [])
        generation_prompt = self.build_memory_augmented_prompt(prompt, retrieved)
        return llm_generate(
            model_name=self.generation_model_name,
            prompt=generation_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

    def update_memory(self, task, output, **kwargs):
        record = kwargs.get("record")
        if not record:
            return
        retrieved = kwargs.get("memory") or []
        enriched = {"task": task.name, **record}
        self.memory.append(enriched)
        enriched["memory_qids"] = [
            r.get("qid") for r in self.memory if r.get("task") == task.name
        ]
        enriched["retrieved_qids"] = [r.get("qid") for r in retrieved]

    def reset_task_state(self, task):
        return None
